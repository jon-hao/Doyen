/**
 * 主线程导演引擎（避免 iOS Safari module Worker “Load failed”）
 */
import {
  env,
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
  full,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/+esm";
import {
  installResumableFetch,
  onResumeProgress,
  clearResumePartials,
  listResumePartials,
} from "./resume-download.js?v=20260924-host";

env.allowLocalModels = false;
env.useBrowserCache = true;

// 全程开启：刷新后再次 from_pretrained 时自动 Range 续传
installResumableFetch();

export { clearResumePartials, listResumePartials };

const MODEL_ID = "onnx-community/Florence-2-base-ft";
const TRANSFORMERS_CACHE = "transformers-cache";
const READY_KEY = "doyen_model_ready";
const HOST_PREF_KEY = "doyen_hf_pref"; // auto | global | mirror
const GLOBAL_HOST = "https://huggingface.co";
const MIRROR_HOST = "https://hf-mirror.com";
const HOST_CANDIDATES = [GLOBAL_HOST, MIRROR_HOST];

// 判定本地是否已下齐：配置 + tokenizer + 本项目 dtype 对应的 onnx
const REQUIRED_FILES = [
  "config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "preprocessor_config.json",
];
const ONNX_FILE_GROUPS = [
  [
    "onnx/decoder_model_merged_q4.onnx",
    "onnx/decoder_model_merged_q4f16.onnx",
  ],
  ["onnx/encoder_model_q4.onnx", "onnx/encoder_model_q4f16.onnx"],
  [
    "onnx/embed_tokens_fp16.onnx",
    "onnx/embed_tokens.onnx",
    "onnx/embed_tokens_fp32.onnx",
  ],
  [
    "onnx/vision_encoder_fp16.onnx",
    "onnx/vision_encoder.onnx",
    "onnx/vision_encoder_fp32.onnx",
  ],
];

function normalizeHost(host) {
  return String(host || "").replace(/\/$/, "");
}

function isMirrorHost(host) {
  return /hf-mirror\.com/i.test(normalizeHost(host));
}

function isGlobalHost(host) {
  return /huggingface\.co/i.test(normalizeHost(host));
}

function hostLabel(host) {
  const h = normalizeHost(host);
  if (isMirrorHost(h)) return "国内镜像 (hf-mirror)";
  if (isGlobalHost(h)) return "全球源 (Hugging Face)";
  return h.replace(/^https?:\/\//, "") || "未知";
}

function applyRemoteHost(host) {
  if (!host) return;
  env.remoteHost = normalizeHost(host);
}

export function getHostPreference() {
  try {
    if (typeof localStorage !== "undefined") {
      const pref = localStorage.getItem(HOST_PREF_KEY);
      if (pref === "global" || pref === "mirror" || pref === "auto") return pref;
    }
  } catch (_) {}
  return "auto";
}

export function setHostPreference(pref) {
  const next =
    pref === "global" || pref === "mirror" || pref === "auto" ? pref : "auto";
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(HOST_PREF_KEY, next);
    }
  } catch (_) {}
  return next;
}

function resolvePreferredHost() {
  try {
    if (typeof window !== "undefined" && window.__DOYEN_HF_HOST__) {
      return normalizeHost(window.__DOYEN_HF_HOST__);
    }
    if (typeof localStorage !== "undefined") {
      const saved = localStorage.getItem("doyen_hf_host");
      if (saved) return normalizeHost(saved);
    }
  } catch (_) {}
  const pref = getHostPreference();
  if (pref === "mirror") return MIRROR_HOST;
  if (pref === "global") return GLOBAL_HOST;
  return GLOBAL_HOST;
}

applyRemoteHost(resolvePreferredHost());

export function getModelHosts() {
  return HOST_CANDIDATES.slice();
}

export function getGlobalHost() {
  return GLOBAL_HOST;
}

export function getMirrorHost() {
  return MIRROR_HOST;
}

export function describeHost(host) {
  return hostLabel(host || env.remoteHost);
}

export function setModelHost(host) {
  applyRemoteHost(host);
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("doyen_hf_host", env.remoteHost);
    }
    if (typeof window !== "undefined") {
      window.__DOYEN_HF_HOST__ = env.remoteHost;
    }
  } catch (_) {}
}

export function getModelHost() {
  return env.remoteHost;
}

/** 清空内存中的模型，切源后必须调用 */
export function resetCoachRuntime() {
  processorPromise = null;
  tokenizerPromise = null;
  modelPromise = null;
  warmed = false;
  deviceUsed = null;
}

function modelFileUrl(host, file) {
  return (
    String(host).replace(/\/$/, "") +
    "/" +
    MODEL_ID +
    "/resolve/main/" +
    file
  );
}

async function openTransformersCache() {
  if (typeof caches === "undefined" || !caches.open) return null;
  try {
    return await caches.open(TRANSFORMERS_CACHE);
  } catch (_) {
    return null;
  }
}

async function cacheHasFile(cache, host, file) {
  if (!cache) return false;
  const url = modelFileUrl(host, file);
  try {
    const hit = await cache.match(url);
    return !!(hit && (hit.ok || hit.type === "opaque"));
  } catch (_) {
    return false;
  }
}

async function hostHasCompleteModel(host) {
  const cache = await openTransformersCache();
  if (!cache) return false;

  for (let i = 0; i < REQUIRED_FILES.length; i++) {
    if (!(await cacheHasFile(cache, host, REQUIRED_FILES[i]))) return false;
  }

  for (let g = 0; g < ONNX_FILE_GROUPS.length; g++) {
    const group = ONNX_FILE_GROUPS[g];
    let ok = false;
    for (let i = 0; i < group.length; i++) {
      if (await cacheHasFile(cache, host, group[i])) {
        ok = true;
        break;
      }
    }
    if (!ok) return false;
  }
  return true;
}

/**
 * 若浏览器 Cache API 里已有完整模型，返回对应 host
 * （尊重用户源偏好：强制全球/国内时不串用另一源的缓存）
 */
export async function findCachedModelHost() {
  const pref = getHostPreference();
  const ordered = [];

  function push(h) {
    const n = normalizeHost(h);
    if (!n || ordered.indexOf(n) >= 0) return;
    ordered.push(n);
  }

  if (pref === "global") {
    push(GLOBAL_HOST);
  } else if (pref === "mirror") {
    push(MIRROR_HOST);
  } else {
    push(resolvePreferredHost());
    HOST_CANDIDATES.forEach(push);
  }

  for (let i = 0; i < ordered.length; i++) {
    const host = ordered[i];
    if (await hostHasCompleteModel(host)) {
      return { ok: true, host: host };
    }
  }
  return { ok: false, host: null };
}

function markModelReady(host, device) {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(
      READY_KEY,
      JSON.stringify({
        host: host,
        device: device || "",
        model: MODEL_ID,
        at: Date.now(),
      })
    );
  } catch (_) {}
}

export function clearModelReadyFlag() {
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.removeItem(READY_KEY);
    }
  } catch (_) {}
}

export function readModelReadyFlag() {
  try {
    if (typeof localStorage === "undefined") return null;
    const raw = localStorage.getItem(READY_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.model !== MODEL_ID) return null;
    return parsed;
  } catch (_) {
    return null;
  }
}

/**
 * 探测哪个模型源可从当前浏览器 CORS 访问
 * @param {string} host
 * @param {number} [timeoutMs]
 */
async function probeSingleHost(host, timeoutMs) {
  const ms = typeof timeoutMs === "number" ? timeoutMs : 8000;
  const ctrl =
    typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctrl
    ? setTimeout(function () {
        ctrl.abort();
      }, ms)
    : null;
  try {
    const url =
      normalizeHost(host) + "/" + MODEL_ID + "/resolve/main/config.json";
    const res = await fetch(url, {
      method: "GET",
      mode: "cors",
      cache: "no-store",
      signal: ctrl ? ctrl.signal : undefined,
    });
    if (timer) clearTimeout(timer);
    return !!(res && res.ok);
  } catch (_) {
    if (timer) clearTimeout(timer);
    return false;
  }
}

function buildProbeOrder() {
  const pref = getHostPreference();
  const saved = resolvePreferredHost();
  const order = [];

  function push(h) {
    const n = normalizeHost(h);
    if (!n || order.indexOf(n) >= 0) return;
    order.push(n);
  }

  if (pref === "mirror") {
    push(MIRROR_HOST);
    push(GLOBAL_HOST);
  } else if (pref === "global") {
    push(GLOBAL_HOST);
    push(MIRROR_HOST);
  } else {
    // auto：先快速试全球源，不通立刻降级国内镜像
    push(GLOBAL_HOST);
    push(MIRROR_HOST);
    // 若上次成功是镜像，也确保镜像在列表里（已有）
    push(saved);
  }
  return order;
}

/**
 * 探测可用模型源。auto 模式下 Hugging Face 仅给短超时，失败则降级 hf-mirror。
 */
export async function probeModelHosts(onProgress) {
  const order = buildProbeOrder();
  const pref = getHostPreference();
  const results = [];

  for (let i = 0; i < order.length; i++) {
    const host = order[i];
    const isGlobal = isGlobalHost(host);
    // 全球源在 auto 下用短超时，避免国内傻等
    const timeoutMs =
      pref === "auto" && isGlobal ? 3500 : isGlobal ? 8000 : 8000;

    if (onProgress) {
      onProgress({
        status: "loading",
        data:
          "探测 " +
          hostLabel(host) +
          (pref === "auto" && isGlobal ? "（超时将用国内镜像）" : "") +
          "…",
      });
    }

    const ok = await probeSingleHost(host, timeoutMs);
    if (ok) {
      results.push({ host: host, ok: true, detail: "可访问" });
      setModelHost(host);
      const degraded = pref === "auto" && isMirrorHost(host) && isGlobalHost(order[0]);
      return {
        ok: true,
        host: host,
        results: results,
        degraded: !!degraded,
        label: hostLabel(host),
      };
    }
    results.push({
      host: host,
      ok: false,
      detail: isGlobal && pref === "auto" ? "超时/不可达，将降级" : "不可用",
    });
  }
  return { ok: false, host: null, results: results, degraded: false };
}

/**
 * 清除指定 host 在 Cache API 中的模型文件
 */
export async function clearHostModelCache(host) {
  const target = normalizeHost(host);
  let deleted = 0;
  const cache = await openTransformersCache();
  if (cache && cache.keys) {
    const keys = await cache.keys();
    for (let i = 0; i < keys.length; i++) {
      const req = keys[i];
      const url = (req && req.url) || String(req);
      if (url.indexOf(target) >= 0) {
        try {
          await cache.delete(req);
          deleted += 1;
        } catch (_) {}
      }
    }
  }

  // 清该源的续传半成品
  try {
    const partials = await listResumePartials();
    for (let i = 0; i < partials.length; i++) {
      const row = partials[i];
      if (row && row.url && row.url.indexOf(target) >= 0) {
        // listResumePartials 无单条删除，整体 clear 太狠；用 IDB 直接删
      }
    }
  } catch (_) {}

  try {
    const dbName = "doyen-resume-v1";
    await new Promise(function (resolve) {
      const open = indexedDB.open(dbName, 1);
      open.onerror = function () {
        resolve();
      };
      open.onupgradeneeded = function () {
        const db = open.result;
        if (!db.objectStoreNames.contains("partials")) {
          db.createObjectStore("partials", { keyPath: "url" });
        }
      };
      open.onsuccess = function () {
        const db = open.result;
        try {
          const tx = db.transaction("partials", "readwrite");
          const store = tx.objectStore("partials");
          const req = store.openCursor();
          req.onsuccess = function () {
            const cursor = req.result;
            if (!cursor) return;
            const val = cursor.value;
            if (val && val.url && String(val.url).indexOf(target) >= 0) {
              cursor.delete();
            }
            cursor.continue();
          };
          tx.oncomplete = function () {
            db.close();
            resolve();
          };
          tx.onerror = function () {
            db.close();
            resolve();
          };
        } catch (_) {
          db.close();
          resolve();
        }
      };
    });
  } catch (_) {}

  const ready = readModelReadyFlag();
  if (ready && normalizeHost(ready.host) === target) {
    clearModelReadyFlag();
  }

  return { deleted: deleted, host: target };
}

/**
 * 切换到全球源：清掉国内镜像缓存，强制用 huggingface.co
 */
export async function switchToGlobalSource() {
  setHostPreference("global");
  const cleared = await clearHostModelCache(MIRROR_HOST);
  setModelHost(GLOBAL_HOST);
  resetCoachRuntime();
  clearModelReadyFlag();
  return {
    host: GLOBAL_HOST,
    label: hostLabel(GLOBAL_HOST),
    clearedMirrorFiles: cleared.deleted,
  };
}

/**
 * 切换到国内镜像：清掉全球源缓存，强制用 hf-mirror
 */
export async function switchToMirrorSource() {
  setHostPreference("mirror");
  const cleared = await clearHostModelCache(GLOBAL_HOST);
  setModelHost(MIRROR_HOST);
  resetCoachRuntime();
  clearModelReadyFlag();
  return {
    host: MIRROR_HOST,
    label: hostLabel(MIRROR_HOST),
    clearedGlobalFiles: cleared.deleted,
  };
}

/**
 * 恢复自动：全球源短超时失败则降级镜像
 */
export function switchToAutoSource() {
  setHostPreference("auto");
  return { pref: "auto" };
}

let processorPromise = null;
let tokenizerPromise = null;
let modelPromise = null;
let deviceUsed = null;
let warmed = false;

async function canWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (_) {
    return false;
  }
}

async function canFp16() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!(adapter && adapter.features && adapter.features.has("shader-f16"));
  } catch (_) {
    return false;
  }
}

function notify(onProgress, payload) {
  if (typeof onProgress === "function") onProgress(payload);
}

async function loadModelOnDevice(device, onProgress, fromCache) {
  const fp16 = device === "webgpu" ? await canFp16() : false;
  notify(onProgress, {
    status: "loading",
    data:
      device === "webgpu"
        ? fromCache
          ? "从本地加载模型（WebGPU）…"
          : "使用 WebGPU 加载模型…"
        : fromCache
          ? "从本地加载模型（WASM）…"
          : "使用 WASM 加载模型（较慢）…",
    fromCache: !!fromCache,
  });

  return Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: {
      embed_tokens: fp16 ? "fp16" : "fp32",
      vision_encoder: fp16 ? "fp16" : "fp32",
      encoder_model: "q4",
      decoder_model_merged: "q4",
    },
    device: device,
    progress_callback: function (info) {
      notify(
        onProgress,
        Object.assign({ channel: "hf-progress", fromCache: !!fromCache }, info || {})
      );
    },
  });
}

export async function loadCoach(onProgress, options) {
  const opts = options || {};
  const forceRemote = !!opts.forceRemote;

  if (modelPromise && warmed) {
    notify(onProgress, { status: "ready", device: deviceUsed, fromCache: true });
    return { device: deviceUsed, host: env.remoteHost, fromCache: true };
  }

  let fromCache = false;
  const prevAllowRemote = env.allowRemoteModels;
  const stopResumeListen = onResumeProgress(function (info) {
    if (!info || info.status !== "resume") return;
    const pct =
      info.total > 0
        ? Math.min(99, Math.round((info.loaded / info.total) * 100))
        : 0;
    notify(onProgress, {
      status: "loading",
      data:
        "继续下载 " +
        (info.file || "模型") +
        (pct ? "（已完成 " + pct + "%）" : "…"),
      resumed: true,
      fromCache: false,
    });
  });

  try {
    let cached = { ok: false, host: null };
    if (!forceRemote) {
      notify(onProgress, { status: "loading", data: "检查本地模型缓存…" });
      cached = await findCachedModelHost();
    }

    if (cached.ok) {
      fromCache = true;
      setModelHost(cached.host);
      // 已下齐：禁止再走远程，强制走 Cache API
      env.allowRemoteModels = false;
      notify(onProgress, {
        status: "loading",
        data: "本地已有模型，正在加载…",
        fromCache: true,
      });
    } else {
      // 无完整缓存：探测可用源；HF 不通时自动降级国内镜像
      notify(onProgress, { status: "loading", data: "选择可用模型源…" });
      const hostProbe = await probeModelHosts(onProgress);
      if (!hostProbe.ok) {
        throw new Error(
          "无法访问模型源（Hugging Face / hf-mirror 均失败）。国内可改用镜像；海外请开代理后点「改用全球源」。"
        );
      }
      setModelHost(hostProbe.host);
      notify(onProgress, {
        status: "loading",
        data: hostProbe.degraded
          ? "Hugging Face 不可达，已降级 " +
            hostProbe.label +
            "（首次需下载）"
          : "使用" + hostProbe.label + "（首次需下载）",
      });
    }

    notify(onProgress, {
      status: "loading",
      data: fromCache ? "从本地加载处理器…" : "下载处理器…",
      fromCache: fromCache,
    });
    processorPromise =
      processorPromise ||
      AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: function (info) {
          notify(
            onProgress,
            Object.assign({ channel: "hf-progress", fromCache: fromCache }, info || {})
          );
        },
      });

    notify(onProgress, {
      status: "loading",
      data: fromCache ? "从本地加载分词器…" : "下载分词器…",
      fromCache: fromCache,
    });
    tokenizerPromise =
      tokenizerPromise ||
      AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: function (info) {
          notify(
            onProgress,
            Object.assign({ channel: "hf-progress", fromCache: fromCache }, info || {})
          );
        },
      });

    const preferGpu = await canWebGPU();
    if (!modelPromise) {
      if (preferGpu) {
        try {
          modelPromise = loadModelOnDevice("webgpu", onProgress, fromCache);
          await modelPromise;
          deviceUsed = "webgpu";
        } catch (gpuErr) {
          modelPromise = null;
          // 本地缓存加载 WebGPU 失败时，允许再试 WASM（仍可走缓存）
          notify(onProgress, {
            status: "loading",
            data:
              "WebGPU 失败，改试 WASM：" +
              ((gpuErr && gpuErr.message) || "unknown"),
            fromCache: fromCache,
          });
          modelPromise = loadModelOnDevice("wasm", onProgress, fromCache);
          await modelPromise;
          deviceUsed = "wasm";
        }
      } else {
        modelPromise = loadModelOnDevice("wasm", onProgress, fromCache);
        await modelPromise;
        deviceUsed = "wasm";
      }
    }

    const model = await modelPromise;
    const tokenizer = await tokenizerPromise;
    await processorPromise;

    if (!warmed) {
      notify(onProgress, {
        status: "loading",
        data: "编译/预热模型…",
        fromCache: fromCache,
      });
      const text_inputs = tokenizer("a");
      const pixel_values = full([1, 3, 768, 768], 0.0);
      await model.generate({
        ...text_inputs,
        pixel_values,
        max_new_tokens: 1,
      });
      warmed = true;
    }

    markModelReady(env.remoteHost, deviceUsed);
    notify(onProgress, {
      status: "ready",
      device: deviceUsed,
      fromCache: fromCache,
    });
    return {
      device: deviceUsed,
      host: env.remoteHost,
      fromCache: fromCache,
    };
  } catch (err) {
    // 若强制本地失败（缓存不完整），清标记并改为联网下载（只重试一次）
    if (fromCache && !forceRemote) {
      env.allowRemoteModels = true;
      clearModelReadyFlag();
      processorPromise = null;
      tokenizerPromise = null;
      modelPromise = null;
      warmed = false;
      deviceUsed = null;
      notify(onProgress, {
        status: "loading",
        data: "本地缓存不完整，改为联网下载…",
      });
      return loadCoach(onProgress, { forceRemote: true });
    }

    processorPromise = null;
    tokenizerPromise = null;
    modelPromise = null;
    warmed = false;
    deviceUsed = null;
    const msg = (err && err.message) || "模型加载失败";
    throw new Error(msg);
  } finally {
    try {
      stopResumeListen();
    } catch (_) {}
    env.allowRemoteModels = prevAllowRemote;
  }
}

async function runTask(task, vision_inputs, image_size) {
  const model = await modelPromise;
  const tokenizer = await tokenizerPromise;
  const processor = await processorPromise;
  const prompts = processor.construct_prompts(task);
  const text_inputs = tokenizer(prompts);
  const generated_ids = await model.generate({
    ...text_inputs,
    ...vision_inputs,
    max_new_tokens: 128,
    num_beams: 1,
    do_sample: false,
  });
  const generated_text = tokenizer.batch_decode(generated_ids, {
    skip_special_tokens: false,
  })[0];
  return processor.post_process_generation(generated_text, task, image_size);
}

/**
 * @param {Blob|HTMLCanvasElement} source
 */
export async function analyzeFrame(source, onProgress) {
  if (!modelPromise || !warmed) {
    await loadCoach(onProgress);
  }

  notify(onProgress, { status: "loading", data: "正在理解画面…" });

  let image;
  if (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) {
    image = RawImage.fromCanvas(source);
  } else if (source && typeof Blob !== "undefined" && source instanceof Blob) {
    image = await RawImage.fromBlob(source);
  } else {
    throw new Error("无法读取当前画面");
  }

  const processor = await processorPromise;
  const image_size = image.size;
  const vision_inputs = await processor(image);

  const captionTask = "<MORE_DETAILED_CAPTION>";
  const odTask = "<OD>";
  const captionResult = await runTask(captionTask, vision_inputs, image_size);
  const odResult = await runTask(odTask, vision_inputs, image_size);

  const caption =
    (captionResult && captionResult[captionTask]) ||
    (typeof captionResult === "string" ? captionResult : "") ||
    "";

  const od = (odResult && odResult[odTask]) || {};
  const labels = od.labels || [];
  const bboxes = od.bboxes || [];
  const objects = labels.map(function (label, i) {
    return { label: label, bbox: bboxes[i] || null };
  });

  return {
    caption: String(caption || "").trim(),
    objects: objects,
    device: deviceUsed,
  };
}

export function getCoachDevice() {
  return deviceUsed;
}
