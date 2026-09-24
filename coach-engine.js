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

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "onnx-community/Florence-2-base-ft";
const TRANSFORMERS_CACHE = "transformers-cache";
const READY_KEY = "doyen_model_ready";
const HOST_CANDIDATES = [
  "https://huggingface.co",
  "https://hf-mirror.com",
];

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

function applyRemoteHost(host) {
  if (!host) return;
  env.remoteHost = host.replace(/\/$/, "");
}

function resolvePreferredHost() {
  try {
    if (typeof window !== "undefined" && window.__DOYEN_HF_HOST__) {
      return String(window.__DOYEN_HF_HOST__);
    }
    if (typeof localStorage !== "undefined") {
      const saved = localStorage.getItem("doyen_hf_host");
      if (saved) return saved;
    }
  } catch (_) {}
  return HOST_CANDIDATES[0];
}

applyRemoteHost(resolvePreferredHost());

export function getModelHosts() {
  return HOST_CANDIDATES.slice();
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
 * 若浏览器 Cache API 里已有完整模型，返回对应 host（优先已保存源）
 */
export async function findCachedModelHost() {
  const ordered = [];
  const preferred = resolvePreferredHost();
  if (preferred) ordered.push(preferred);
  HOST_CANDIDATES.forEach(function (h) {
    if (ordered.indexOf(h) < 0) ordered.push(h);
  });

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
 */
async function probeSingleHost(host) {
  const ctrl =
    typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctrl
    ? setTimeout(function () {
        ctrl.abort();
      }, 8000)
    : null;
  try {
    const url =
      host.replace(/\/$/, "") + "/" + MODEL_ID + "/resolve/main/config.json";
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

export async function probeModelHosts(onProgress) {
  const results = [];
  for (let i = 0; i < HOST_CANDIDATES.length; i++) {
    const host = HOST_CANDIDATES[i];
    if (onProgress) {
      onProgress({ status: "loading", data: "探测 " + host + "…" });
    }
    const ok = await probeSingleHost(host);
    if (ok) {
      results.push({ host: host, ok: true, detail: "可访问" });
      setModelHost(host);
      return { ok: true, host: host, results: results };
    }
    results.push({ host: host, ok: false, detail: "不可用" });
  }
  return { ok: false, host: null, results: results };
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
      // 无完整缓存时再探测可用源（并优先保留上次成功的源）
      notify(onProgress, { status: "loading", data: "选择可用模型源…" });
      const preferred = resolvePreferredHost();
      let hostProbe = null;

      // 先试上次成功的源，避免每次被 huggingface.co 抢走导致换源重下
      if (preferred) {
        const preferredOk = await probeSingleHost(preferred);
        if (preferredOk) {
          hostProbe = { ok: true, host: preferred };
        }
      }
      if (!hostProbe || !hostProbe.ok) {
        hostProbe = await probeModelHosts(onProgress);
      }
      if (!hostProbe.ok) {
        throw new Error(
          "无法访问模型源（huggingface.co / hf-mirror.com 均失败）。请换网络或开代理后重试。"
        );
      }
      setModelHost(hostProbe.host);
      notify(onProgress, {
        status: "loading",
        data: "使用模型源 " + hostProbe.host + "（首次需下载）",
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
