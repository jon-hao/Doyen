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
const HOST_CANDIDATES = [
  "https://huggingface.co",
  "https://hf-mirror.com",
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

/**
 * 探测哪个模型源可从当前浏览器 CORS 访问
 */
export async function probeModelHosts(onProgress) {
  const results = [];
  for (let i = 0; i < HOST_CANDIDATES.length; i++) {
    const host = HOST_CANDIDATES[i];
    if (onProgress) {
      onProgress({ status: "loading", data: "探测 " + host + "…" });
    }
    const ctrl =
      typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl
      ? setTimeout(function () {
          ctrl.abort();
        }, 10000)
      : null;
    try {
      const url =
        host.replace(/\/$/, "") +
        "/" +
        MODEL_ID +
        "/resolve/main/config.json";
      const res = await fetch(url, {
        method: "GET",
        mode: "cors",
        cache: "no-store",
        signal: ctrl ? ctrl.signal : undefined,
      });
      if (timer) clearTimeout(timer);
      if (res.ok) {
        results.push({ host: host, ok: true, detail: "可访问" });
        setModelHost(host);
        return { ok: true, host: host, results: results };
      }
      results.push({
        host: host,
        ok: false,
        detail: "HTTP " + res.status,
      });
    } catch (err) {
      if (timer) clearTimeout(timer);
      const detail =
        err && err.name === "AbortError"
          ? "超时"
          : (err && err.message) || "Failed to fetch";
      results.push({ host: host, ok: false, detail: detail });
    }
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

async function loadModelOnDevice(device, onProgress) {
  const fp16 = device === "webgpu" ? await canFp16() : false;
  notify(onProgress, {
    status: "loading",
    data: device === "webgpu" ? "使用 WebGPU 加载模型…" : "使用 WASM 加载模型（较慢）…",
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
      notify(onProgress, Object.assign({ channel: "hf-progress" }, info || {}));
    },
  });
}

export async function loadCoach(onProgress) {
  if (modelPromise && warmed) {
    notify(onProgress, { status: "ready" });
    return { device: deviceUsed, host: env.remoteHost };
  }

  try {
    // 加载前再确认可用模型源
    notify(onProgress, { status: "loading", data: "选择可用模型源…" });
    const hostProbe = await probeModelHosts(onProgress);
    if (!hostProbe.ok) {
      throw new Error(
        "无法访问模型源（huggingface.co / hf-mirror.com 均失败）。请换网络或开代理后重试。"
      );
    }
    notify(onProgress, {
      status: "loading",
      data: "使用模型源 " + hostProbe.host,
    });

    notify(onProgress, { status: "loading", data: "下载处理器…" });
    processorPromise =
      processorPromise ||
      AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: function (info) {
          notify(onProgress, Object.assign({ channel: "hf-progress" }, info || {}));
        },
      });

    notify(onProgress, { status: "loading", data: "下载分词器…" });
    tokenizerPromise =
      tokenizerPromise ||
      AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: function (info) {
          notify(onProgress, Object.assign({ channel: "hf-progress" }, info || {}));
        },
      });

    const preferGpu = await canWebGPU();
    if (!modelPromise) {
      if (preferGpu) {
        try {
          modelPromise = loadModelOnDevice("webgpu", onProgress);
          await modelPromise;
          deviceUsed = "webgpu";
        } catch (gpuErr) {
          modelPromise = null;
          notify(onProgress, {
            status: "loading",
            data:
              "WebGPU 失败，改试 WASM：" +
              ((gpuErr && gpuErr.message) || "unknown"),
          });
          modelPromise = loadModelOnDevice("wasm", onProgress);
          await modelPromise;
          deviceUsed = "wasm";
        }
      } else {
        modelPromise = loadModelOnDevice("wasm", onProgress);
        await modelPromise;
        deviceUsed = "wasm";
      }
    }

    const model = await modelPromise;
    const tokenizer = await tokenizerPromise;
    await processorPromise;

    if (!warmed) {
      notify(onProgress, { status: "loading", data: "编译/预热模型…" });
      const text_inputs = tokenizer("a");
      const pixel_values = full([1, 3, 768, 768], 0.0);
      await model.generate({
        ...text_inputs,
        pixel_values,
        max_new_tokens: 1,
      });
      warmed = true;
    }

    notify(onProgress, { status: "ready", device: deviceUsed });
    return { device: deviceUsed, host: env.remoteHost };
  } catch (err) {
    processorPromise = null;
    tokenizerPromise = null;
    modelPromise = null;
    warmed = false;
    deviceUsed = null;
    const msg = (err && err.message) || "模型加载失败";
    throw new Error(msg);
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
