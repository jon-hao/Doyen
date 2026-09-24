/**
 * 轻量主体检测引擎（MediaPipe EfficientDet-Lite0，约 7MB）
 * 替代 Florence-2，显著降低 iPhone 内存占用。
 */
import {
  ObjectDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm";

const READY_KEY = "doyen_model_ready";
const MODEL_ID = "mediapipe/efficientdet_lite0_float16";
const MODEL_BYTES = 7244197; // 约 6.9MB，用于进度估算
const MP_VERSION = "0.10.18";
const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION + "/wasm";

let detector = null;
let warmed = false;
let deviceUsed = "wasm";
let loadPromise = null;

function notify(onProgress, payload) {
  if (typeof onProgress === "function") onProgress(payload);
}

function assetUrl(rel) {
  const pathName = location.pathname || "/";
  let dir;
  if (pathName.endsWith("/")) dir = pathName;
  else if (/\.html?$/i.test(pathName)) dir = pathName.replace(/[^/]+$/, "");
  else dir = pathName.replace(/\/?$/, "/");
  return dir + rel.replace(/^\//, "");
}

function modelAssetUrl() {
  return assetUrl("models/efficientdet_lite0.tflite");
}

function markModelReady() {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(
      READY_KEY,
      JSON.stringify({
        host: "local",
        device: deviceUsed,
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

/** 兼容旧调用：轻量模型无 HF 源概念 */
export function switchToAutoSource() {
  return { pref: "auto" };
}
export function setModelHost() {}
export function getModelHost() {
  return "local";
}
export function probeModelHosts() {
  return Promise.resolve({
    ok: true,
    host: "local",
    results: [{ host: "local", ok: true, detail: "bundled" }],
    degraded: false,
    label: "本地轻量模型",
  });
}
export async function findCachedModelHost() {
  if (detector && warmed) return { ok: true, host: "local" };
  const flag = readModelReadyFlag();
  if (flag) return { ok: true, host: "local" };
  // 同域模型文件视为「可本地加载」；是否已在 Cache 里由浏览器决定
  try {
    const res = await fetch(modelAssetUrl(), { method: "HEAD", cache: "force-cache" });
    if (res && res.ok) return { ok: true, host: "local" };
  } catch (_) {}
  return { ok: false, host: null };
}

export async function clearResumePartials() {}
export async function listResumePartials() {
  return [];
}
export async function hasIncompleteDownloads() {
  return false;
}
export async function estimateResumeProgress() {
  return { pct: 0, loaded: 0, total: MODEL_BYTES, files: 0 };
}
export async function flushActiveDownloads() {}
export function onResumeProgress() {
  return function () {};
}
export function resetCoachRuntime() {
  try {
    if (detector && typeof detector.close === "function") detector.close();
  } catch (_) {}
  detector = null;
  warmed = false;
  loadPromise = null;
  deviceUsed = "wasm";
}
export function getCoachDevice() {
  return deviceUsed;
}

async function fetchWithProgress(url, onProgress, label) {
  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) throw new Error("下载失败 HTTP " + res.status);

  const totalHeader = parseInt(res.headers.get("Content-Length") || "0", 10);
  const total = totalHeader > 0 ? totalHeader : MODEL_BYTES;

  if (!res.body || !res.body.getReader) {
    const buf = new Uint8Array(await res.arrayBuffer());
    notify(onProgress, {
      status: "progress",
      file: label,
      loaded: buf.byteLength,
      total: buf.byteLength,
    });
    return buf;
  }

  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    chunks.push(chunk.value);
    loaded += chunk.value.byteLength || 0;
    notify(onProgress, {
      status: "progress",
      file: label,
      loaded: loaded,
      total: total,
    });
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], offset);
    offset += chunks[i].byteLength;
  }
  notify(onProgress, {
    status: "done",
    file: label,
    loaded: out.byteLength,
    total: out.byteLength,
  });
  return out;
}

async function createDetector(onProgress) {
  notify(onProgress, { status: "loading", data: "加载检测运行时…" });
  notify(onProgress, {
    status: "initiate",
    file: "efficientdet_lite0.tflite",
    total: MODEL_BYTES,
  });

  const vision = await FilesetResolver.forVisionTasks(WASM_CDN);

  notify(onProgress, { status: "loading", data: "下载轻量检测模型…" });
  notify(onProgress, {
    status: "download",
    file: "efficientdet_lite0.tflite",
    total: MODEL_BYTES,
  });

  const modelBuffer = await fetchWithProgress(
    modelAssetUrl(),
    onProgress,
    "efficientdet_lite0.tflite"
  );

  notify(onProgress, { status: "loading", data: "初始化检测器…" });

  // iOS 优先 CPU，避开 WebGL/GPU 兼容坑；内存占用仍远低于 Florence
  let created;
  try {
    created = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "CPU",
      },
      scoreThreshold: 0.35,
      maxResults: 10,
      runningMode: "IMAGE",
    });
    deviceUsed = "wasm";
  } catch (cpuErr) {
    created = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "GPU",
      },
      scoreThreshold: 0.35,
      maxResults: 10,
      runningMode: "IMAGE",
    });
    deviceUsed = "webgl";
  }

  return created;
}

export async function loadCoach(onProgress) {
  if (detector && warmed) {
    notify(onProgress, { status: "ready", device: deviceUsed, fromCache: true });
    return { device: deviceUsed, host: "local", fromCache: true };
  }
  if (loadPromise) return loadPromise;

  loadPromise = (async function () {
    try {
      detector = await createDetector(onProgress);
      warmed = true;
      markModelReady();
      notify(onProgress, {
        status: "ready",
        device: deviceUsed,
        fromCache: false,
      });
      return { device: deviceUsed, host: "local", fromCache: false };
    } catch (err) {
      resetCoachRuntime();
      const raw = (err && err.message) || String(err || "");
      let msg = raw || "模型加载失败";
      if (/Load failed|fetch|network|Failed to fetch/i.test(raw)) {
        msg = "轻量模型加载失败，请检查网络后重试";
      } else if (/memory|OOM|Allocation/i.test(raw)) {
        msg = "内存不足，请关闭其他标签页后重试";
      }
      throw new Error(msg);
    } finally {
      loadPromise = null;
    }
  })();

  return loadPromise;
}

function canvasFromSource(source) {
  if (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) {
    return source;
  }
  throw new Error("无法读取当前画面");
}

/**
 * 检测画面主体，返回 { objects: [{label, bbox:[x1,y1,x2,y2], score}] }
 */
export async function detectSubjects(source, onProgress) {
  if (!detector || !warmed) {
    await loadCoach(onProgress);
  }
  const canvas = canvasFromSource(source);
  const result = detector.detect(canvas);
  const detections = (result && result.detections) || [];
  const objects = [];

  for (let i = 0; i < detections.length; i++) {
    const d = detections[i];
    const box = d.boundingBox;
    if (!box) continue;
    const cats = d.categories || [];
    const top = cats[0] || {};
    const x1 = Number(box.originX) || 0;
    const y1 = Number(box.originY) || 0;
    const x2 = x1 + (Number(box.width) || 0);
    const y2 = y1 + (Number(box.height) || 0);
    if (x2 - x1 < 2 || y2 - y1 < 2) continue;
    objects.push({
      label: String(top.categoryName || top.displayName || "object"),
      score: Number(top.score) || 0,
      bbox: [x1, y1, x2, y2],
    });
  }

  return {
    objects: objects,
    imageSize: [canvas.width, canvas.height],
    device: deviceUsed,
  };
}

/** 兼容旧接口 */
export async function analyzeFrame(source, onProgress) {
  const vision = await detectSubjects(source, onProgress);
  return {
    caption: "",
    objects: vision.objects,
    device: deviceUsed,
  };
}

/**
 * 优先人，否则置信度最高 / 面积最大
 */
export function pickPrimarySubject(objects) {
  const list = objects || [];
  if (!list.length) return null;
  const priority = [
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "human",
    "face",
    "people",
  ];
  for (let i = 0; i < list.length; i++) {
    const lab = String(list[i].label || "").toLowerCase();
    if (
      priority.some(function (p) {
        return lab.indexOf(p) >= 0;
      })
    ) {
      return list[i];
    }
  }
  let best = null;
  let bestScore = -1;
  for (let i = 0; i < list.length; i++) {
    const o = list[i];
    const b = o.bbox;
    if (!b || b.length < 4) continue;
    const area = Math.abs(b[2] - b[0]) * Math.abs(b[3] - b[1]);
    const score = (Number(o.score) || 0) * 1000 + area;
    if (score > bestScore) {
      bestScore = score;
      best = o;
    }
  }
  return best;
}
