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
let videoTs = 0;
/** @type {null | {label:string, score:number, bbox:number[]}} */
let trackedSubject = null;

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

function modelCandidateUrls() {
  const local = assetUrl("models/efficientdet_lite0.tflite");
  return [
    local,
    "https://cdn.jsdelivr.net/gh/jon-hao/Doyen@main/models/efficientdet_lite0.tflite",
    "https://raw.githubusercontent.com/jon-hao/Doyen/main/models/efficientdet_lite0.tflite",
  ];
}

async function fetchWithProgress(url, onProgress, label, rangeStart, rangeEnd) {
  const start = typeof rangeStart === "number" ? rangeStart : 0;
  const end = typeof rangeEnd === "number" ? rangeEnd : 1;
  const span = Math.max(0.01, end - start);

  function report(loadedPart, totalPart) {
    const ratio = totalPart > 0 ? loadedPart / totalPart : 1;
    const overall = start + Math.min(1, Math.max(0, ratio)) * span;
    notify(onProgress, {
      status: "progress",
      file: label,
      loaded: Math.round(MODEL_BYTES * overall),
      total: MODEL_BYTES,
    });
  }

  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) throw new Error("下载失败 HTTP " + res.status);

  const totalHeader = parseInt(res.headers.get("Content-Length") || "0", 10);
  const total = totalHeader > 0 ? totalHeader : MODEL_BYTES;

  if (!res.body || !res.body.getReader) {
    const buf = new Uint8Array(await res.arrayBuffer());
    report(buf.byteLength, buf.byteLength);
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
    report(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], offset);
    offset += chunks[i].byteLength;
  }
  report(out.byteLength, out.byteLength);
  return out;
}

async function fetchModelBuffer(onProgress) {
  const urls = modelCandidateUrls();
  let lastErr = null;
  for (let i = 0; i < urls.length; i++) {
    try {
      return await fetchWithProgress(
        urls[i],
        onProgress,
        "efficientdet_lite0.tflite",
        0.15,
        0.9
      );
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("模型文件不可用");
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
  // 仅：内存已就绪，或本机曾经成功加载过（避免 HEAD 探测误判成「已下载」而不显示进度）
  if (detector && warmed) return { ok: true, host: "local" };
  if (readModelReadyFlag()) return { ok: true, host: "local" };
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
  videoTs = 0;
  trackedSubject = null;
}
export function getCoachDevice() {
  return deviceUsed;
}

function emitResourceProgress(onProgress, fraction) {
  const pct = Math.max(0, Math.min(0.99, Number(fraction) || 0));
  notify(onProgress, {
    status: "progress",
    file: "efficientdet_lite0.tflite",
    loaded: Math.round(MODEL_BYTES * pct),
    total: MODEL_BYTES,
  });
}

async function createDetector(onProgress) {
  notify(onProgress, {
    status: "initiate",
    file: "efficientdet_lite0.tflite",
    total: MODEL_BYTES,
  });
  notify(onProgress, {
    status: "download",
    file: "efficientdet_lite0.tflite",
    total: MODEL_BYTES,
  });
  emitResourceProgress(onProgress, 0.02);

  // 运行时 WASM（计入资源下载进度，避免长时间停在 0%）
  const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
  emitResourceProgress(onProgress, 0.15);

  const modelBuffer = await fetchModelBuffer(onProgress);
  emitResourceProgress(onProgress, 0.92);

  let created;
  try {
    created = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "CPU",
      },
      scoreThreshold: 0.25,
      maxResults: 10,
      runningMode: "VIDEO",
    });
    deviceUsed = "wasm";
  } catch (_) {
    created = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "GPU",
      },
      scoreThreshold: 0.25,
      maxResults: 10,
      runningMode: "VIDEO",
    });
    deviceUsed = "webgl";
  }

  videoTs = 0;
  trackedSubject = null;

  emitResourceProgress(onProgress, 0.99);
  notify(onProgress, {
    status: "done",
    file: "efficientdet_lite0.tflite",
    loaded: MODEL_BYTES,
    total: MODEL_BYTES,
  });
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

function isPersonLabel(label) {
  const lab = String(label || "").toLowerCase();
  return (
    lab.indexOf("person") >= 0 ||
    lab.indexOf("man") >= 0 ||
    lab.indexOf("woman") >= 0 ||
    lab.indexOf("boy") >= 0 ||
    lab.indexOf("girl") >= 0 ||
    lab.indexOf("human") >= 0 ||
    lab.indexOf("face") >= 0 ||
    lab.indexOf("people") >= 0
  );
}

function boxArea(b) {
  if (!b || b.length < 4) return 0;
  return Math.abs(b[2] - b[0]) * Math.abs(b[3] - b[1]);
}

function boxIoU(a, b) {
  if (!a || !b || a.length < 4 || b.length < 4) return 0;
  const ax1 = Math.min(a[0], a[2]);
  const ay1 = Math.min(a[1], a[3]);
  const ax2 = Math.max(a[0], a[2]);
  const ay2 = Math.max(a[1], a[3]);
  const bx1 = Math.min(b[0], b[2]);
  const by1 = Math.min(b[1], b[3]);
  const bx2 = Math.max(b[0], b[2]);
  const by2 = Math.max(b[1], b[3]);
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return uni > 0 ? inter / uni : 0;
}

function lerpBox(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
    a[3] + (b[3] - a[3]) * t,
  ];
}

/**
 * 摄影向主体分：面积占比 × 置信度 × 靠近画面中心
 * 人像加权；过滤过小/铺满全屏的噪声框
 */
function subjectSalience(obj, imgW, imgH) {
  const b = obj && obj.bbox;
  if (!b || b.length < 4) return -1;
  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const frameArea = w * h;
  const area = boxArea(b);
  const areaRatio = area / frameArea;
  if (areaRatio < 0.02) return -1;
  if (areaRatio > 0.92) return -1;

  const cx = (Math.min(b[0], b[2]) + Math.max(b[0], b[2])) / 2;
  const cy = (Math.min(b[1], b[3]) + Math.max(b[1], b[3])) / 2;
  const dx = (cx - w * 0.5) / w;
  const dy = (cy - h * 0.5) / h;
  const center = 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy) * 1.35);
  const conf = Math.max(0, Math.min(1, Number(obj.score) || 0));
  const personBoost = isPersonLabel(obj.label) ? 1.55 : 1;
  return (
    Math.sqrt(areaRatio) *
    (0.35 + 0.65 * conf) *
    (0.45 + 0.55 * center) *
    personBoost
  );
}

function detectionsToObjects(result) {
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
  return objects;
}

/**
 * 检测画面主体。VIDEO 模式 + 时序平滑，优先锁定人像主体。
 */
export async function detectSubjects(source, onProgress) {
  if (!detector || !warmed) {
    await loadCoach(onProgress);
  }
  const canvas = canvasFromSource(source);
  videoTs += 33;
  let result;
  try {
    result = detector.detectForVideo(canvas, videoTs);
  } catch (_) {
    result = detector.detect(canvas);
  }

  const objects = detectionsToObjects(result);
  const primaryRaw = pickPrimarySubject(objects, canvas.width, canvas.height);
  const primary = stabilizeSubject(primaryRaw, canvas.width, canvas.height);

  return {
    objects: objects,
    primary: primary,
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
    primary: vision.primary,
    device: deviceUsed,
  };
}

/**
 * 有人优先；多人时用「面积×置信度×居中」选最像拍摄主体的一个
 */
export function pickPrimarySubject(objects, imgW, imgH) {
  const list = objects || [];
  if (!list.length) return null;

  const people = [];
  const others = [];
  for (let i = 0; i < list.length; i++) {
    if (isPersonLabel(list[i].label)) people.push(list[i]);
    else others.push(list[i]);
  }
  const pool = people.length ? people : others;

  let best = null;
  let bestScore = -1;
  for (let i = 0; i < pool.length; i++) {
    const s = subjectSalience(pool[i], imgW, imgH);
    if (s > bestScore) {
      bestScore = s;
      best = pool[i];
    }
  }
  return best;
}

/**
 * IoU 跟踪平滑：同一主体插值；切换需明显更优，减少绿框乱跳
 */
export function stabilizeSubject(next, imgW, imgH) {
  if (!next || !next.bbox) {
    return trackedSubject;
  }
  if (!trackedSubject || !trackedSubject.bbox) {
    trackedSubject = {
      label: next.label,
      score: next.score,
      bbox: next.bbox.slice(),
    };
    return trackedSubject;
  }

  const iou = boxIoU(trackedSubject.bbox, next.bbox);
  const nextSal = subjectSalience(next, imgW, imgH);
  const prevSal = subjectSalience(trackedSubject, imgW, imgH);

  if (iou >= 0.25) {
    trackedSubject = {
      label: next.label || trackedSubject.label,
      score: next.score,
      bbox: lerpBox(trackedSubject.bbox, next.bbox, 0.4),
    };
    return trackedSubject;
  }

  if (nextSal > prevSal * 1.25 && nextSal > 0.08) {
    trackedSubject = {
      label: next.label,
      score: next.score,
      bbox: next.bbox.slice(),
    };
    return trackedSubject;
  }

  if (iou > 0.05) {
    trackedSubject = {
      label: trackedSubject.label,
      score: trackedSubject.score,
      bbox: lerpBox(trackedSubject.bbox, next.bbox, 0.12),
    };
  }
  return trackedSubject;
}

export function clearTrackedSubject() {
  trackedSubject = null;
}
