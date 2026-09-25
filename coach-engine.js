/**
 * 轻量主体检测 + 轮廓分割
 * - EfficientDet-Lite0：选主体 / 跟踪
 * - MagicTouch Interactive Segmenter：精确轮廓
 */
import {
  ObjectDetector,
  InteractiveSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm";

const READY_KEY = "doyen_model_ready_v2";
const MODEL_ID = "mediapipe/efficientdet_lite0_float16";
const OD_BYTES = 7244197;
const SEG_BYTES = 6227884;
const MODEL_BYTES = OD_BYTES + SEG_BYTES;
const MP_VERSION = "0.10.18";
const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION + "/wasm";

let detector = null;
let segmenter = null;
let warmed = false;
let deviceUsed = "wasm";
let loadPromise = null;
let videoTs = 0;
/** @type {null | {label:string, score:number, bbox:number[]}} */
let trackedSubject = null;
/** 用户点击锁定的主体（优先级最高） */
let pinnedSubject = null;
/** 连续未检测到主体的帧数 */
let trackMissStreak = 0;
let pinMissStreak = 0;
/** @type {"landscape" | "portrait"} */
let subjectMode = "landscape";
/** @type {"landscape" | "portrait"} 手机横握/竖握 */
let holdOrientation = "portrait";

/**
 * 风景 / 人像 切换检测策略。
 * 人像：检测器只输出 person；风景：全类别里挑独立主体。
 */
export async function setSubjectMode(mode) {
  const next = mode === "portrait" ? "portrait" : "landscape";
  if (next === subjectMode && detector) return subjectMode;
  subjectMode = next;
  trackedSubject = null;
  pinnedSubject = null;
  trackMissStreak = 0;
  pinMissStreak = 0;
  if (!detector || typeof detector.setOptions !== "function") return subjectMode;
  try {
    if (subjectMode === "portrait") {
      await detector.setOptions({ categoryAllowlist: ["person"] });
    } else {
      await detector.setOptions({ categoryAllowlist: [] });
    }
  } catch (_) {}
  return subjectMode;
}

export function setHoldOrientation(orient) {
  holdOrientation =
    orient === "landscape" ? "landscape" : "portrait";
  return holdOrientation;
}

export function getHoldOrientation() {
  return holdOrientation;
}

export function getSubjectMode() {
  return subjectMode;
}

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

function modelCandidateUrls(fileName) {
  const local = assetUrl("models/" + fileName);
  return [
    local,
    "https://cdn.jsdelivr.net/gh/jon-hao/Doyen@main/models/" + fileName,
    "https://raw.githubusercontent.com/jon-hao/Doyen/main/models/" + fileName,
  ];
}

async function fetchWithProgress(url, onProgress, label, rangeStart, rangeEnd) {
  const start = typeof rangeStart === "number" ? rangeStart : 0;
  const end = typeof rangeEnd === "number" ? rangeEnd : 1;
  const span = Math.max(0.01, end - start);
  const fileTotal =
    label === "magic_touch.tflite" ? SEG_BYTES : OD_BYTES;

  function report(loadedPart, totalPart) {
    const ratio = totalPart > 0 ? loadedPart / totalPart : 1;
    notify(onProgress, {
      status: "progress",
      file: label,
      loaded: Math.round((totalPart > 0 ? totalPart : fileTotal) * Math.min(1, ratio)),
      total: totalPart > 0 ? totalPart : fileTotal,
    });
  }

  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) throw new Error("下载失败 HTTP " + res.status);

  const totalHeader = parseInt(res.headers.get("Content-Length") || "0", 10);
  const total = totalHeader > 0 ? totalHeader : fileTotal;

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

async function fetchNamedModelBuffer(fileName, onProgress, rangeStart, rangeEnd) {
  const urls = modelCandidateUrls(fileName);
  let lastErr = null;
  for (let i = 0; i < urls.length; i++) {
    try {
      return await fetchWithProgress(
        urls[i],
        onProgress,
        fileName,
        rangeStart,
        rangeEnd
      );
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("模型文件不可用");
}

async function fetchModelBuffer(onProgress) {
  return fetchNamedModelBuffer(
    "efficientdet_lite0.tflite",
    onProgress,
    0.12,
    0.55
  );
}

async function fetchSegmenterBuffer(onProgress) {
  return fetchNamedModelBuffer("magic_touch.tflite", onProgress, 0.55, 0.92);
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
  try {
    if (segmenter && typeof segmenter.close === "function") segmenter.close();
  } catch (_) {}
  detector = null;
  segmenter = null;
  warmed = false;
  loadPromise = null;
  deviceUsed = "wasm";
  videoTs = 0;
  trackedSubject = null;
  pinnedSubject = null;
  trackMissStreak = 0;
  pinMissStreak = 0;
  subjectMode = "landscape";
}
export function getCoachDevice() {
  return deviceUsed;
}

async function createDetector(onProgress) {
  notify(onProgress, {
    status: "initiate",
    file: "efficientdet_lite0.tflite",
    total: OD_BYTES,
  });
  notify(onProgress, {
    status: "download",
    file: "efficientdet_lite0.tflite",
    total: OD_BYTES,
  });
  emitResourceProgress(onProgress, 0.02, "efficientdet_lite0.tflite", OD_BYTES);

  const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
  emitResourceProgress(onProgress, 0.1, "efficientdet_lite0.tflite", OD_BYTES);

  const modelBuffer = await fetchModelBuffer(onProgress);
  emitResourceProgress(onProgress, 0.52, "efficientdet_lite0.tflite", OD_BYTES);

  let created;
  try {
    created = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "CPU",
      },
      scoreThreshold: 0.26,
      maxResults: 16,
      runningMode: "VIDEO",
    });
    deviceUsed = "wasm";
  } catch (_) {
    created = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: modelBuffer,
        delegate: "GPU",
      },
      scoreThreshold: 0.26,
      maxResults: 16,
      runningMode: "VIDEO",
    });
    deviceUsed = "webgl";
  }

  notify(onProgress, {
    status: "done",
    file: "efficientdet_lite0.tflite",
    loaded: OD_BYTES,
    total: OD_BYTES,
  });

  // 轮廓分割（失败则仅用框，不阻断辅助拍摄）
  try {
    notify(onProgress, {
      status: "initiate",
      file: "magic_touch.tflite",
      total: SEG_BYTES,
    });
    notify(onProgress, {
      status: "download",
      file: "magic_touch.tflite",
      total: SEG_BYTES,
    });
    const segBuf = await fetchSegmenterBuffer(onProgress);
    segmenter = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: segBuf,
        delegate: "CPU",
      },
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
    notify(onProgress, {
      status: "done",
      file: "magic_touch.tflite",
      loaded: SEG_BYTES,
      total: SEG_BYTES,
    });
  } catch (_) {
    segmenter = null;
  }

  videoTs = 0;
  trackedSubject = null;
  pinnedSubject = null;
  emitResourceProgress(onProgress, 0.99, "magic_touch.tflite", SEG_BYTES);
  return created;
}

function emitResourceProgress(onProgress, fraction, file, totalBytes) {
  const pct = Math.max(0, Math.min(0.99, Number(fraction) || 0));
  const total = totalBytes || MODEL_BYTES;
  notify(onProgress, {
    status: "progress",
    file: file || "efficientdet_lite0.tflite",
    loaded: Math.round(total * pct),
    total: total,
  });
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
 * 完整度：框是否整段落在画面内（贴边 = 可能被裁切，完整度下降）。
 * 1 = 四边都留白；贴边越多越低。
 */
function boxCompleteness(b, imgW, imgH) {
  if (!b || b.length < 4) return 0;
  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const x1 = Math.min(b[0], b[2]);
  const y1 = Math.min(b[1], b[3]);
  const x2 = Math.max(b[0], b[2]);
  const y2 = Math.max(b[1], b[3]);
  const margin = Math.max(2, Math.min(w, h) * 0.02);
  let score = 1;
  if (x1 <= margin) score -= 0.28;
  if (y1 <= margin) score -= 0.28;
  if (x2 >= w - margin) score -= 0.28;
  if (y2 >= h - margin) score -= 0.28;
  if (x1 < 0 || y1 < 0 || x2 > w || y2 > h) score -= 0.15;
  return Math.max(0, Math.min(1, score));
}

/** 框中心到画面中心的接近度 0–1 */
function boxCenterScore(b, imgW, imgH) {
  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const cx = (Math.min(b[0], b[2]) + Math.max(b[0], b[2])) / 2;
  const cy = (Math.min(b[1], b[3]) + Math.max(b[1], b[3])) / 2;
  const dx = (cx - w * 0.5) / w;
  const dy = (cy - h * 0.5) / h;
  return 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy) * 1.35);
}

/**
 * 三分法兴趣点接近度（摄影构图常用）：离四个交叉点越近越好。
 * 参考移动端实时构图评分研究。
 */
function boxRuleOfThirdsScore(b, imgW, imgH) {
  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const cx = (Math.min(b[0], b[2]) + Math.max(b[0], b[2])) / 2;
  const cy = (Math.min(b[1], b[3]) + Math.max(b[1], b[3])) / 2;
  const xs = [w / 3, (2 * w) / 3];
  const ys = [h / 3, (2 * h) / 3];
  let best = 0;
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const dx = (cx - xs[i]) / w;
      const dy = (cy - ys[j]) / h;
      const near = 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy) * 2.2);
      if (near > best) best = near;
    }
  }
  return best;
}

/**
 * 轻量「显著性」代理：框内颜色相对周围边环的对比度。
 * 类 SOD 思路，不引入额外模型（省 iPhone 内存）。
 */
function regionContrastScore(canvas, bbox) {
  if (!canvas || !bbox || typeof canvas.getContext !== "function") return 0.5;
  let ctx;
  try {
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  } catch (_) {
    ctx = canvas.getContext("2d");
  }
  if (!ctx) return 0.5;

  const W = canvas.width | 0;
  const H = canvas.height | 0;
  if (W < 8 || H < 8) return 0.5;

  const x1 = Math.max(0, Math.floor(Math.min(bbox[0], bbox[2])));
  const y1 = Math.max(0, Math.floor(Math.min(bbox[1], bbox[3])));
  const x2 = Math.min(W, Math.ceil(Math.max(bbox[0], bbox[2])));
  const y2 = Math.min(H, Math.ceil(Math.max(bbox[1], bbox[3])));
  const bw = x2 - x1;
  const bh = y2 - y1;
  if (bw < 6 || bh < 6) return 0.35;

  function meanRGB(sx, sy, sw, sh) {
    const w = Math.max(1, Math.min(sw, W - sx));
    const h = Math.max(1, Math.min(sh, H - sy));
    if (sx < 0 || sy < 0 || sx >= W || sy >= H) return null;
    let data;
    try {
      data = ctx.getImageData(sx, sy, w, h).data;
    } catch (_) {
      return null;
    }
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    const step = Math.max(4, Math.floor((data.length / 4) / 48) * 4);
    for (let i = 0; i < data.length; i += step) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      n += 1;
    }
    if (!n) return null;
    return [r / n, g / n, b / n];
  }

  const pad = Math.max(2, Math.round(Math.min(bw, bh) * 0.12));
  const inner = meanRGB(
    x1 + pad,
    y1 + pad,
    Math.max(1, bw - pad * 2),
    Math.max(1, bh - pad * 2)
  );
  if (!inner) return 0.5;

  const rings = [
    meanRGB(Math.max(0, x1 - pad), y1, pad, bh),
    meanRGB(x2, y1, pad, bh),
    meanRGB(x1, Math.max(0, y1 - pad), bw, pad),
    meanRGB(x1, y2, bw, pad),
  ];
  let dr = 0;
  let count = 0;
  for (let i = 0; i < rings.length; i++) {
    const o = rings[i];
    if (!o) continue;
    const d =
      Math.abs(inner[0] - o[0]) +
      Math.abs(inner[1] - o[1]) +
      Math.abs(inner[2] - o[2]);
    dr += d / (3 * 255);
    count += 1;
  }
  if (!count) return 0.5;
  return Math.max(0, Math.min(1, (dr / count) * 2.2));
}

/** 风景：常见「可拍主体」类别加权；家具/场景平面降权（避免沙发/墙面抢框） */
const LANDSCAPE_SUBJECT_BOOST = {
  bird: 1.35,
  cat: 1.35,
  dog: 1.35,
  horse: 1.2,
  sheep: 1.15,
  cow: 1.15,
  elephant: 1.2,
  bear: 1.25,
  zebra: 1.2,
  giraffe: 1.2,
  "potted plant": 1.3,
  vase: 1.25,
  bottle: 1.15,
  cup: 1.1,
  bowl: 1.1,
  "wine glass": 1.1,
  cake: 1.2,
  "teddy bear": 1.25,
  "sports ball": 1.2,
  frisbee: 1.15,
  kite: 1.2,
  umbrella: 1.1,
  backpack: 1.05,
  handbag: 1.05,
  suitcase: 1.05,
  bicycle: 1.15,
  motorcycle: 1.1,
  boat: 1.15,
  airplane: 1.1,
  car: 1.05,
  clock: 1.1,
  book: 1.05,
  "cell phone": 1.05,
  laptop: 1.05,
  banana: 1.1,
  apple: 1.1,
  orange: 1.1,
  sandwich: 1.05,
  pizza: 1.05,
  donut: 1.05,
  "hot dog": 1.05,
  broccoli: 1.05,
  carrot: 1.05,
};

const LANDSCAPE_BG_PENALTY = {
  couch: 0.35,
  sofa: 0.35,
  bed: 0.4,
  "dining table": 0.3,
  table: 0.45,
  chair: 0.55,
  bench: 0.55,
  refrigerator: 0.35,
  oven: 0.4,
  microwave: 0.45,
  sink: 0.4,
  toilet: 0.35,
  tv: 0.45,
  "tvmonitor": 0.45,
  "traffic light": 0.5,
  "stop sign": 0.55,
  "fire hydrant": 0.7,
  "parking meter": 0.55,
  truck: 0.7,
  bus: 0.65,
  train: 0.65,
};

function landscapeClassWeight(label) {
  const lab = String(label || "").toLowerCase().trim();
  if (LANDSCAPE_SUBJECT_BOOST[lab] != null) return LANDSCAPE_SUBJECT_BOOST[lab];
  if (LANDSCAPE_BG_PENALTY[lab] != null) return LANDSCAPE_BG_PENALTY[lab];
  if (isPersonLabel(lab)) return 0.72; // 风景不优先路人
  return 1;
}

/**
 * 人像策略：只框人。
 * 权重：① 身体完整度 ② 接近画面中心 ③ 合适占比 + 人形长宽比。
 * 横握/竖握调整：竖握偏全身竖构图；横握略放宽宽高比、更看重水平居中。
 */
function portraitSalience(obj, imgW, imgH) {
  const b = obj && obj.bbox;
  if (!b || b.length < 4) return -1;
  if (!isPersonLabel(obj.label)) return -1;

  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const frameArea = w * h;
  const bw = Math.abs(b[2] - b[0]);
  const bh = Math.abs(b[3] - b[1]);
  const areaRatio = (bw * bh) / frameArea;
  if (areaRatio < 0.03 || areaRatio > 0.78) return -1;

  const completeness = boxCompleteness(b, w, h);
  const center = boxCenterScore(b, w, h);
  const conf = Math.max(0, Math.min(1, Number(obj.score) || 0));
  const aspect = bw / Math.max(bh, 1);
  const holdLand = holdOrientation === "landscape";

  let shape = 0.7;
  if (holdLand) {
    if (aspect >= 0.28 && aspect <= 1.05) shape = 1.25;
    else if (aspect > 1.05 && aspect <= 1.35) shape = 1.0;
    else if (aspect > 1.5) shape = 0.4;
  } else {
    if (aspect >= 0.22 && aspect <= 0.72) shape = 1.25;
    else if (aspect > 0.72 && aspect <= 1.05) shape = 1.05;
    else if (aspect > 1.2) shape = 0.4;
  }

  const sizeSweet =
    areaRatio < 0.08
      ? areaRatio / 0.08
      : areaRatio <= (holdLand ? 0.5 : 0.45)
        ? 1
        : Math.max(0.2, 1 - (areaRatio - 0.45) / 0.35);

  const centerW = holdLand ? 3.8 : 3.2;

  return (
    completeness * 12 +
    center * centerW +
    sizeSweet * 1.4 +
    shape * 0.9 +
    conf * 0.8
  );
}

/**
 * 风景策略：类显著目标选独立主体。
 * 竖握偏中心偏上构图；横握更看三分法横向兴趣点与中等宽度主体。
 */
function landscapeSalience(obj, imgW, imgH, canvas) {
  const b = obj && obj.bbox;
  if (!b || b.length < 4) return -1;

  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const frameArea = w * h;
  const bw = Math.abs(b[2] - b[0]);
  const bh = Math.abs(b[3] - b[1]);
  const areaRatio = (bw * bh) / frameArea;
  if (areaRatio < 0.018 || areaRatio > 0.62) return -1;

  const completeness = boxCompleteness(b, w, h);
  const center = boxCenterScore(b, w, h);
  const thirds = boxRuleOfThirdsScore(b, w, h);
  const conf = Math.max(0, Math.min(1, Number(obj.score) || 0));
  const contrast = regionContrastScore(canvas, b);
  const classW = landscapeClassWeight(obj.label);
  const aspect = bw / Math.max(bh, 1);
  const holdLand = holdOrientation === "landscape";
  const shape = aspect > 2.4 || aspect < 0.12 ? 0.55 : 1;

  const sizeSweet =
    areaRatio < 0.06
      ? areaRatio / 0.06
      : areaRatio <= (holdLand ? 0.4 : 0.35)
        ? 1
        : Math.max(0.15, 1 - (areaRatio - 0.35) / 0.3);

  // 竖握：略偏画面中心偏上；横握：中心与三分法并重
  let placement;
  if (holdLand) {
    placement = Math.max(center * 0.9, thirds);
  } else {
    const cy = (Math.min(b[1], b[3]) + Math.max(b[1], b[3])) / 2;
    const upper = 1 - Math.min(1, Math.abs(cy / h - 0.42) * 2.2);
    placement = Math.max(center, thirds * 0.85) * (0.75 + 0.25 * upper);
  }

  const holdBoost = holdLand
    ? aspect >= 0.7 && aspect <= 1.8
      ? 1.08
      : 1
    : aspect <= 1.1
      ? 1.06
      : 1;

  return (
    (completeness * 10 +
      placement * 3 +
      contrast * 2.4 +
      sizeSweet * 1.5 +
      conf * 0.7) *
    classW *
    shape *
    holdBoost
  );
}

function subjectSalience(obj, imgW, imgH, mode, canvas) {
  if (mode === "portrait") return portraitSalience(obj, imgW, imgH);
  return landscapeSalience(obj, imgW, imgH, canvas);
}

function intersectionArea(a, b) {
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
  return Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
}

/** outer 是否大面积包住 inner（典型「组合大框」） */
function mostlyContains(outer, inner) {
  const innerArea = boxArea(inner);
  const outerArea = boxArea(outer);
  if (innerArea <= 0 || outerArea <= 0) return false;
  if (outerArea < innerArea * 1.12) return false;
  return intersectionArea(outer, inner) / innerArea >= 0.7;
}

/**
 * 去掉「包住多个体」的并集大框，只留更像独立个体的框
 */
function keepAtomicIndividuals(list) {
  const items = list || [];
  if (items.length <= 1) return items.slice();

  const significant = items.filter(function (o) {
    return (Number(o.score) || 0) >= 0.2 && boxArea(o.bbox) > 0;
  });
  const pool = significant.length ? significant : items;

  const atomic = [];
  for (let i = 0; i < pool.length; i++) {
    const cand = pool[i];
    let childCount = 0;
    let dominatedBySmaller = false;

    for (let j = 0; j < pool.length; j++) {
      if (i === j) continue;
      const other = pool[j];
      if (mostlyContains(cand.bbox, other.bbox)) {
        childCount += 1;
        // 内部已有更紧的个体 → 大框是组合框，丢弃
        if (boxArea(cand.bbox) > boxArea(other.bbox) * 1.2) {
          dominatedBySmaller = true;
        }
      }
    }

    // 包住 ≥2 个其它检测 = 多物体并集
    if (childCount >= 2) continue;
    if (dominatedBySmaller && childCount >= 1) continue;
    atomic.push(cand);
  }

  return atomic.length ? atomic : pool;
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
 * @param {Blob|HTMLCanvasElement} source
 * @param {function=} onProgress
 * @param {{mode?: "landscape"|"portrait", holdOrientation?: "landscape"|"portrait"}=} options
 */
export async function detectSubjects(source, onProgress, options) {
  if (!detector || !warmed) {
    await loadCoach(onProgress);
  }
  const mode =
    options && options.mode === "portrait" ? "portrait" : "landscape";
  if (options && options.holdOrientation) {
    setHoldOrientation(options.holdOrientation);
  }
  if (mode !== subjectMode) {
    await setSubjectMode(mode);
  }

  const canvas = canvasFromSource(source);
  videoTs += 33;
  let result;
  try {
    result = detector.detectForVideo(canvas, videoTs);
  } catch (_) {
    result = detector.detect(canvas);
  }

  let objects = detectionsToObjects(result);
  if (mode === "portrait") {
    objects = objects.filter(function (o) {
      return isPersonLabel(o.label);
    });
  }

  const primaryRaw = pickPrimarySubject(
    objects,
    canvas.width,
    canvas.height,
    mode,
    canvas
  );
  const primary = stabilizeSubject(primaryRaw, canvas.width, canvas.height, mode);

  let mask = null;
  if (primary && primary.bbox) {
    const cx =
      (Math.min(primary.bbox[0], primary.bbox[2]) +
        Math.max(primary.bbox[0], primary.bbox[2])) /
      2;
    const cy =
      (Math.min(primary.bbox[1], primary.bbox[3]) +
        Math.max(primary.bbox[1], primary.bbox[3])) /
      2;
    const nx = pinnedSubject && typeof pinnedSubject.nx === "number"
      ? pinnedSubject.nx
      : cx / Math.max(1, canvas.width);
    const ny = pinnedSubject && typeof pinnedSubject.ny === "number"
      ? pinnedSubject.ny
      : cy / Math.max(1, canvas.height);
    mask = segmentSubjectMask(canvas, nx, ny);
  }

  return {
    objects: objects,
    primary: primary,
    mask: mask,
    mode: mode,
    imageSize: [canvas.width, canvas.height],
    device: deviceUsed,
  };
}

/**
 * 在点击位置附近锁定可识别主体（优先级最高）
 * @returns {object|null} 锁定的主体
 */
export function pinSubjectAt(objects, x, y, imgW, imgH, mode) {
  const list = objects || [];
  const m = mode === "portrait" ? "portrait" : "landscape";
  let pool = list;
  if (m === "portrait") {
    pool = list.filter(function (o) {
      return isPersonLabel(o.label);
    });
  }
  pool = keepAtomicIndividuals(pool);
  if (!pool.length) {
    pinnedSubject = null;
    return null;
  }

  const w = Math.max(1, imgW || 1);
  const h = Math.max(1, imgH || 1);
  const radius = Math.min(w, h) * 0.1;
  let best = null;
  let bestRank = Infinity;

  for (let i = 0; i < pool.length; i++) {
    const o = pool[i];
    const b = o.bbox;
    if (!b) continue;
    const x1 = Math.min(b[0], b[2]);
    const y1 = Math.min(b[1], b[3]);
    const x2 = Math.max(b[0], b[2]);
    const y2 = Math.max(b[1], b[3]);
    const inside = x >= x1 && x <= x2 && y >= y1 && y <= y2;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    let dist;
    if (inside) {
      dist = Math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) * 0.25;
    } else {
      const dx = x < x1 ? x1 - x : x > x2 ? x - x2 : 0;
      const dy = y < y1 ? y1 - y : y > y2 ? y - y2 : 0;
      dist = Math.sqrt(dx * dx + dy * dy);
    }
    if (dist > radius && !inside) continue;
    const rank = inside ? dist : dist + 50;
    if (rank < bestRank) {
      bestRank = rank;
      best = o;
    }
  }

  if (!best) {
    pinnedSubject = null;
    return null;
  }

  pinnedSubject = {
    label: best.label,
    score: best.score,
    bbox: best.bbox.slice(),
    nx: x / w,
    ny: y / h,
  };
  pinMissStreak = 0;
  trackedSubject = {
    label: best.label,
    score: best.score,
    bbox: best.bbox.slice(),
  };
  trackMissStreak = 0;
  return pinnedSubject;
}

export function clearPinnedSubject() {
  pinnedSubject = null;
  pinMissStreak = 0;
}

/**
 * MagicTouch：在归一化点击/主体中心处分割轮廓掩码
 * 输出统一为 1=主体、0=背景（按点击点极性自动校正，避免内外反了）
 * @returns {null|{width:number,height:number,data:Uint8Array}}
 */
export function segmentSubjectMask(canvas, nx, ny) {
  if (!segmenter || !canvas) return null;
  const x = Math.max(0, Math.min(1, Number(nx) || 0.5));
  const y = Math.max(0, Math.min(1, Number(ny) || 0.5));
  let out = null;
  try {
    segmenter.segment(
      canvas,
      { keypoint: { x: x, y: y } },
      function (result) {
        const mask = result && result.categoryMask;
        if (!mask) return;
        const mw = mask.width | 0;
        const mh = mask.height | 0;
        if (!mw || !mh) return;
        let raw;
        try {
          raw = mask.getAsUint8Array();
        } catch (_) {
          try {
            const f32 = mask.getAsFloat32Array();
            raw = new Uint8Array(f32.length);
            for (let i = 0; i < f32.length; i++) {
              raw[i] = f32[i] > 0.5 ? 1 : 0;
            }
          } catch (_) {
            return;
          }
        }
        const ix = Math.max(0, Math.min(mw - 1, Math.round(x * (mw - 1))));
        const iy = Math.max(0, Math.min(mh - 1, Math.round(y * (mh - 1))));
        const atClick = raw[iy * mw + ix] > 0;
        // 点击点应落在主体上：若模型把主体标成 0，则整幅取反
        const data = new Uint8Array(mw * mh);
        for (let i = 0; i < data.length; i++) {
          const fg = raw[i] > 0;
          data[i] = atClick ? (fg ? 1 : 0) : fg ? 0 : 1;
        }
        out = {
          width: mw,
          height: mh,
          data: data,
        };
      }
    );
  } catch (_) {
    return null;
  }
  return out;
}

export function hasSegmenter() {
  return !!segmenter;
}

/** 兼容旧接口 */
export async function analyzeFrame(source, onProgress, options) {
  const vision = await detectSubjects(source, onProgress, options);
  return {
    caption: "",
    objects: vision.objects,
    primary: vision.primary,
    mask: vision.mask,
    device: deviceUsed,
  };
}

/**
 * 风景：类 SOD + 构图先验，挑一个独立主体（不强制人）
 * 人像：只框人；没人则返回 null
 * 用户点击锁定的主体优先级最高
 */
export function pickPrimarySubject(objects, imgW, imgH, mode, canvas) {
  const list = objects || [];
  if (!list.length) return null;
  const m = mode === "portrait" ? "portrait" : "landscape";

  let basePool;
  if (m === "portrait") {
    basePool = list.filter(function (o) {
      return isPersonLabel(o.label);
    });
    if (!basePool.length) return null;
  } else {
    basePool = list;
  }

  const pool = keepAtomicIndividuals(basePool);
  let best = null;
  let bestScore = -1;
  for (let i = 0; i < pool.length; i++) {
    let s = subjectSalience(pool[i], imgW, imgH, m, canvas);

    if (pinnedSubject && pinnedSubject.bbox && pool[i].bbox) {
      const iou = boxIoU(pinnedSubject.bbox, pool[i].bbox);
      const px = (pinnedSubject.nx || 0.5) * imgW;
      const py = (pinnedSubject.ny || 0.5) * imgH;
      const b = pool[i].bbox;
      const inside =
        px >= Math.min(b[0], b[2]) &&
        px <= Math.max(b[0], b[2]) &&
        py >= Math.min(b[1], b[3]) &&
        py <= Math.max(b[1], b[3]);
      if (inside || iou >= 0.2) {
        s += 100;
      } else if (iou >= 0.08) {
        s += 40;
      }
    }

    if (trackedSubject && trackedSubject.bbox && pool[i].bbox) {
      const iou = boxIoU(trackedSubject.bbox, pool[i].bbox);
      if (iou >= 0.25) s += m === "portrait" ? 2.2 : 1.4;
      else if (iou >= 0.12) s += m === "portrait" ? 1.0 : 0.6;
    }
    if (s > bestScore) {
      bestScore = s;
      best = pool[i];
    }
  }

  if (pinnedSubject) {
    if (
      best &&
      best.bbox &&
      (boxIoU(pinnedSubject.bbox, best.bbox) >= 0.12 ||
        (function () {
          const px = (pinnedSubject.nx || 0.5) * imgW;
          const py = (pinnedSubject.ny || 0.5) * imgH;
          const b = best.bbox;
          return (
            px >= Math.min(b[0], b[2]) &&
            px <= Math.max(b[0], b[2]) &&
            py >= Math.min(b[1], b[3]) &&
            py <= Math.max(b[1], b[3])
          );
        })())
    ) {
      pinMissStreak = 0;
      pinnedSubject.bbox = best.bbox.slice();
      pinnedSubject.label = best.label;
      pinnedSubject.score = best.score;
    } else {
      pinMissStreak += 1;
      if (pinMissStreak >= 8) {
        pinnedSubject = null;
        pinMissStreak = 0;
      }
    }
  }

  return best;
}

/**
 * 防抖：主体未大变时保持框不变，避免小幅晃动抖动。
 */
export function stabilizeSubject(next, imgW, imgH, mode) {
  const m = mode === "portrait" ? "portrait" : "landscape";
  if (m === "portrait" && next && !isPersonLabel(next.label)) {
    next = null;
  }
  if (m === "portrait" && trackedSubject && !isPersonLabel(trackedSubject.label)) {
    trackedSubject = null;
  }

  if (!next || !next.bbox) {
    trackMissStreak += 1;
    if (trackMissStreak >= 4) {
      trackedSubject = null;
      return null;
    }
    return trackedSubject;
  }
  trackMissStreak = 0;

  if (!trackedSubject || !trackedSubject.bbox) {
    trackedSubject = {
      label: next.label,
      score: next.score,
      bbox: next.bbox.slice(),
    };
    return trackedSubject;
  }

  const prevArea = boxArea(trackedSubject.bbox);
  const nextArea = boxArea(next.bbox);
  if (
    nextArea > prevArea * 1.85 &&
    mostlyContains(next.bbox, trackedSubject.bbox)
  ) {
    return trackedSubject;
  }

  const iou = boxIoU(trackedSubject.bbox, next.bbox);
  const pcx =
    (Math.min(trackedSubject.bbox[0], trackedSubject.bbox[2]) +
      Math.max(trackedSubject.bbox[0], trackedSubject.bbox[2])) /
    2;
  const pcy =
    (Math.min(trackedSubject.bbox[1], trackedSubject.bbox[3]) +
      Math.max(trackedSubject.bbox[1], trackedSubject.bbox[3])) /
    2;
  const ncx =
    (Math.min(next.bbox[0], next.bbox[2]) + Math.max(next.bbox[0], next.bbox[2])) /
    2;
  const ncy =
    (Math.min(next.bbox[1], next.bbox[3]) + Math.max(next.bbox[1], next.bbox[3])) /
    2;
  const minSide = Math.max(1, Math.min(imgW || 1, imgH || 1));
  const shift =
    Math.sqrt((ncx - pcx) * (ncx - pcx) + (ncy - pcy) * (ncy - pcy)) / minSide;
  const areaDelta =
    Math.abs(nextArea - prevArea) / Math.max(prevArea, 1);

  // 死区加宽：轻微晃动完全不改框
  if (iou >= 0.5 && shift < 0.045 && areaDelta < 0.16) {
    trackedSubject = {
      label: next.label || trackedSubject.label,
      score: next.score,
      bbox: trackedSubject.bbox.slice(),
    };
    return trackedSubject;
  }

  // 同主体缓动：极低跟随
  if (iou >= 0.28) {
    const t = iou >= 0.48 && shift < 0.08 ? 0.05 : 0.12;
    trackedSubject = {
      label: next.label || trackedSubject.label,
      score: next.score,
      bbox: lerpBox(trackedSubject.bbox, next.bbox, t),
    };
    return trackedSubject;
  }

  if (iou >= 0.12) {
    trackedSubject = {
      label: next.label || trackedSubject.label,
      score: next.score,
      bbox: lerpBox(trackedSubject.bbox, next.bbox, 0.32),
    };
    return trackedSubject;
  }

  trackedSubject = {
    label: next.label,
    score: next.score,
    bbox: next.bbox.slice(),
  };
  return trackedSubject;
}

export function clearTrackedSubject() {
  trackedSubject = null;
  pinnedSubject = null;
  trackMissStreak = 0;
  pinMissStreak = 0;
}
