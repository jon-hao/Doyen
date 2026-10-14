/**
 * 本地免费显著性 / 人像分割（不联网推理、不付费）
 * - 人像：MediaPipe Selfie Segmenter (.tflite)
 * - 风景：U²-Net portable u2netp (.onnx via onnxruntime-web)
 */
import { ImageSegmenter } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm";

const ORT_VERSION = "1.19.2";
const ORT_ESM =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/+esm";
const ORT_WASM =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";

const U2_SIZE = 320;
const U2_MEAN = [0.485, 0.456, 0.406];
const U2_STD = [0.229, 0.224, 0.225];

/** @type {any} */
let ortMod = null;
/** @type {any} */
let u2Session = null;
/** @type {ImageSegmenter|null} */
let selfieSegmenter = null;
/** @type {HTMLCanvasElement|null} */
let u2Scratch = null;
let u2InputName = "input.1";
let u2Ts = 0;

export function hasSelfieSegmenter() {
  return !!selfieSegmenter;
}

export function hasU2Net() {
  return !!u2Session;
}

export function closeSaliencyLocal() {
  try {
    if (selfieSegmenter && typeof selfieSegmenter.close === "function") {
      selfieSegmenter.close();
    }
  } catch (_) {}
  selfieSegmenter = null;
  u2Session = null;
  ortMod = null;
  u2Scratch = null;
}

/**
 * @param {any} vision FilesetResolver result
 * @param {Uint8Array} selfieBuffer
 * @param {Uint8Array} u2Buffer
 * @param {function=} onProgress
 */
export async function initSaliencyLocal(vision, selfieBuffer, u2Buffer, onProgress) {
  // Selfie（人像主路径）
  if (selfieBuffer && selfieBuffer.byteLength > 1000 && vision) {
    try {
      selfieSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: selfieBuffer,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
    } catch (_) {
      try {
        selfieSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetBuffer: selfieBuffer,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          outputCategoryMask: true,
          outputConfidenceMasks: false,
        });
      } catch (_) {
        selfieSegmenter = null;
      }
    }
  }

  if (typeof onProgress === "function") {
    onProgress({ status: "progress", file: "selfie_segmenter.tflite", loaded: 1, total: 1 });
  }

  // u2netp（风景显著性）
  if (u2Buffer && u2Buffer.byteLength > 1000) {
    try {
      ortMod = await import(ORT_ESM);
      if (ortMod && ortMod.env && ortMod.env.wasm) {
        ortMod.env.wasm.wasmPaths = ORT_WASM;
      }
      const buf =
        u2Buffer.buffer && u2Buffer.byteOffset === 0 && u2Buffer.byteLength === u2Buffer.buffer.byteLength
          ? u2Buffer.buffer
          : u2Buffer.slice().buffer;
      u2Session = await ortMod.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
      });
      if (u2Session && u2Session.inputNames && u2Session.inputNames[0]) {
        u2InputName = u2Session.inputNames[0];
      }
    } catch (_) {
      u2Session = null;
    }
  }

  if (typeof onProgress === "function") {
    onProgress({ status: "progress", file: "u2netp.onnx", loaded: 1, total: 1 });
  }

  return {
    selfie: !!selfieSegmenter,
    u2net: !!u2Session,
  };
}

function getU2Scratch() {
  if (typeof document === "undefined") return null;
  if (!u2Scratch) u2Scratch = document.createElement("canvas");
  if (u2Scratch.width !== U2_SIZE || u2Scratch.height !== U2_SIZE) {
    u2Scratch.width = U2_SIZE;
    u2Scratch.height = U2_SIZE;
  }
  return u2Scratch;
}

/**
 * 将 category 原图归一成 1=人 / 0=背景。
 * 部分机型/模型极性相反，按前景占比 + 可选种子点自动校正，避免「反选」。
 * @param {Uint8Array} raw
 * @param {number} mw
 * @param {number} mh
 * @param {{nx?:number,ny?:number}|null} seed
 */
function normalizePersonMask(raw, mw, mh, seed) {
  const n = mw * mh;
  function build(invert) {
    const data = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const fg = raw[i] > 0;
      data[i] = invert ? (fg ? 0 : 1) : fg ? 1 : 0;
    }
    return data;
  }
  function fgRatio(data) {
    let c = 0;
    for (let i = 0; i < n; i++) if (data[i]) c += 1;
    return c / Math.max(1, n);
  }
  function scoreRatio(r) {
    // 人像主体常见 5%–45%；过大几乎一定是反选（整背景）
    if (r < 0.02 || r > 0.55) return -2;
    return 1 - Math.abs(r - 0.22) * 2.2;
  }

  const a = build(false);
  const b = build(true);
  const ra = fgRatio(a);
  const rb = fgRatio(b);
  const sa = scoreRatio(ra);
  const sb = scoreRatio(rb);

  if (seed && typeof seed.nx === "number" && typeof seed.ny === "number") {
    const ix = Math.max(0, Math.min(mw - 1, Math.round(seed.nx * (mw - 1))));
    const iy = Math.max(0, Math.min(mh - 1, Math.round(seed.ny * (mh - 1))));
    const i = iy * mw + ix;
    // 种子应落在人身上，且前景占比合理
    if (a[i] && sa > sb && sa > -1) return a;
    if (b[i] && sb > sa && sb > -1) return b;
    if (a[i] && sa > -1 && ra < 0.5) return a;
    if (b[i] && sb > -1 && rb < 0.5) return b;
  }

  // 无种子：宁可偏小人像，也不要整屏背景
  if (sa >= sb) return a;
  return b;
}

/**
 * 人像：Selfie 分割 → 人物掩码（1=人）
 * @param {HTMLCanvasElement} canvas
 * @param {number=} timestampMs
 * @param {{nx?:number,ny?:number}|null=} seed
 * @returns {null|{width:number,height:number,data:Uint8Array}}
 */
export function runSelfieMask(canvas, timestampMs, seed) {
  if (!selfieSegmenter || !canvas) return null;
  u2Ts = typeof timestampMs === "number" ? timestampMs : u2Ts + 33;
  const seedOpt = seed || null;
  let out = null;
  try {
    selfieSegmenter.segmentForVideo(canvas, u2Ts, function (result) {
      const mask = result && result.categoryMask;
      if (!mask) return;
      const mw = mask.width | 0;
      const mh = mask.height | 0;
      if (!mw || !mh) return;
      let raw;
      try {
        raw = mask.getAsUint8Array();
      } catch (_) {
        return;
      }
      out = {
        width: mw,
        height: mh,
        data: normalizePersonMask(raw, mw, mh, seedOpt),
      };
    });
  } catch (_) {
    try {
      selfieSegmenter.segment(canvas, function (result) {
        const mask = result && result.categoryMask;
        if (!mask) return;
        const mw = mask.width | 0;
        const mh = mask.height | 0;
        let raw;
        try {
          raw = mask.getAsUint8Array();
        } catch (_) {
          return;
        }
        out = {
          width: mw,
          height: mh,
          data: normalizePersonMask(raw, mw, mh, seedOpt),
        };
      });
    } catch (_) {
      return null;
    }
  }
  return out;
}

/**
 * 风景：u2netp 显著性图（软值 0–1 已二值化）
 * @returns {Promise<null|{width:number,height:number,data:Uint8Array,soft:Float32Array}>}
 */
export async function runU2NetMask(canvas) {
  if (!u2Session || !ortMod || !canvas) return null;
  const scratch = getU2Scratch();
  if (!scratch) return null;
  let ctx;
  try {
    ctx = scratch.getContext("2d", { willReadFrequently: true });
  } catch (_) {
    ctx = scratch.getContext("2d");
  }
  if (!ctx) return null;
  try {
    ctx.drawImage(canvas, 0, 0, U2_SIZE, U2_SIZE);
  } catch (_) {
    return null;
  }
  let img;
  try {
    img = ctx.getImageData(0, 0, U2_SIZE, U2_SIZE);
  } catch (_) {
    return null;
  }

  const plane = U2_SIZE * U2_SIZE;
  const floatArr = new Float32Array(3 * plane);
  const px = img.data;
  for (let i = 0, p = 0; i < px.length; i += 4, p++) {
    const r = px[i] / 255;
    const g = px[i + 1] / 255;
    const b = px[i + 2] / 255;
    floatArr[p] = (r - U2_MEAN[0]) / U2_STD[0];
    floatArr[plane + p] = (g - U2_MEAN[1]) / U2_STD[1];
    floatArr[2 * plane + p] = (b - U2_MEAN[2]) / U2_STD[2];
  }

  try {
    const input = new ortMod.Tensor("float32", floatArr, [1, 3, U2_SIZE, U2_SIZE]);
    const feeds = {};
    feeds[u2InputName] = input;
    const results = await u2Session.run(feeds);
    const keys = u2Session.outputNames || Object.keys(results);
    const firstKey = keys[0];
    const pred = results[firstKey];
    if (!pred || !pred.data) return null;
    const soft = pred.data;
    // rembg 风格：min-max 归一化再阈值
    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < soft.length; i++) {
      const v = soft[i];
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const span = Math.max(1e-6, maxV - minV);
    const data = new Uint8Array(plane);
    const softOut = new Float32Array(plane);
    for (let i = 0; i < plane; i++) {
      const n = (soft[i] - minV) / span;
      softOut[i] = n;
      data[i] = n >= 0.45 ? 1 : 0;
    }
    return { width: U2_SIZE, height: U2_SIZE, data: data, soft: softOut };
  } catch (_) {
    return null;
  }
}

/**
 * 掩码 → 主体框（图像坐标）
 * @param {{width:number,height:number,data:Uint8Array,soft?:Float32Array}} mask
 * @param {number} imgW
 * @param {number} imgH
 * @param {string=} label
 * @param {{nx?:number,ny?:number,prevBbox?:number[],requirePrev?:boolean}=} preferSeed
 *   - nx/ny：优先包含该点的连通域（点击）
 *   - prevBbox：优先与上一框重叠/邻近的连通域（移镜跟钉选，避免跳到最大主体）
 *   - requirePrev：有 prevBbox 时若找不到足够像的连通域则返回 null（宁可不换）
 * @returns {null|{label:string,score:number,bbox:number[],mask:{width:number,height:number,data:Uint8Array}}}
 */
export function primaryFromMask(mask, imgW, imgH, label, preferSeed) {
  if (!mask || !mask.data || !mask.width || !mask.height) return null;
  const mw = mask.width | 0;
  const mh = mask.height | 0;
  const src = mask.data;
  const visited = new Uint8Array(mw * mh);
  const scaleX = Math.max(1, imgW) / mw;
  const scaleY = Math.max(1, imgH) / mh;

  const wantSeed =
    preferSeed &&
    typeof preferSeed.nx === "number" &&
    typeof preferSeed.ny === "number";
  const sx = wantSeed
    ? Math.max(0, Math.min(mw - 1, Math.round(preferSeed.nx * (mw - 1))))
    : -1;
  const sy = wantSeed
    ? Math.max(0, Math.min(mh - 1, Math.round(preferSeed.ny * (mh - 1))))
    : -1;
  const prev =
    preferSeed && preferSeed.prevBbox && preferSeed.prevBbox.length >= 4
      ? preferSeed.prevBbox
      : null;
  const requirePrev = !!(preferSeed && preferSeed.requirePrev && prev);

  /** @type {Array<{comp:Uint8Array,count:number,box:number[],rank:number,iou:number}>} */
  const cands = [];

  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      const start = y * mw + x;
      if (visited[start] || !(src[start] > 0)) continue;

      const stack = [start];
      visited[start] = 1;
      let count = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      const comp = new Uint8Array(mw * mh);
      let hitsSeed = false;

      while (stack.length) {
        const i = stack.pop();
        const cx = i % mw;
        const cy = (i / mw) | 0;
        if (!(src[i] > 0)) continue;
        comp[i] = 1;
        count += 1;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        if (wantSeed && cx === sx && cy === sy) hitsSeed = true;

        const nbs = [i - 1, i + 1, i - mw, i + mw];
        for (let k = 0; k < 4; k++) {
          const j = nbs[k];
          if (j < 0 || j >= src.length || visited[j]) continue;
          const nx = j % mw;
          const ny = (j / mw) | 0;
          if (Math.abs(nx - cx) + Math.abs(ny - cy) !== 1) continue;
          visited[j] = 1;
          if (src[j] > 0) stack.push(j);
        }
      }

      if (count < 40) continue;

      const bb = [
        minX * scaleX,
        minY * scaleY,
        (maxX + 1) * scaleX,
        (maxY + 1) * scaleY,
      ];
      let rank = count;
      let iou = 0;
      if (hitsSeed) rank += count * 1.25;
      if (prev) {
        const ax1 = Math.min(bb[0], bb[2]);
        const ay1 = Math.min(bb[1], bb[3]);
        const ax2 = Math.max(bb[0], bb[2]);
        const ay2 = Math.max(bb[1], bb[3]);
        const bx1 = Math.min(prev[0], prev[2]);
        const by1 = Math.min(prev[1], prev[3]);
        const bx2 = Math.max(prev[0], prev[2]);
        const by2 = Math.max(prev[1], prev[3]);
        const ix1 = Math.max(ax1, bx1);
        const iy1 = Math.max(ay1, by1);
        const ix2 = Math.min(ax2, bx2);
        const iy2 = Math.min(ay2, by2);
        const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
        const uni =
          Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1) +
          Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1) -
          inter;
        iou = uni > 0 ? inter / uni : 0;
        const pcx = (bx1 + bx2) * 0.5;
        const pcy = (by1 + by2) * 0.5;
        const ccx = (ax1 + ax2) * 0.5;
        const ccy = (ay1 + ay2) * 0.5;
        const dist =
          Math.hypot(ccx - pcx, ccy - pcy) /
          Math.max(1, Math.min(imgW, imgH));
        // 钉选跟随时：重叠/邻近远比「最大连通域」重要
        rank = iou * 50000 + (1 - Math.min(1, dist * 2.2)) * 20000 + count;
        if (hitsSeed) rank += 8000;
      }

      cands.push({
        comp: comp,
        count: count,
        box: [minX, minY, maxX, maxY],
        rank: rank,
        iou: iou,
      });
    }
  }

  if (!cands.length) return null;
  cands.sort(function (a, b) {
    return b.rank - a.rank;
  });

  let best = cands[0];
  if (requirePrev) {
    // 必须像上一框：IoU 或中心足够近，否则视为钉选主体本帧未找到
    const pcx =
      (Math.min(prev[0], prev[2]) + Math.max(prev[0], prev[2])) / 2;
    const pcy =
      (Math.min(prev[1], prev[3]) + Math.max(prev[1], prev[3])) / 2;
    let ok = null;
    for (let i = 0; i < cands.length; i++) {
      const c = cands[i];
      const bb = [
        c.box[0] * scaleX,
        c.box[1] * scaleY,
        (c.box[2] + 1) * scaleX,
        (c.box[3] + 1) * scaleY,
      ];
      const ccx = (Math.min(bb[0], bb[2]) + Math.max(bb[0], bb[2])) / 2;
      const ccy = (Math.min(bb[1], bb[3]) + Math.max(bb[1], bb[3])) / 2;
      const dist =
        Math.hypot(ccx - pcx, ccy - pcy) /
        Math.max(1, Math.min(imgW, imgH));
      if (c.iou >= 0.1 || (c.iou >= 0.04 && dist < 0.16)) {
        ok = c;
        break;
      }
    }
    if (!ok) return null;
    best = ok;
  }

  const useComp = best.comp;
  const useCount = best.count;
  const box = best.box;
  const pad = Math.max(
    1,
    Math.round(Math.min(box[2] - box[0], box[3] - box[1]) * 0.04)
  );
  const bbox = [
    Math.max(0, (box[0] - pad) * scaleX),
    Math.max(0, (box[1] - pad) * scaleY),
    Math.min(imgW, (box[2] + 1 + pad) * scaleX),
    Math.min(imgH, (box[3] + 1 + pad) * scaleY),
  ];

  let score = Math.min(1, useCount / (mw * mh * 0.35));
  if (mask.soft) {
    let sum = 0;
    let n = 0;
    for (let i = 0; i < useComp.length; i++) {
      if (!useComp[i]) continue;
      sum += mask.soft[i];
      n += 1;
    }
    if (n) score = Math.max(score, Math.min(1, sum / n));
  }

  return {
    label: label || "subject",
    score: score,
    bbox: bbox,
    mask: { width: mw, height: mh, data: useComp },
  };
}
