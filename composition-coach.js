/**
 * 风景构图实时评分与引导
 * 依据：
 * - Liu et al. Optimizing Photo Composition（三分法 / 视觉平衡 / 尺寸甜区）
 * - Sensors 2020 Photo Composition with Real-Time Rating（主体重心距体育三分点）
 * - EVA / AADB 美学属性相对重要性（composition+depth > light+color > quality）
 * - 实时相机引导：移镜 / 变焦建议
 *
 * 权重定稿（合计 1.00）：
 * - EVA：composition&depth 影响最大，light&color 略次
 * - AADB+XAI：interesting content / object emphasis 靠前
 * - 实拍：提高主体存在 + 分离 + 反杂乱，避免白墙/杂桌/马桶拿绿
 */

/**
 * @type {Readonly<{
 *   subjectPresence:number,
 *   separation:number,
 *   antiClutter:number,
 *   placement:number,
 *   balance:number,
 *   size:number,
 *   completeness:number,
 *   tone:number,
 *   color:number,
 *   level:number
 * }>}
 */
export const AESTHETIC_WEIGHTS = Object.freeze({
  subjectPresence: 0.16,
  separation: 0.12,
  antiClutter: 0.14,
  placement: 0.14,
  balance: 0.08,
  size: 0.08,
  completeness: 0.06,
  tone: 0.1,
  color: 0.07,
  level: 0.05,
});

export const AESTHETIC_GATES = Object.freeze({
  noSubjectScoreCap: 0.4,
  nonScenicMultiplier: 0.45,
  multiSubjectMultiplier: 0.7,
});

/** COCO 类：风景意图下不该拿高分的室内/设施主体 */
const NON_SCENIC_LABELS = {
  toilet: 1,
  sink: 1,
  couch: 1,
  sofa: 1,
  bed: 1,
  "dining table": 1,
  table: 1,
  chair: 1,
  refrigerator: 1,
  oven: 1,
  microwave: 1,
  toaster: 1,
  "hair drier": 1,
  toothbrush: 1,
  tv: 1,
  tvmonitor: 1,
  laptop: 1,
  keyboard: 1,
  mouse: 1,
  remote: 1,
};

/**
 * @typedef {{bbox:number[], score?:number, label?:string}} SubjectBox
 * @typedef {{
 *   score:number,
 *   grade:"B"|"A"|"S",
 *   color:string,
 *   target:{x:number,y:number},
 *   subjectCenter:{x:number,y:number}|null,
 *   arrows:{up:number,down:number,left:number,right:number},
 *   suggestedZoom:number,
 *   suggestedMm:number,
 *   suggestedName:string,
 *   needFocalChange:boolean,
 *   areaRatio:number,
 *   potentialGrade:"B"|"A"|"S",
 *   factors:Object.<string,number>
 * }} CompositionGuide
 */

const POWER_POINTS = [
  [1 / 3, 1 / 3],
  [2 / 3, 1 / 3],
  [1 / 3, 2 / 3],
  [2 / 3, 2 / 3],
];

const GRADE_A = 0.48;
const GRADE_S = 0.72;

/** @type {HTMLCanvasElement|null} */
let scratchCanvas = null;

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function gradeFromScore(score) {
  if (score >= GRADE_S) return "S";
  if (score >= GRADE_A) return "A";
  return "B";
}

function boxCenter(b) {
  return {
    x: (Math.min(b[0], b[2]) + Math.max(b[0], b[2])) / 2,
    y: (Math.min(b[1], b[3]) + Math.max(b[1], b[3])) / 2,
  };
}

function boxAreaRatio(b, imgW, imgH) {
  const bw = Math.abs(b[2] - b[0]);
  const bh = Math.abs(b[3] - b[1]);
  return (bw * bh) / Math.max(1, imgW * imgH);
}

function boxCompleteness(b, imgW, imgH) {
  const margin = Math.max(2, Math.min(imgW, imgH) * 0.02);
  const x1 = Math.min(b[0], b[2]);
  const y1 = Math.min(b[1], b[3]);
  const x2 = Math.max(b[0], b[2]);
  const y2 = Math.max(b[1], b[3]);
  let s = 1;
  if (x1 <= margin) s -= 0.28;
  if (y1 <= margin) s -= 0.28;
  if (x2 >= imgW - margin) s -= 0.28;
  if (y2 >= imgH - margin) s -= 0.28;
  return clamp01(s);
}

function scoreRuleOfThirds(nx, ny) {
  let best = 0;
  for (let i = 0; i < POWER_POINTS.length; i++) {
    const dx = nx - POWER_POINTS[i][0];
    const dy = ny - POWER_POINTS[i][1];
    const d = Math.sqrt(dx * dx + dy * dy);
    const s = Math.exp(-(d * d) / (2 * 0.12 * 0.12));
    if (s > best) best = s;
  }
  return best;
}

function scoreCenter(nx, ny) {
  const dx = nx - 0.5;
  const dy = ny - 0.5;
  const d = Math.sqrt(dx * dx + dy * dy);
  return Math.exp(-(d * d) / (2 * 0.16 * 0.16));
}

function scoreSize(areaRatio) {
  if (areaRatio < 0.03) return (areaRatio / 0.03) * 0.35;
  if (areaRatio < 0.08) return 0.35 + ((areaRatio - 0.03) / 0.05) * 0.45;
  if (areaRatio <= 0.32) return 1;
  if (areaRatio <= 0.5) return Math.max(0.2, 1 - (areaRatio - 0.32) / 0.18);
  return Math.max(0, 0.2 - (areaRatio - 0.5) * 0.5);
}

function scoreVisualBalance(nx, ny, areaRatio) {
  const mass = 0.35 + 0.65 * clamp01(areaRatio / 0.25);
  const dx = (nx - 0.5) * mass;
  const dy = (ny - 0.5) * mass;
  const d = Math.sqrt(dx * dx + dy * dy);
  return Math.exp(-(d * d) / (2 * 0.22 * 0.22));
}

function nearestPowerPoint(nx, ny) {
  let best = POWER_POINTS[0];
  let bestD = Infinity;
  for (let i = 0; i < POWER_POINTS.length; i++) {
    const dx = nx - POWER_POINTS[i][0];
    const dy = ny - POWER_POINTS[i][1];
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = POWER_POINTS[i];
    }
  }
  return { x: best[0], y: best[1] };
}

function isNonScenicLabel(label) {
  const lab = String(label || "")
    .toLowerCase()
    .trim();
  return !!NON_SCENIC_LABELS[lab];
}

function getScratch(tw, th) {
  if (typeof document === "undefined") return null;
  if (!scratchCanvas) {
    scratchCanvas = document.createElement("canvas");
  }
  if (scratchCanvas.width !== tw || scratchCanvas.height !== th) {
    scratchCanvas.width = tw;
    scratchCanvas.height = th;
  }
  return scratchCanvas;
}

/**
 * 降采样整帧，估明暗层次与色彩丰富度（轻量，适合 ~10fps 辅助环）。
 * @returns {{tone:number, color:number, meanLuma:number, sat:number, contrast:number}|null}
 */
export function sampleFrameAesthetics(canvas) {
  if (!canvas || typeof canvas.getContext !== "function") return null;
  const W = canvas.width | 0;
  const H = canvas.height | 0;
  if (W < 8 || H < 8) return null;

  const tw = 64;
  const th = 48;
  const scratch = getScratch(tw, th);
  if (!scratch) return null;
  let ctx;
  try {
    ctx = scratch.getContext("2d", { willReadFrequently: true });
  } catch (_) {
    ctx = scratch.getContext("2d");
  }
  if (!ctx) return null;
  try {
    ctx.drawImage(canvas, 0, 0, tw, th);
  } catch (_) {
    return null;
  }

  let data;
  try {
    data = ctx.getImageData(0, 0, tw, th).data;
  } catch (_) {
    return null;
  }

  let sumY = 0;
  let sumY2 = 0;
  let sumSat = 0;
  let n = 0;
  const hist = [0, 0, 0, 0, 0]; // 5-bin luma
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const maxc = Math.max(r, g, b);
    const minc = Math.min(r, g, b);
    const sat = maxc > 1e-6 ? (maxc - minc) / maxc : 0;
    sumY += y;
    sumY2 += y * y;
    sumSat += sat;
    hist[Math.min(4, (y * 5) | 0)] += 1;
    n += 1;
  }
  if (!n) return null;

  const meanLuma = sumY / n;
  const variance = Math.max(0, sumY2 / n - meanLuma * meanLuma);
  const contrast = Math.sqrt(variance);
  const sat = sumSat / n;

  // 死黑/死白/灰墙：层次差；有中间调且有一定对比更好
  let tone = 1;
  if (meanLuma < 0.12 || meanLuma > 0.9) tone *= 0.25;
  else if (meanLuma < 0.2 || meanLuma > 0.82) tone *= 0.55;
  else if (meanLuma >= 0.28 && meanLuma <= 0.72) tone *= 1;
  else tone *= 0.8;

  // 对比度过低 = 白墙/雾灰；过高略扣
  if (contrast < 0.04) tone *= 0.2;
  else if (contrast < 0.08) tone *= 0.45;
  else if (contrast < 0.12) tone *= 0.75;
  else if (contrast > 0.32) tone *= 0.85;

  // 直方图占满两端但中间空 → 略扣
  const midShare = (hist[1] + hist[2] + hist[3]) / n;
  if (midShare < 0.25) tone *= 0.7;

  // 色彩：几乎无饱和（白墙/灰桌）低分；适中饱和高分
  let color = 1;
  if (sat < 0.04) color = 0.15;
  else if (sat < 0.08) color = 0.35;
  else if (sat < 0.14) color = 0.65;
  else if (sat <= 0.45) color = 1;
  else if (sat <= 0.65) color = 0.8;
  else color = 0.55;

  return {
    tone: clamp01(tone),
    color: clamp01(color),
    meanLuma: meanLuma,
    sat: sat,
    contrast: contrast,
  };
}

/**
 * 主体框相对周围的颜色分离度（显著性代理）。
 */
export function scoreSeparation(canvas, bbox, imgW, imgH) {
  if (!canvas || !bbox || typeof canvas.getContext !== "function") return 0.45;
  const W = canvas.width | 0;
  const H = canvas.height | 0;
  if (W < 8 || H < 8) return 0.45;

  const tw = 64;
  const th = 48;
  const scratch = getScratch(tw, th);
  if (!scratch) return 0.45;
  let ctx;
  try {
    ctx = scratch.getContext("2d", { willReadFrequently: true });
  } catch (_) {
    ctx = scratch.getContext("2d");
  }
  if (!ctx) return 0.45;
  try {
    ctx.drawImage(canvas, 0, 0, tw, th);
  } catch (_) {
    return 0.45;
  }

  const sx = tw / Math.max(1, imgW || W);
  const sy = th / Math.max(1, imgH || H);
  const x1 = Math.max(0, Math.floor(Math.min(bbox[0], bbox[2]) * sx));
  const y1 = Math.max(0, Math.floor(Math.min(bbox[1], bbox[3]) * sy));
  const x2 = Math.min(tw, Math.ceil(Math.max(bbox[0], bbox[2]) * sx));
  const y2 = Math.min(th, Math.ceil(Math.max(bbox[1], bbox[3]) * sy));
  const bw = x2 - x1;
  const bh = y2 - y1;
  if (bw < 3 || bh < 3) return 0.3;

  let data;
  try {
    data = ctx.getImageData(0, 0, tw, th).data;
  } catch (_) {
    return 0.45;
  }

  function meanRegion(rx1, ry1, rx2, ry2) {
    const a = Math.max(0, rx1);
    const b = Math.max(0, ry1);
    const c = Math.min(tw, rx2);
    const d = Math.min(th, ry2);
    if (c <= a || d <= b) return null;
    let r = 0;
    let g = 0;
    let bl = 0;
    let n = 0;
    for (let y = b; y < d; y++) {
      for (let x = a; x < c; x++) {
        const i = (y * tw + x) * 4;
        r += data[i];
        g += data[i + 1];
        bl += data[i + 2];
        n += 1;
      }
    }
    if (!n) return null;
    return [r / n, g / n, bl / n];
  }

  const pad = Math.max(1, Math.round(Math.min(bw, bh) * 0.15));
  const inner = meanRegion(x1 + pad, y1 + pad, x2 - pad, y2 - pad);
  if (!inner) return 0.4;

  const rings = [
    meanRegion(Math.max(0, x1 - pad), y1, x1, y2),
    meanRegion(x2, y1, Math.min(tw, x2 + pad), y2),
    meanRegion(x1, Math.max(0, y1 - pad), x2, y1),
    meanRegion(x1, y2, x2, Math.min(th, y2 + pad)),
  ];
  let dr = 0;
  let count = 0;
  for (let i = 0; i < rings.length; i++) {
    const o = rings[i];
    if (!o) continue;
    dr +=
      (Math.abs(inner[0] - o[0]) +
        Math.abs(inner[1] - o[1]) +
        Math.abs(inner[2] - o[2])) /
      (3 * 255);
    count += 1;
  }
  if (!count) return 0.4;
  return clamp01((dr / count) * 2.4);
}

function scoreSubjectPresence(subject, areaRatio) {
  if (!subject || !subject.bbox) return 0;
  const conf = clamp01(Number(subject.score) || 0);
  const sizeOk =
    areaRatio < 0.02
      ? areaRatio / 0.02
      : areaRatio > 0.7
        ? Math.max(0.15, 1 - (areaRatio - 0.7) / 0.3)
        : 1;
  // 低置信弱框（白墙误检）压低
  const confPart = conf < 0.25 ? conf * 1.2 : 0.3 + conf * 0.7;
  let presence = clamp01(confPart * (0.45 + 0.55 * sizeOk));
  if (isNonScenicLabel(subject.label)) presence *= 0.55;
  return presence;
}

/**
 * 画面干净度：竞争主体越多越低；空场景不算「干净加分」。
 */
function scoreAntiClutter(objects, primary, imgW, imgH) {
  const list = objects || [];
  if (!primary || !primary.bbox) {
    // 无主体：干净但无聊 — 中低分，靠 presence 拉红
    return 0.4;
  }

  const primaryArea = boxAreaRatio(primary.bbox, imgW, imgH);
  let rivals = 0;
  let rivalArea = 0;
  for (let i = 0; i < list.length; i++) {
    const o = list[i];
    if (!o || !o.bbox) continue;
    if (primary.bbox && boxesSame(o.bbox, primary.bbox)) continue;
    const conf = Number(o.score) || 0;
    const ar = boxAreaRatio(o.bbox, imgW, imgH);
    if (conf < 0.22 || ar < 0.012) continue;
    // IoU 高 → 同一主体碎片，忽略
    if (boxIoU(o.bbox, primary.bbox) > 0.45) continue;
    rivals += 1;
    rivalArea += ar;
  }

  let clean = 1;
  if (rivals === 0) clean = 1;
  else if (rivals === 1) clean = rivalArea > primaryArea * 0.55 ? 0.55 : 0.75;
  else if (rivals === 2) clean = 0.45;
  else if (rivals <= 4) clean = 0.28;
  else clean = 0.12;

  // 桌面/家具类主体本身暗示杂乱室内
  if (isNonScenicLabel(primary.label)) clean *= 0.5;

  return clamp01(clean);
}

function boxesSame(a, b) {
  if (!a || !b) return false;
  return (
    Math.abs(a[0] - b[0]) < 1.5 &&
    Math.abs(a[1] - b[1]) < 1.5 &&
    Math.abs(a[2] - b[2]) < 1.5 &&
    Math.abs(a[3] - b[3]) < 1.5
  );
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
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter <= 0) return 0;
  const uni =
    (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return uni > 0 ? inter / uni : 0;
}

function scoreLevel(tiltDeg, hasTilt) {
  if (!hasTilt || typeof tiltDeg !== "number" || !isFinite(tiltDeg)) {
    return 0.85; // 无传感器：不重罚
  }
  const a = Math.abs(tiltDeg);
  if (a <= 2.5) return 1;
  if (a <= 5) return 0.85;
  if (a <= 8) return 0.55;
  if (a <= 12) return 0.3;
  return 0.12;
}

function multiSubjectGate(objects, primary, imgW, imgH) {
  if (!primary || !primary.bbox) return 1;
  const pArea = boxAreaRatio(primary.bbox, imgW, imgH);
  let bestRival = 0;
  const list = objects || [];
  for (let i = 0; i < list.length; i++) {
    const o = list[i];
    if (!o || !o.bbox) continue;
    if (boxesSame(o.bbox, primary.bbox)) continue;
    if ((Number(o.score) || 0) < 0.28) continue;
    if (boxIoU(o.bbox, primary.bbox) > 0.45) continue;
    const ar = boxAreaRatio(o.bbox, imgW, imgH);
    if (ar > bestRival) bestRival = ar;
  }
  if (bestRival >= pArea * 0.45 && bestRival >= 0.04) {
    return AESTHETIC_GATES.multiSubjectMultiplier;
  }
  return 1;
}

/**
 * @param {Object.<string, number>} factors
 * @param {number} gateMul
 * @param {boolean} hasSubject
 */
function composeWeightedScore(factors, gateMul, hasSubject) {
  const w = AESTHETIC_WEIGHTS;
  let raw = 0;
  let weightSum = 0;
  const keys = Object.keys(w);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const fv = factors[k];
    if (typeof fv !== "number" || !isFinite(fv)) continue;
    raw += fv * w[k];
    weightSum += w[k];
  }
  let score = weightSum > 0 ? raw / weightSum : 0;
  score *= gateMul;
  if (!hasSubject) {
    score = Math.min(score, AESTHETIC_GATES.noSubjectScoreCap);
  }
  return clamp01(score);
}

/**
 * 仅当当前焦段即使构图拉满也只能红档(B)时，才建议换焦。
 */
export function suggestFocalForSubject(
  areaRatio,
  currentZoom,
  focals,
  potentialGrade
) {
  const list = focals || [];
  const cur = currentZoom || 1;
  if (!list.length) {
    return {
      suggestedZoom: cur,
      suggestedMm: 24,
      suggestedName: "广角",
      needFocalChange: false,
    };
  }

  const targetArea = 0.18;
  const safeArea = Math.max(0.02, Math.min(0.55, areaRatio || 0.1));
  const idealZoom = cur * Math.sqrt(targetArea / safeArea);
  let best = list[0];
  let bestDiff = Infinity;
  for (let i = 0; i < list.length; i++) {
    const diff = Math.abs(list[i].zoom - idealZoom);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = list[i];
    }
  }

  const canReachYellowOrBetter =
    potentialGrade === "A" || potentialGrade === "S";
  const zoomGap = Math.abs(cur - best.zoom) / Math.max(best.zoom, 0.5) > 0.18;
  const need = !canReachYellowOrBetter && zoomGap && potentialGrade === "B";

  return {
    suggestedZoom: best.zoom,
    suggestedMm: best.mm,
    suggestedName: best.name,
    needFocalChange: need,
  };
}

/**
 * @param {SubjectBox|null} subject
 * @param {number} imgW
 * @param {number} imgH
 * @param {{
 *   currentZoom?:number,
 *   focals?:{name:string,mm:number,zoom:number}[],
 *   canvas?:HTMLCanvasElement,
 *   objects?:SubjectBox[],
 *   tiltDeg?:number,
 *   hasTilt?:boolean,
 *   frameStats?:{tone:number,color:number}
 * }=} options
 * @returns {CompositionGuide|null}
 */
export function evaluateLandscapeComposition(subject, imgW, imgH, options) {
  if (imgW < 2 || imgH < 2) return null;
  const opts = options || {};
  const hasSubject = !!(subject && subject.bbox);

  let nx = 0.5;
  let ny = 0.5;
  let areaRatio = 0;
  let completeness = 0;
  let placement = 0;
  let size = 0;
  let balance = 0.5;

  if (hasSubject) {
    const b = subject.bbox;
    const c = boxCenter(b);
    nx = c.x / imgW;
    ny = c.y / imgH;
    areaRatio = boxAreaRatio(b, imgW, imgH);
    completeness = boxCompleteness(b, imgW, imgH);
    const rot = scoreRuleOfThirds(nx, ny);
    const cen = scoreCenter(nx, ny);
    placement = Math.max(rot, cen * 0.92);
    size = scoreSize(areaRatio);
    balance = scoreVisualBalance(nx, ny, areaRatio);
  }

  const presence = scoreSubjectPresence(subject, areaRatio);
  const antiClutter = scoreAntiClutter(
    opts.objects,
    subject,
    imgW,
    imgH
  );

  let frame =
    opts.frameStats ||
    (opts.canvas ? sampleFrameAesthetics(opts.canvas) : null);
  const tone = frame ? frame.tone : 0.55;
  const color = frame ? frame.color : 0.55;

  let separation = 0.45;
  if (hasSubject && opts.canvas) {
    separation = scoreSeparation(opts.canvas, subject.bbox, imgW, imgH);
  } else if (!hasSubject) {
    separation = 0.2;
  }

  const level = scoreLevel(opts.tiltDeg, !!opts.hasTilt);

  const factors = {
    subjectPresence: presence,
    separation: separation,
    antiClutter: antiClutter,
    placement: hasSubject ? placement : 0.15,
    balance: hasSubject ? balance : 0.4,
    size: hasSubject ? size : 0.1,
    completeness: hasSubject ? completeness : 0.2,
    tone: tone,
    color: color,
    level: level,
  };

  let gateMul = 1;
  if (hasSubject && isNonScenicLabel(subject.label)) {
    gateMul *= AESTHETIC_GATES.nonScenicMultiplier;
  }
  gateMul *= multiSubjectGate(opts.objects, subject, imgW, imgH);

  const score = composeWeightedScore(factors, gateMul, hasSubject);
  const grade = gradeFromScore(score);

  let borderColor = "rgba(239, 68, 68, 0.72)";
  if (grade === "S") borderColor = "rgba(74, 222, 128, 0.75)";
  else if (grade === "A") borderColor = "rgba(250, 204, 21, 0.75)";

  const target =
    !hasSubject
      ? { x: 0.5, y: 0.5 }
      : areaRatio > 0.28
        ? { x: 0.5, y: 0.5 }
        : nearestPowerPoint(nx, ny);

  // 潜力分：placement 拉满，其它因子保持
  const potentialFactors = Object.assign({}, factors, {
    placement: 1,
    balance: scoreVisualBalance(target.x, target.y, areaRatio || 0.15),
  });
  const potentialScore = composeWeightedScore(
    potentialFactors,
    gateMul,
    hasSubject
  );
  const potentialGrade = gradeFromScore(potentialScore);

  const arrows = { up: 0, down: 0, left: 0, right: 0 };
  if (hasSubject) {
    const ex = nx - target.x;
    const ey = ny - target.y;
    const dead = 0.035;
    if (ex > dead) arrows.right = clamp01((ex - dead) / 0.22);
    if (ex < -dead) arrows.left = clamp01((-ex - dead) / 0.22);
    if (ey > dead) arrows.down = clamp01((ey - dead) / 0.22);
    if (ey < -dead) arrows.up = clamp01((-ey - dead) / 0.22);
  }

  const focal = suggestFocalForSubject(
    areaRatio,
    opts.currentZoom || 1,
    opts.focals || [],
    potentialGrade
  );

  return {
    score: score,
    grade: grade,
    color: borderColor,
    target: target,
    subjectCenter: hasSubject ? { x: nx, y: ny } : null,
    arrows: arrows,
    suggestedZoom: focal.suggestedZoom,
    suggestedMm: focal.suggestedMm,
    suggestedName: focal.suggestedName,
    needFocalChange: hasSubject ? focal.needFocalChange : false,
    areaRatio: areaRatio,
    potentialGrade: potentialGrade,
    factors: factors,
  };
}
