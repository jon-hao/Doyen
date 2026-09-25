/**
 * 风景构图实时评分与引导
 * 依据：
 * - Liu et al. Optimizing Photo Composition（三分法 / 视觉平衡 / 尺寸甜区）
 * - Sensors 2020 Photo Composition with Real-Time Rating（主体重心距三分点）
 * - 实时相机引导：把主体移向最优兴趣点，并建议变焦（ACM MM 2024 类 view adjustment）
 */

/**
 * @typedef {{bbox:number[], score?:number, label?:string}} SubjectBox
 * @typedef {{
 *   score:number,
 *   grade:"B"|"A"|"S",
 *   color:string,
 *   target:{x:number,y:number},
 *   subjectCenter:{x:number,y:number},
 *   arrows:{up:number,down:number,left:number,right:number},
 *   suggestedZoom:number,
 *   suggestedMm:number,
 *   suggestedName:string,
 *   needFocalChange:boolean,
 *   areaRatio:number
 * }} CompositionGuide
 */

const POWER_POINTS = [
  [1 / 3, 1 / 3],
  [2 / 3, 1 / 3],
  [1 / 3, 2 / 3],
  [2 / 3, 2 / 3],
];

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
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

/** 三分法：主体中心距最近力量点（高斯衰减） */
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

/** 中心构图备选（风景主体偏大时更合理） */
function scoreCenter(nx, ny) {
  const dx = nx - 0.5;
  const dy = ny - 0.5;
  const d = Math.sqrt(dx * dx + dy * dy);
  return Math.exp(-(d * d) / (2 * 0.16 * 0.16));
}

/** 尺寸甜区：风景主体约占画面 8%–32% 最佳 */
function scoreSize(areaRatio) {
  if (areaRatio < 0.03) return areaRatio / 0.03 * 0.35;
  if (areaRatio < 0.08) return 0.35 + ((areaRatio - 0.03) / 0.05) * 0.45;
  if (areaRatio <= 0.32) return 1;
  if (areaRatio <= 0.5) return Math.max(0.2, 1 - (areaRatio - 0.32) / 0.18);
  return Math.max(0, 0.2 - (areaRatio - 0.5) * 0.5);
}

/** 视觉平衡：主体「视觉重心」靠近画面中心的和谐度（弱权重） */
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

/**
 * 在可用焦段里选最接近「理想主体占比」的焦距
 * 假设物距不变：zoom ∝ 1/sqrt(area) 近似
 */
export function suggestFocalForSubject(areaRatio, currentZoom, focals) {
  const list = focals || [];
  if (!list.length) {
    return {
      suggestedZoom: currentZoom,
      suggestedMm: 24,
      suggestedName: "广角",
      needFocalChange: false,
    };
  }
  const targetArea = 0.18;
  const safeArea = Math.max(0.02, Math.min(0.55, areaRatio || 0.1));
  const idealZoom = (currentZoom || 1) * Math.sqrt(targetArea / safeArea);
  let best = list[0];
  let bestDiff = Infinity;
  for (let i = 0; i < list.length; i++) {
    const diff = Math.abs(list[i].zoom - idealZoom);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = list[i];
    }
  }
  const need =
    Math.abs((currentZoom || 1) - best.zoom) / Math.max(best.zoom, 0.5) > 0.18;
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
 * @param {{currentZoom?:number, focals?:{name:string,mm:number,zoom:number}[]}=} options
 * @returns {CompositionGuide|null}
 */
export function evaluateLandscapeComposition(subject, imgW, imgH, options) {
  if (!subject || !subject.bbox || imgW < 2 || imgH < 2) return null;
  const b = subject.bbox;
  const c = boxCenter(b);
  const nx = c.x / imgW;
  const ny = c.y / imgH;
  const areaRatio = boxAreaRatio(b, imgW, imgH);
  const completeness = boxCompleteness(b, imgW, imgH);

  const rot = scoreRuleOfThirds(nx, ny);
  const cen = scoreCenter(nx, ny);
  const placement = Math.max(rot, cen * 0.92);
  const size = scoreSize(areaRatio);
  const balance = scoreVisualBalance(nx, ny, areaRatio);

  // 综合：placement 主导，尺寸与完整度次之，平衡微调
  const score = clamp01(
    placement * 0.48 + size * 0.28 + completeness * 0.16 + balance * 0.08
  );

  let grade = "B";
  let color = "rgba(239, 68, 68, 0.72)";
  if (score >= 0.72) {
    grade = "S";
    color = "rgba(74, 222, 128, 0.75)";
  } else if (score >= 0.48) {
    grade = "A";
    color = "rgba(250, 204, 21, 0.75)";
  }

  // 大主体更偏中心构图目标，小主体偏三分法
  const target =
    areaRatio > 0.28
      ? { x: 0.5, y: 0.5 }
      : nearestPowerPoint(nx, ny);

  const ex = nx - target.x;
  const ey = ny - target.y;
  const dead = 0.035;
  const arrows = { up: 0, down: 0, left: 0, right: 0 };
  // 主体相对目标偏哪边 → 镜头朝同侧移动，使主体移向目标
  if (ex > dead) arrows.right = clamp01((ex - dead) / 0.22);
  if (ex < -dead) arrows.left = clamp01((-ex - dead) / 0.22);
  if (ey > dead) arrows.down = clamp01((ey - dead) / 0.22);
  if (ey < -dead) arrows.up = clamp01((-ey - dead) / 0.22);

  const focal = suggestFocalForSubject(
    areaRatio,
    (options && options.currentZoom) || 1,
    (options && options.focals) || []
  );

  return {
    score: score,
    grade: grade,
    color: color,
    target: target,
    subjectCenter: { x: nx, y: ny },
    arrows: arrows,
    suggestedZoom: focal.suggestedZoom,
    suggestedMm: focal.suggestedMm,
    suggestedName: focal.suggestedName,
    needFocalChange: focal.needFocalChange,
    areaRatio: areaRatio,
  };
}
