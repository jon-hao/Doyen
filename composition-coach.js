/**
 * 风景构图实时评分与引导
 * 依据：
 * - Liu et al. Optimizing Photo Composition（三分法 / 视觉平衡 / 尺寸甜区）
 * - Sensors 2020 Photo Composition with Real-Time Rating（主体重心距三分点）
 * - EVA / AADB 美学属性相对重要性（composition+depth > light+color > quality）
 * - 实时相机引导：移镜 / 变焦建议
 *
 * 权重定稿（合计 1.00），来源摘要：
 * - EVA：观察者自评中 composition&depth 影响最大，light&color 略次，quality/semantic 再次
 * - AADB+XAI：interesting content / object emphasis 与构图类属性对总分贡献靠前
 * - Liu：placement + size + balance 为可计算构图核心
 * - 实拍问题（白墙/杂乱桌/马桶拿绿）：提高「主体存在 + 分离 + 反杂乱」权重，压低纯几何构图
 */

/**
 * 美学评分权重（定稿）。实现新 skill 时按此加权；未实现的项暂不计入总分。
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
  /** 清晰可辨主体（无主体/空墙应接近 0）— 对应 semantic / interesting content */
  subjectPresence: 0.16,
  /** 主体与背景分离（对比/显著性）— object emphasis */
  separation: 0.12,
  /** 画面干净度（杂乱/多竞争体越低）— 构图组织失败的主要惩罚项 */
  antiClutter: 0.14,
  /** 三分法 / 中心构图放置 — composition 核心 */
  placement: 0.14,
  /** 视觉平衡 — Liu / Gestalt balance */
  balance: 0.08,
  /** 主体尺寸甜区 — Liu size region */
  size: 0.08,
  /** 主体完整度（少裁切） */
  completeness: 0.06,
  /** 明暗层次 / 曝光不过死 — light */
  tone: 0.10,
  /** 色彩丰富度与和谐 — color */
  color: 0.07,
  /** 画面水平 — quality / stability */
  level: 0.05,
});

/** 硬门槛（不占软权重，直接封顶或乘子） */
export const AESTHETIC_GATES = Object.freeze({
  /** 无有效主体时总分上限（红档） */
  noSubjectScoreCap: 0.4,
  /** 风景模式「非风景意图」类别（马桶/洗手池等）乘子 */
  nonScenicMultiplier: 0.45,
  /** 单一焦点：次主体面积接近主角时的乘子下限 */
  multiSubjectMultiplier: 0.7,
});

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
 *   areaRatio:number,
 *   potentialGrade:"B"|"A"|"S"
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

/**
 * 当前已实现子项按定稿权重归一化加权。
 * 未实现项（分离/杂乱/光色/水平）待 skill 落地后再并入。
 */
function composeScore(placement, size, completeness, balance) {
  const w = AESTHETIC_WEIGHTS;
  const active =
    w.placement + w.size + w.completeness + w.balance;
  const raw =
    placement * w.placement +
    size * w.size +
    completeness * w.completeness +
    balance * w.balance;
  return clamp01(raw / active);
}

/** 仅靠移镜能达到的上限分（placement 满分） */
function estimatePotentialScore(areaRatio, completeness, target) {
  const size = scoreSize(areaRatio);
  const balance = scoreVisualBalance(target.x, target.y, areaRatio);
  return composeScore(1, size, completeness, balance);
}

/**
 * 仅当当前焦段即使构图拉满也只能红档(B)时，才建议换焦。
 * 有望到黄(A)/绿(S) → 不推荐换焦。
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

  const score = composeScore(placement, size, completeness, balance);
  const grade = gradeFromScore(score);

  let color = "rgba(239, 68, 68, 0.72)";
  if (grade === "S") color = "rgba(74, 222, 128, 0.75)";
  else if (grade === "A") color = "rgba(250, 204, 21, 0.75)";

  const target =
    areaRatio > 0.28
      ? { x: 0.5, y: 0.5 }
      : nearestPowerPoint(nx, ny);

  const potentialScore = estimatePotentialScore(
    areaRatio,
    completeness,
    target
  );
  const potentialGrade = gradeFromScore(potentialScore);

  const ex = nx - target.x;
  const ey = ny - target.y;
  const dead = 0.035;
  const arrows = { up: 0, down: 0, left: 0, right: 0 };
  if (ex > dead) arrows.right = clamp01((ex - dead) / 0.22);
  if (ex < -dead) arrows.left = clamp01((-ex - dead) / 0.22);
  if (ey > dead) arrows.down = clamp01((ey - dead) / 0.22);
  if (ey < -dead) arrows.up = clamp01((-ey - dead) / 0.22);

  const focal = suggestFocalForSubject(
    areaRatio,
    (options && options.currentZoom) || 1,
    (options && options.focals) || [],
    potentialGrade
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
    potentialGrade: potentialGrade,
  };
}
