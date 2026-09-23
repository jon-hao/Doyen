/**
 * 根据视觉模型结果 + 拍摄模式/焦段，生成中文艺术指导
 */
export function buildCoachTips(vision, context) {
  const caption = String((vision && vision.caption) || "").trim();
  const objects = (vision && vision.objects) || [];
  const mode = (context && context.mode) || "landscape";
  const focal = (context && context.focal) || { name: "广角", mm: 24 };
  const lower = caption.toLowerCase();
  const labels = objects.map(function (o) {
    return String(o.label || "").toLowerCase();
  });

  function has() {
    const words = Array.prototype.slice.call(arguments);
    return words.some(function (w) {
      return lower.includes(w) || labels.some(function (l) {
        return l.includes(w);
      });
    });
  }

  function subjectBox() {
    const priority = [
      "person",
      "man",
      "woman",
      "boy",
      "girl",
      "human",
      "face",
    ];
    for (let i = 0; i < objects.length; i++) {
      const lab = String(objects[i].label || "").toLowerCase();
      if (priority.some(function (p) { return lab.includes(p); })) {
        return objects[i].bbox;
      }
    }
    return objects[0] && objects[0].bbox ? objects[0].bbox : null;
  }

  /** bbox: [x1,y1,x2,y2] normalized or absolute — Florence often absolute on image size */
  function boxRatio(bbox) {
    if (!bbox || bbox.length < 4) return null;
    const w = Math.abs(bbox[2] - bbox[0]);
    const h = Math.abs(bbox[3] - bbox[1]);
    const cx = (bbox[0] + bbox[2]) / 2;
    const cy = (bbox[1] + bbox[3]) / 2;
    // 若坐标很大，当作像素；用相对粗略估计：中心偏离与面积
    return { w: w, h: h, cx: cx, cy: cy, area: w * h };
  }

  const tips = [];
  const box = boxRatio(subjectBox());

  if (mode === "portrait") {
    if (!has("person", "man", "woman", "boy", "girl", "face", "people", "human")) {
      tips.push("人像模式里主体还不清晰：让人物占据画面更明确的位置，或靠近三分点。");
    } else {
      tips.push("人物已入画：试着让眼睛落在上三分线附近，并留出视线方向的空间。");
    }
    if (focal.mm < 45) {
      tips.push(
        "当前约 " +
          focal.mm +
          "mm 偏环境人像；想更突出面部，可切到 50mm / 85mm。"
      );
    } else if (focal.mm >= 85) {
      tips.push("长焦利于压缩背景：与人物保持距离，背景会更干净、更有分离感。");
    }
    if (has("crowd", "people", "group") && !has("portrait")) {
      tips.push("周围干扰偏多：换个角度躲开杂物，或再走近一步简化背景。");
    }
  } else {
    if (has("sky", "cloud", "mountain", "sea", "ocean", "lake", "beach", "field", "forest")) {
      tips.push("大场景不错：用前景（石头、枝叶、路面）压一层，画面会更有纵深。");
    } else {
      tips.push("风景模式可找一条引导线（路、河、栏杆）把视线带向远方。");
    }
    if (focal.mm <= 16) {
      tips.push("超广会夸张近大远小：把有意思的前景放近一点，远处层次会更戏剧。");
    } else if (focal.mm >= 50) {
      tips.push("中长焦适合「抽取」风景局部：对准层次或光影交界，比硬拍全景更耐看。");
    }
    if (has("building", "skyscraper", "tower", "bridge")) {
      tips.push("拍建筑时注意垂直线：机位尽量端正，或刻意仰拍但让主体居中稳定。");
    }
  }

  if (has("dark", "night", "dim", "shadow") && !has("sunset", "sunrise", "golden")) {
    tips.push("光线偏暗：转向有光的一侧，或等主体进入亮部再按快门。");
  }
  if (has("backlit", "silhouette", "against the light")) {
    tips.push("逆光很有氛围：对主体脸部测光，或保留剪影但让轮廓干净。");
  }
  if (has("clutter", "messy", "wire", "trash", "pole")) {
    tips.push("杂物抢戏：挪半步改背景，比后期裁切更干净。");
  }

  // 主体位置粗判（像素坐标时用相对中心）
  if (box && box.cx > 0 && box.cy > 0) {
    // 无图像尺寸时只能启发式：若 bbox 数值 < 2 当作归一化
    const norm = box.cx <= 1.5 && box.cy <= 1.5;
    const nx = norm ? box.cx : null;
    if (nx !== null) {
      if (nx > 0.38 && nx < 0.62) {
        tips.push("主体偏居中：试着移到左/右三分点，立刻更有叙事感。");
      }
    }
  }

  if (objects.length >= 6) {
    tips.push("元素偏多：选一个主角，其余当配角——减法构图往往更高级。");
  }

  if (!tips.length) {
    tips.push(
      "先稳住水平线，再决定要讲的故事：靠近、走开、或换焦段，三者只改一个。"
    );
  }

  // 去重并最多 3 条
  const uniq = [];
  tips.forEach(function (t) {
    if (uniq.indexOf(t) === -1) uniq.push(t);
  });

  return {
    caption: caption,
    tips: uniq.slice(0, 3),
    objectCount: objects.length,
  };
}
