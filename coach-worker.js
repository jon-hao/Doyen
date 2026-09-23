/**
 * Florence-2 Web Worker — 本地看图（首次需下载模型，之后可离线）
 * 默认走 hf-mirror，避免国内访问 huggingface.co 卡住。
 */
import {
  env,
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
  full,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/+esm";

// 浏览器缓存 + 国内镜像（可显著避免“一直加载、进度不动”）
env.allowLocalModels = false;
env.useBrowserCache = true;
env.remoteHost = "https://hf-mirror.com";

const MODEL_ID = "onnx-community/Florence-2-base-ft";

self.postMessage({ status: "boot", data: "Worker 已启动" });

async function hasWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (_) {
    return false;
  }
}

async function hasFp16() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!(adapter && adapter.features && adapter.features.has("shader-f16"));
  } catch (_) {
    return false;
  }
}

function forwardProgress(info) {
  // 原样转发，主线程统一解析 progress / progress_total
  try {
    self.postMessage(
      Object.assign({ channel: "hf-progress" }, info || { status: "unknown" })
    );
  } catch (_) {}
}

class Florence2Singleton {
  static async getInstance(progress_callback) {
    const cb = progress_callback || forwardProgress;

    self.postMessage({ status: "loading", data: "下载处理器…" });
    this.processor ??= AutoProcessor.from_pretrained(MODEL_ID, {
      progress_callback: cb,
    });

    self.postMessage({ status: "loading", data: "下载分词器…" });
    this.tokenizer ??= AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: cb,
    });

    this.supports_fp16 ??= await hasFp16();
    const useGpu = await hasWebGPU();
    if (!useGpu) {
      throw new Error(
        "当前浏览器没有可用的 WebGPU。请用桌面 Chrome / Edge 打开；部分手机浏览器暂不支持。"
      );
    }

    self.postMessage({ status: "loading", data: "下载视觉模型（体积较大）…" });
    this.model ??= Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: {
        embed_tokens: this.supports_fp16 ? "fp16" : "fp32",
        vision_encoder: this.supports_fp16 ? "fp16" : "fp32",
        encoder_model: "q4",
        decoder_model_merged: "q4",
      },
      device: "webgpu",
      progress_callback: cb,
    });

    return Promise.all([this.model, this.tokenizer, this.processor]);
  }
}

let vision_inputs = null;
let image_size = null;

async function load() {
  self.postMessage({ status: "loading", data: "检查 WebGPU…" });

  try {
    if (!(await hasWebGPU())) {
      self.postMessage({
        status: "error",
        error:
          "当前浏览器不支持 WebGPU，无法运行本地视觉模型。请用桌面版 Chrome 或 Edge。",
      });
      return;
    }

    self.postMessage({
      status: "loading",
      data: "开始拉取模型（首次约数百 MB，请保持网络畅通）…",
    });

    const [model, tokenizer] = await Florence2Singleton.getInstance(
      forwardProgress
    );

    self.postMessage({
      status: "loading",
      data: "编译着色器（第一次会稍慢）…",
    });

    const text_inputs = tokenizer("a");
    const pixel_values = full([1, 3, 768, 768], 0.0);
    await model.generate({
      ...text_inputs,
      pixel_values,
      max_new_tokens: 1,
    });

    self.postMessage({ status: "ready" });
  } catch (err) {
    self.postMessage({
      status: "error",
      error: (err && err.message) || "模型加载失败",
    });
  }
}

async function runTask(task) {
  const [model, tokenizer, processor] = await Florence2Singleton.getInstance();
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

async function analyze(data) {
  const start = performance.now();
  try {
    let image;
    if (data && data.buffer) {
      const blob = new Blob([data.buffer], {
        type: data.mime || "image/jpeg",
      });
      image = await RawImage.fromBlob(blob);
    } else if (data && data.url) {
      image = await RawImage.fromURL(data.url);
    } else {
      throw new Error("没有可分析的画面");
    }

    image_size = image.size;
    const [, , processor] = await Florence2Singleton.getInstance();
    vision_inputs = await processor(image);

    self.postMessage({ status: "loading", data: "正在理解画面…" });
    const captionTask = "<MORE_DETAILED_CAPTION>";
    const odTask = "<OD>";
    const captionResult = await runTask(captionTask);
    const odResult = await runTask(odTask);

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

    vision_inputs = null;
    image_size = null;

    self.postMessage({
      status: "complete",
      result: {
        caption: String(caption || "").trim(),
        objects: objects,
        time: performance.now() - start,
      },
    });
  } catch (err) {
    vision_inputs = null;
    image_size = null;
    self.postMessage({
      status: "error",
      error: (err && err.message) || "分析失败",
    });
  }
}

self.addEventListener("message", async function (e) {
  const msg = e.data || {};
  try {
    if (msg.type === "load") await load();
    else if (msg.type === "analyze") await analyze(msg.data || {});
  } catch (err) {
    self.postMessage({
      status: "error",
      error: (err && err.message) || "Worker 异常",
    });
  }
});
