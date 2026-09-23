/**
 * Florence-2 Web Worker — 本地看图（首次需下载模型，之后可离线）
 */
import {
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
  full,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/+esm";

const MODEL_ID = "onnx-community/Florence-2-base-ft";

async function hasFp16() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!(adapter && adapter.features && adapter.features.has("shader-f16"));
  } catch (_) {
    return false;
  }
}

class Florence2Singleton {
  static async getInstance(progress_callback) {
    this.processor ??= AutoProcessor.from_pretrained(MODEL_ID);
    this.tokenizer ??= AutoTokenizer.from_pretrained(MODEL_ID);
    this.supports_fp16 ??= await hasFp16();
    this.model ??= Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: {
        embed_tokens: this.supports_fp16 ? "fp16" : "fp32",
        vision_encoder: this.supports_fp16 ? "fp16" : "fp32",
        encoder_model: "q4",
        decoder_model_merged: "q4",
      },
      device: "webgpu",
      progress_callback,
    });
    return Promise.all([this.model, this.tokenizer, this.processor]);
  }
}

let vision_inputs = null;
let image_size = null;

async function load() {
  if (!navigator.gpu) {
    self.postMessage({
      status: "error",
      error: "当前浏览器不支持 WebGPU，无法运行本地视觉模型。请用桌面 Chrome / Edge，或较新的 Safari。",
    });
    return;
  }

  self.postMessage({ status: "loading", data: "正在加载视觉模型…" });

  try {
    const [model, tokenizer, processor] = await Florence2Singleton.getInstance(
      function (x) {
        self.postMessage(x);
      }
    );

    self.postMessage({
      status: "loading",
      data: "正在编译着色器（首次稍慢）…",
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

async function analyze({ url }) {
  const start = performance.now();
  try {
    const image = await RawImage.fromURL(url);
    image_size = image.size;
    vision_inputs = await processorFor(image);

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

async function processorFor(image) {
  const [, , processor] = await Florence2Singleton.getInstance();
  return processor(image);
}

self.addEventListener("message", async function (e) {
  const msg = e.data || {};
  if (msg.type === "load") await load();
  else if (msg.type === "analyze") await analyze(msg.data || {});
});
