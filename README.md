# Doyen

网页版相机：打开即拍摄页，底部快门拍照，可分享/保存到相册。

## 功能

- 进入即取景（权限失败时轻触重试）
- 前后摄像头切换（前置时隐藏焦段控制）
- 拍摄模式：
  - **风景**（默认）：超广 13mm / 广角 24mm / 中景 35mm / 压缩 70mm
  - **人像**：环境 35mm / 标准 50mm / 经典 85mm / 特写 135mm
- **导演指导**：本地 Florence-2 看当前取景，给出艺术拍摄建议（首次下载模型，之后可离线推理）
- 默认 **风景 · 广角 24mm**

## 导演指导说明

- 引擎：[Transformers.js](https://huggingface.co/docs/transformers.js) + [Florence-2-base-ft](https://huggingface.co/onnx-community/Florence-2-base-ft)（约数百 MB）
- 需要 **WebGPU**（推荐桌面 Chrome / Edge；部分新版 Safari 可用）
- 点「导演」→ 截取当前帧 → 本地看图 → 中文建议
- 画面与推理都在浏览器内，不上传服务器

## License

本项目为专有软件，**未开源**。详见 [LICENSE](./LICENSE)。

## 在线使用

https://jon-hao.github.io/Doyen/
