# Doyen

网页版相机：先做环境检测，再进入拍摄；可分享/保存到相册。

## 功能

- 进页检测：HTTPS、相机 API、WebGPU、Transformers.js CDN、hf-mirror
- 全屏取景、快门拍照、前后摄像头
- 风景 / 人像焦段预设（默认风景 · 广角 24mm）
- **辅助拍摄**：本地 Florence-2 看当前取景并给出建议（需检测通过；桌面 Chrome 最稳）

## 辅助拍摄

- 引擎：Transformers.js + Florence-2-base-ft（首次下载约数百 MB）
- 推荐 **桌面 Chrome / Edge**（WebGPU）
- iPhone Safari 常因 WebGPU / 大模型加载失败；检测页会提前标红

## License

本项目为专有软件，**未开源**。详见 [LICENSE](./LICENSE)。

## 在线使用

https://jon-hao.github.io/Doyen/
