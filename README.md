# Doyen

网页版 AI 相机：进页环境检测 → 全屏取景拍照；支持保存/分享。「添加到主屏幕」后可作为桌面 Web App 使用。开启 **辅助拍摄** 后，实时勾勒**单一主体**，并在风景模式下给出构图评级与移镜引导。

**当前版本：`0.105`**（见 `version.json`；发版时同步改 `index.html` 内 `APP_VERSION` / `meta doyen-app-version`）

在线使用：https://jon-hao.github.io/Doyen/

## 功能

- 环境检测：HTTPS、相机 API、辅助拍摄可用性
- 全屏取景（JS cover，减少 iOS 黑边）、快门、前后摄像头
- 风景 / 人像焦段预设（默认风景 · 广角）
- **辅助拍摄**：本地免费模型（不联网推理、不付费）+ 风景构图引导；首次下载后可缓存静默热启动
- **桌面图标**：光圈负形 **D**（`icons/` + `manifest.webmanifest`）
- **桌面版自动更新**：启动/回前台拉取 `version.json`，不一致则清缓存并强制刷新
- 拍照预览：全屏铺满 + 半透明「保存 / 取消」

## 发版

1. 同时提高 `version.json` 的 `version` 与 `index.html` 中的 `APP_VERSION` / `meta doyen-app-version`（格式 `0.xxx`）
2. 推送到 GitHub Pages 后，已「添加到主屏幕」的客户端下次打开或切回前台会自动更新
3. 更换图标后，用户可能需删除桌面快捷方式再添加一次

## 辅助拍摄

### 引擎（全部本地、免费）

| 组件 | 体积 | 作用 |
|------|------|------|
| **u2netp**（Apache-2.0） | ~4.4MB | **风景主路径**：显著性主体 |
| **Selfie Segmenter**（MediaPipe） | ~0.24MB | **人像主路径**：人物分割 |
| EfficientDet-Lite0 | ~6.9MB | 候选列表 / 主路径失败时回退 |
| MagicTouch Interactive Segmenter | ~5.9MB | 轮廓修边与点击分割 |
| 构图评分（启发式） | — | 美学加权 + 移镜 / 换焦 |

首次合计约 **17–18MB**（另需 MediaPipe / onnxruntime-web 运行时，走 CDN，与现网一致）。无云端推理、无按次付费。

### 稳定单体管线

1. **模式分流定主** — 风景 u2netp / 人像 Selfie；失败才回退 EfficientDet  
2. **检测降频 + 跟踪升频** — 显著性约 420ms；中间帧分割跟踪  
3. **NMS + 并集框剔除 + 锁定独占** — 强制单主体  
4. **掩码连通域 + EMA** — 去粘连、减闪烁  
5. **丢失保持** — holding 后提示点选  

### 交互

- **单击**：对焦框动画；能力允许时尝试硬件点按对焦（多为 Android）；辅助开启时**选定主体**  
- 不做单独长按对焦  
- 切换焦距 / 镜头 / 模式 / 横竖握：强制重检  

### 风景构图引导

权重与硬门槛同前（EVA/AADB + Liu）。风景模式：贴边色框、移镜箭头、换焦建议、水平提醒。

### 模式策略

| 模式 | 定主模型 | 说明 |
|------|----------|------|
| **风景** | u2netp 显著性 | 最大显著连通域；点击可锁指定区域 |
| **人像** | Selfie Segmenter | 人物掩码；OD 仅作回退 |

### 使用注意

- 请用**后置镜头**
- 推荐较新的移动 Safari / Chrome；首次需联网拉模型与 WASM 运行时  
- iPhone 上「选主体」不保证光学合焦（Safari 无点按对焦 API）

## 本地文件

| 路径 | 说明 |
|------|------|
| `version.json` | 线上版本号 |
| `index.html` | 相机 UI |
| `coach-engine.js` | 加载/调度/锁定/跟踪 |
| `saliency-local.js` | Selfie + u2netp 本地推理 |
| `composition-coach.js` | 风景美学评分 |
| `manifest.webmanifest` / `icons/` | PWA 图标 |
| `models/efficientdet_lite0.tflite` | 目标检测 |
| `models/magic_touch.tflite` | 交互分割 |
| `models/selfie_segmenter.tflite` | 人像分割 |
| `models/u2netp.onnx` | 风景显著性 |

## License

本项目为专有软件，**未开源**。详见 [LICENSE](./LICENSE)。  
第三方模型：MediaPipe Selfie Segmenter；U²-Net portable（u2netp，Apache-2.0）。运行时：MediaPipe Tasks Vision、ONNX Runtime Web（MIT）。
