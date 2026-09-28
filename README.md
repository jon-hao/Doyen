# Doyen

网页版 AI 相机：进页环境检测 → 全屏取景拍照；支持保存/分享。「添加到主屏幕」后可作为桌面 Web App 使用。开启 **辅助拍摄** 后，实时勾勒**单一主体**，并在风景模式下给出构图评级与移镜引导。

**当前版本：`0.103`**（见 `version.json`；发版时同步改 `index.html` 内 `APP_VERSION` / `meta doyen-app-version`）

在线使用：https://jon-hao.github.io/Doyen/

## 功能

- 环境检测：HTTPS、相机 API、辅助拍摄可用性
- 全屏取景（JS cover，减少 iOS 黑边）、快门、前后摄像头
- 风景 / 人像焦段预设（默认风景 · 广角）
- **辅助拍摄**：本地轻量检测 + 轮廓分割 + 风景构图引导；首次约 13MB，可缓存后静默热启动
- **桌面图标**：光圈负形 **D**（`icons/` + `manifest.webmanifest`）；Safari「添加到主屏幕」后显示
- **桌面版自动更新**：启动/回前台时拉取 `version.json`，与本地版本不一致则清缓存并强制刷新
- 拍照预览：全屏铺满 + 半透明「保存 / 取消」（不按快门即存）

## 发版

1. 同时提高 `version.json` 的 `version` 与 `index.html` 中的 `APP_VERSION` / `meta doyen-app-version`（格式 `0.xxx`）
2. 推送到 GitHub Pages 后，已「添加到主屏幕」的客户端下次打开或切回前台会自动更新
3. 更换图标后，用户可能需删除桌面快捷方式再添加一次才能看到新图标

## 辅助拍摄

### 引擎

| 组件 | 体积 | 作用 |
|------|------|------|
| EfficientDet-Lite0 | ~6.9MB | 目标检测，候选主体 |
| MagicTouch Interactive Segmenter | ~5.9MB | 点击/中心点交互式轮廓掩码 |
| 构图评分（启发式，无额外模型） | — | 美学加权打分 + 移镜 / 换焦建议 |

- 首次下载显示「下载资源 x%」；已缓存则进页静默热启动，点一次即可开关
- 轮廓：绿色描边 + 更浅的半透明绿色填充

### 单一主体管线（强制只框一个）

检测器常会给出多框或「两人并一框」。Doyen 在后处理中强制单体：

1. **NMS 去重** — 重叠框只留最高分  
2. **并集大框剔除** — 框内有 ≥2 个检测中心，或明显「包子框」的父框丢弃  
3. **Winner-takes-all** — 显著度只取第一名；若第一名包住更紧的次优则改选次优  
4. **掩码单连通域** — 只保留种子点所在连通块，再裁到框内（避免粘连描边）  
5. **跟踪防胀框** — 稳定跟踪时拒绝突然胀成并集大框  

用户点选锁定的主体优先级最高。

### 交互

- 开启后点画面：附近有可识别主体则锁定（优先级最高）
- 取景区点击：类似 iPhone 的半透明对焦框闪现（按钮区域不触发）
- 切换焦距 / 镜头 / 风景·人像 / 横竖握：清空旧结果并重新分析
- 辅助开启时切换前后镜头会先关闭辅助再切换
- 镜头小幅晃动：框体死区稳定，避免抖动

### 风景构图引导

算法参考：

- Liu et al. *Optimizing Photo Composition*（三分法、视觉平衡、尺寸区域）
- Sensors 2020 *Photo Composition with Real-Time Rating*（主体重心距三分点）
- EVA / AADB 美学属性相对重要性（构图+景深 > 光色 > 画质）

**软权重（合计 1.00）**

| 因子 | 权重 |
|------|------|
| 主体存在 subjectPresence | 0.16 |
| 反杂乱 antiClutter | 0.14 |
| 三分/放置 placement | 0.14 |
| 主体分离 separation | 0.12 |
| 明暗 tone | 0.10 |
| 视觉平衡 balance | 0.08 |
| 尺寸甜区 size | 0.08 |
| 色彩 color | 0.07 |
| 完整度 completeness | 0.06 |
| 水平 level | 0.05 |

**硬门槛**：无主体封顶红档（≤0.40）；马桶/洗手池等非风景意图 ×0.45；多竞争主体 ×0.7。

开启辅助且处于**风景模式**时：

1. **8px 贴边细框**随评级变红 / 黄 / 绿；无主体、空墙、杂乱场景也会给红框  
2. **四角双箭头**：仅在需要往该方向移镜时出现  
3. **换焦建议**：仅当当前焦段即使移镜也只能到红档时才提示；有望到黄/绿则不推荐换焦  
4. **水平提醒**：加速度计测倾斜；旋转锁定时仍可识别物理横/竖握  

### 模式策略

| 模式 | 框什么 | 选主体依据 |
|------|--------|------------|
| **风景** | 单一独立主体（不优先路人） | NMS → 去并集框 → 显著度选一 → 掩码单连通域 |
| **人像** | 只框一个人 | 同上；多人时 winner-takes-all |

### 使用注意

- 请用**后置镜头**
- 推荐较新的移动 Safari / Chrome；首次需联网拉模型
- 本项目跑在浏览器 WASM/WebGL，稳定度与原生 App（Neural Engine + 系统 Vision）仍有差距，见下节

### 与原生 AI 相机（如浅影）的技术差异

公开资料里，浅影等原生 App **未公开**具体模型权重与管线。结合其产品形态（原生 iOS、长按选主体、实时构图框）与业界同类实现，稳定主体识别通常依赖：

- **系统级实例分割**：如 Apple Vision `VNGenerateForegroundInstanceMaskRequest`（class-agnostic 前景实例掩码，可按点击取单一 instance）
- **专用人像 API**：`VNGeneratePersonInstanceMaskRequest` 等，多人分开编号
- **时序跟踪**：跨帧关联 + 平滑，而不是每帧独立重检
- **端侧 NPU / Neural Engine**：比浏览器 WASM 的 EfficientDet-Lite 更稳、更快

Doyen 当前为 **Web 可部署**路线：EfficientDet-Lite（COCO 检测）+ MagicTouch（交互分割）+ 启发式单体后处理。要逼近原生稳定度，中长期可考虑：

1. 换用 / 叠加 **显著性或实例分割**模型（非「检出所有 COCO 类」）  
2. 加强 **跨帧 tracker**（IoU / 卡尔曼 / BYTE-Track 类）  
3. 若做原生壳：直接调用系统 Vision / ML Kit Subject Segmentation  

## 本地文件

| 路径 | 说明 |
|------|------|
| `version.json` | 线上版本号（桌面 Web App 自动更新） |
| `index.html` | 相机 UI、构图引导叠加层 |
| `coach-engine.js` | 检测 + 分割 + 单体管线 + 点击锁定 / 防抖 |
| `composition-coach.js` | 风景美学加权评分、移镜箭头、焦段建议 |
| `manifest.webmanifest` | PWA 名称与图标 |
| `icons/` | 正式图标（`apple-touch-icon` / 192 / 512 / favicon）；`icons/concepts/` 为本地概念稿，不入库 |
| `models/efficientdet_lite0.tflite` | 目标检测 |
| `models/magic_touch.tflite` | 交互式轮廓分割 |

## License

本项目为专有软件，**未开源**。详见 [LICENSE](./LICENSE)。
