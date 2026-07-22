# Gesto Engine — 轻量级 Web 端手势识别与动态交互系统

`Gesto Engine` 是一个基于 **Google MediaPipe** 与 **前端 KNN 分类器** 的轻量级、高扩展性的 Web 端手势识别与动态交互系统。支持实时骨骼点追踪、自定义手势录制、样本持久化及高精度实时分类匹配。

原作品 **[gestomeme.com](https://gestomeme.com/)**(作者 **@hi.jessie**)

本项目是根据**[gesto-engine](https://github.com/jiaxiwangau-star/gesto-engine)**该项目修改成前端+后端FastAPI

---

## 🚀 功能特性

* **实时骨骼追踪**：依托 Google MediaPipe 强大的 Vision Tasks，实现低延迟的手部与姿态关键点提取。
* **自定义手势录制**：支持在网页端一键录制专属的动态/静态手势，并自动提取多维骨骼特征向量。
* **轻量化 KNN 分类**：基于前端轻量级 K-Nearest Neighbors 算法，无需庞大云端算力即可实现高精度手势识别。
* **数据持久化与预设库**：支持将录制好的手势训练集保存为 `samples.json`，支持本地导入导出及服务端静态托管。
* **前后端分离架构**：前端使用纯原生 JavaScript (ES Modules) + Canvas 渲染，后端采用 FastAPI 提供轻量化静态资源与 API 支持。

---

## 📂 项目结构

```text
gesto-engine-online/
├── backend/
│   ├── static/
│   │   ├── uploads/                  # 用户上传的图片/临时文件
│   │   ├── hand_landmarker.task      # MediaPipe 手部关键点模型
│   │   ├── pose_landmarker_lite.task # MediaPipe 姿态关键点模型
│   │   └── samples.json              # 全量手势训练样本数据集
│   └── main.py                       # FastAPI 后端服务入口
├── engine.js                         # 核心手势引擎（封装 MediaPipe 与 KNN 逻辑）
├── index.html                        # 前端可视化交互面板
├── requirements.txt                  # Python 后端依赖包
└── README.md                         # 项目说明文档
```

## 🛠️ 快速开始

### 1. 环境准备

确保你的本地或服务器环境中已安装：

- **Python 3.8+**
- 现代支持 WebGL 与摄像头调用的浏览器（如 Chrome、Edge、Safari）

### 2. 安装依赖

将项目克隆到本地后安装所需的 Python 依赖：

```linux
# 安装依赖（）
pip install -r ../requirements.txt
```

### 3. 启动后端服务

#### ① 启动后端服务

打开一个终端，进入后端目录并启动 FastAPI 服务（默认端口 `8090`）：

Bash

```
cd backend
python3 -m uvicorn main:app --reload --port 8090
```

#### ② 启动前端托管

打开另一个新终端，返回项目**根目录下**，启动轻量级静态 Web 服务（占用 `3000` 端口）：

Bash

```
# 在项目根目录下启动静态 Web 服务（3000 端口）
python3 -m http.server 3000
```

### 4. 访问

- **浏览器打开**：在浏览器中输入 `http://localhost:3000` 即可开始使用。
- **外网/局域网访问**：若配合 Ngrok 等内网穿透工具，并修改相关代码，可通过生成的 HTTPS 链接在多端进行访问与演示。

## 💡 使用说明

- **授予权限**：首次打开网页时，浏览器会请求调用摄像头的权限，点击“允许”。
- **加载预设**：系统初始化时会自动从后端静态目录加载 `samples.json` 中的手势样本库，右侧“预设手势库”将展示当前支持识别的手势列表。
- **导入数据**：将根目录` sample.json`导入网页
- **识别交互**：选图 / 填 emoji → 起名 → 「录制手势」→ 摆好姿势(准备 ~2s,录制 ~3s)。同名再录会累加样本;记得也录几段「🧘 中性」(放松、不做手势)避免乱识别。做训练过的手势 → 对应图/emoji 就弹出来。数据存浏览器 `localStorage`,可导出 json 备份。

## 📁 文件说明

| 文件                   | 作用                                            |
| ---------------------- | ----------------------------------------------- |
| `index.html`           | 完整 demo(中英双语,训练面板 + 结果显示)         |
| `sample.json`          | 完整样本数据                                    |
| `engine.js`            | 识别核心:特征提取 + KNN + 摄像头循环 + 录制 API |
| `samples.example.json` | 样本数据格式示例                                |
| `main.py`              | 后端数据处理代码                                |

## 🧠 原理 

MediaPipe `@mediapipe/tasks-vision`(pose + hand,lite 模型,CDN 加载)提取关键点 → 22 维特征向量(姿态按肩宽归一 + 双手指状态/捏合)→ 录样本 → KNN 最近邻投票分类。纯前端,无第三方分类库。

