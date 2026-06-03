## <div align="center"> **FireRedVAD-ASR-SRT 🎙️🎬**</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align="center">

**高级语音识别与字幕生成工具**

</div>

集成 FireRedVAD 与多种 ASR 模型，支持单文件或批量输出 SRT 字幕。

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## 🌟 核心特性

- 🎯 **多模型支持**: SenseVoiceSmall、Paraformer、Fun-ASR-Nano、Fun-ASR-MLT-Nano、Granite-Speech、GLM-ASR-Nano
- 🎭 **多语言界面**: 英文、中文、日文（轻松扩展）
- 📝 **批量处理**: 支持单文件或批量转录
- ⚡ **高性能**: 优化支持 CPU 和 GPU 加速
- 🎛️ **灵活配置**: 可调节 VAD 参数和模型设置
- 📊 **丰富输出**: SRT 字幕格式，带时间戳

## 🚀 快速开始

### 1. 环境配置

使用 **uv** 创建虚拟环境（推荐）：

```bash
uv venv --python 3.12
```

### 2. 安装依赖

使用 uv 安装：

```bash
uv pip install -r requirements.txt
```

### 3. 模型配置

#### 下载并配置模型：

- **SenseVoiceSmall**: 自动下载
- **Fun-ASR-Nano**: 自动下载
- **GLM-ASR-Nano**: 自动下载
- **Granite-Speech**: 支持语言：英语、法语、德语、西班牙语、葡萄牙语、日语。下载地址：[modelscope](https://www.modelscope.cn/models/ibm-granite/granite-speech-4.1-2b) 或 [Huggingface](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- **VAD 模型**: FireRedVAD文件夹

### 4. 硬件加速

#### 🎮 NVIDIA 独显（CUDA）：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 💻 仅 CPU：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. 运行应用

```bash
uv run main.py
```

## 🌐 多语言支持（gradio）

应用自动检测系统语言：

- 🇺🇸 **英文**: 自动切换到英文界面
- 🇨🇳 **中文**: 自动切换到中文界面
- 🇯🇵 **日文**: 自动切换到日文界面
- ➕ **易于扩展**: 只需在 `locales/` 中添加 JSON 文件即可支持新语言

#### 指定语言：

```bash
uv run main.py --lang=en    # 英文界面
uv run main.py --lang=zh    # 中文界面
uv run main.py --lang=ja    # 日文界面
```

## 📁 项目结构

```
FireRed/
├── 📄 main.py                 # 主程序
├── 📄 ctc.py                  # CTC 模块
├── 📄 model.py                # Fun-ASR 模型代码
├── 📄 requirements.txt        # Python 依赖
├── 📁 asr/                    # ASR 包源码
├── 📁 FireRedVAD/             # 语音活动检测模型文件
├── 📁 FunAudioLLM/            # Fun-ASR 本地模型目录
├── 📁 ibm-granite/            # Granite Speech 模型文件
├── 📁 iic/                    # 额外语音模型目录
├── 📁 locales/                # 语言翻译文件
├── 📁 tools/                  # 实用脚本和工具
├── 📁 utils/                  # 辅助模块
├── 📁 ZhipuAI/                # 额外本地模型目录
├── 📄 README.md               # 本文件
├── 📄 README_ZH.md            # 中文版
└── 📄 README_JA.md            # 日文版
```

## 🎯 使用技巧

> <span style="color: red;"> **重要提示**</span>: 进行批量转录时，建议先做单文件转录测试，以找到最佳 VAD 参数并确保断句准确。VAD参数请优先调整：Min Silence Frames，以获得更好的断句效果。

## 🤝 贡献指南

欢迎各类贡献！包括：

- 🐛 报告问题
- 💡 提出建议
- 🌍 添加翻译
- 🔧 改进代码

## 📄 开源协议

本项目基于 MIT 协议开源。详情请查看 [LICENSE](LICENSE)。

## 📸 界面预览

![主界面](main.png)
