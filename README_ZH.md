# <div align="center">SenseVoice-SRT 🎙️🎬</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align="center">

**高级语音识别与字幕生成工具**

</div>

基于SenseVoice官方WebUI修改而来，支持单文件或批量输出SRT字幕，可选择ASR模型。

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## 🌟 核心特性

- 🎯 **多模型支持**: SenseVoiceSmall、Whisper、Paraformer、Fun-ASR-Nano-2512、Fun-ASR-MLT-Nano-2512
- 🎭 **多语言界面**: 英文、中文、日文（轻松扩展）
- 📝 **批量处理**: 单文件或批量转录功能
- ⚡ **高性能**: 优化支持CPU和GPU加速
- 🎛️ **灵活配置**: 可调节静音阈值和模型设置
- 📊 **丰富输出**: SRT字幕格式，带时间戳

## <div align="center">🚀 快速开始</div>

### 1. 环境配置

使用 **uv** 创建虚拟环境（推荐）：

```bash
uv venv --python 3.12
```

### 2. 安装依赖

使用uv安装：

```bash
uv pip install -r requirements.txt
# 或者
uv add -r requirements.txt
```

### 3. 模型配置

#### 下载并配置模型：

- **SenseVoiceSmall**: 设置 `disable_update=False` 自动下载
- **Whisper模型**: 需要额外安装：`uv pip install -U openai-whisper`
- **VAD模型**: 配置 `fsmn_vad` 路径

下载完成后，设置 `disable_update=True` 减少启动时间。

### 4. 硬件加速

#### 🎮 NVIDIA独显（CUDA）：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 💻 仅CPU：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. 运行应用

```bash
uv run main.py
```

## 🌐 多语言支持

应用自动检测系统语言：

- 🇺🇸 **英文**: 自动切换到英文界面
- 🇨🇳 **中文**: 自动切换到中文界面
- 🇯🇵 **日文**: 自动切换到日文界面
- ➕ **易于扩展**: 只需在 `locales/` 中添加JSON文件即可支持新语言

### 语言选项

#### 强制指定语言：

```bash
uv run main.py --lang=en    # 英文界面
uv run main.py --lang=zh    # 中文界面
uv run main.py --lang=ja    # 日文界面
```

## 📁 项目结构

```
SenseVoice-SRT/
├── 📄 main.py                 # 主程序
├── 📁 utils/                  # 工具模块
│   └── 🌐 translator.py        # 多语言支持
├── 📁 locales/                # 语言翻译
│   ├── 🇺🇸 en.json          # 英文
│   ├── 🇨🇳 zh.json          # 中文
│   └── 🇯🇵 ja.json          # 日文
├── 📄 requirements.txt         # 依赖列表
├── 📄 README.md              # 本文件
└── 📄 README_ZH.md           # 中文版
```

## ⚙️ 配置说明

### 支持的模型：

- **SenseVoiceSmall**: 快速、准确的多语言ASR
- **Whisper-large-v3-turbo**: 优化版本（默认）

### VAD设置：

- **静音阈值**: 可调节（默认：800ms）
- **片段长度**: 针对语音识别优化

## 🎯 使用技巧

> **重要提示**: 进行批量转录时，务必先尝试单文件转录，找到最佳的静音阈值，确保断句的准确性。

### 使用流程：

1. **单文件测试**: 找到最佳设置
2. **批量处理**: 将设置应用到多个文件
3. **质量检查**: 检查生成的字幕
4. **导出保存**: 保存到指定位置

## 🤝 贡献指南

我们欢迎各种贡献！欢迎：

- 🐛 报告问题
- 💡 提出功能建议
- 🌍 添加翻译
- 🔧 改进代码

## 📄 开源协议

本项目基于 MIT 协议开源 - 详情请查看 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 核心语音识别
- [OpenAI Whisper](https://github.com/openai/whisper) - Whisper模型支持
- [Gradio](https://gradio.app/) - Web界面框架

---

<div align="center">

**❤️ 为语音识别社区精心打造**

[![Star History Chart](https://api.star-history.com/svg?repos=FunAudioLLM/SenseVoice&type=Date)](https://star-history.com/#FunAudioLLM/SenseVoice&Date)

</div>
