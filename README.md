# <div align="center">SenseVoice-SRT 🎙️🎬</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align="center">

**Advanced Speech Recognition & Subtitle Generation Tool**

</div>

Based on the official SenseVoice WebUI, enhanced with support for single-file or batch SRT subtitle output with selectable ASR models.

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## 🌟 Features

- 🎯 **Multi-Model Support**: SenseVoiceSmall, Whisper, Paraformer
- 🎭 **Multi-Language Interface**: English, Chinese, Japanese (easily extensible)
- 📝 **Batch Processing**: Single file or batch transcription capabilities
- ⚡ **High Performance**: Optimized for both CPU and GPU acceleration
- 🎛️ **Flexible Configuration**: Adjustable silence threshold and model settings
- 📊 **Rich Output**: SRT subtitle format with timestamps

## <div align="center">🚀 Quick Start</div>

### 1. Environment Setup

Create a virtual environment using **uv** (recommended):
```bash
uv venv --python 3.12
```

### 2. Install Dependencies

Install using uv:
```bash
uv pip install -r requirements.txt
# or
uv add -r requirements.txt
```

### 3. Model Configuration

#### Download and Configure Models:
- **SenseVoiceSmall**: Set `disable_update=False` to auto-download
- **Whisper Models**: Additional installation required: `uv pip install -U openai-whisper`
- **VAD Model**: Configure `fsmn_vad` path

After downloading, set `disable_update=True` to reduce startup time.

### 4. Hardware Acceleration

#### 🎮 NVIDIA GPU (CUDA):
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 💻 CPU-Only:
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Run Application

```bash
uv run main.py
```

## 🌐 Multi-Language Support

The application automatically detects your system language:

- 🇺🇸 **English**: Automatically switches to English interface
- 🇨🇳 **Chinese**: Automatically switches to Chinese interface  
- 🇯🇵 **Japanese**: Automatically switches to Japanese interface
- ➕ **Extensible**: Add new languages by simply adding JSON files to `locales/`

### Language Options

#### Force Specific Language:
```bash
uv run main.py --lang=en    # English interface
uv run main.py --lang=zh    # Chinese interface
uv run main.py --lang=ja    # Japanese interface
```

## 📁 Project Structure

```
SenseVoice-SRT/
├── 📄 main.py                 # Main application
├── 📁 utils/                  # Utility modules
│   └── 🌐 translator.py        # Multi-language support
├── 📁 locales/                # Language translations
│   ├── 🇺🇸 en.json          # English
│   ├── 🇨🇳 zh.json          # Chinese
│   └── 🇯🇵 ja.json          # Japanese
├── 📄 requirements.txt         # Dependencies
├── 📄 README.md              # This file
└── 📄 README_ZH.md           # Chinese version
```

## ⚙️ Configuration

### Supported Models:
- **SenseVoiceSmall**: Fast, accurate, multi-language ASR
- **Whisper-large-v3-turbo**: Optimized version (default)

### VAD Settings:
- **Silence Threshold**: Adjustable (default: 800ms)
- **Segment Length**: Optimized for speech recognition

## 🎯 Usage Tips

> **Important**: When performing batch transcription, always test with a single file first to find the optimal silence threshold and ensure accurate sentence segmentation.

### Workflow:
1. **Single File Test**: Find optimal settings
2. **Batch Processing**: Apply settings to multiple files
3. **Quality Check**: Review generated subtitles
4. **Export**: Save to desired location

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report issues
- 💡 Suggest features  
- 🌍 Add translations
- 🔧 Improve code

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - Core speech recognition
- [OpenAI Whisper](https://github.com/openai/whisper) - Whisper model support
- [Gradio](https://gradio.app/) - Web interface framework

---

<div align="center">

**Made with ❤️ for the speech recognition community**

[![Star History Chart](https://api.star-history.com/svg?repos=FunAudioLLM/SenseVoice&type=Date)](https://star-history.com/#FunAudioLLM/SenseVoice&Date)

</div>