# <div align="center">FireRedVAD-ASR-SRT 🎙️🎬</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align="center">

**Advanced Speech Recognition and Subtitle Generation Tool**

</div>

Integrated FireRedVAD with multiple ASR models, supporting single-file or batch SRT subtitle generation.

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## 🌟 Core Features

- 🎯 **Multi-model Support**: SenseVoiceSmall, Paraformer, Fun-ASR-Nano, Fun-ASR-MLT-Nano, Granite-Speech, GLM-ASR-Nano
- 🎭 **Multi-language Interface**: English, Chinese, Japanese (easily extensible)
- 📝 **Batch Processing**: Single-file or batch transcription
- ⚡ **High Performance**: Optimized for both CPU and GPU acceleration
- 🎛️ **Flexible Configuration**: Adjustable VAD parameters and model settings
- 📊 **Rich Output**: SRT subtitle format with timestamps

## <div align="center">🚀 Quick Start</div>

### 1. Environment Setup

Use **uv** to create a virtual environment (recommended):

```bash
uv venv --python 3.12
```

### 2. Install Dependencies

Install using uv:

```bash
uv pip install -r requirements.txt
```

### 3. Model Configuration

#### Download and configure models:

- **SenseVoiceSmall**: Automatically downloads
- **FunAsr-nano**: Automatically downloads
- **GLM-Asr-nano**: Automatically downloads
- **Granite-Speech**: Supported languages: English, French, German, Spanish, Portuguese and Japanese. Download from [modelscope](https://www.modelscope.cn/models/ibm-granite/granite-speech-4.1-2b) or [Huggingface](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- **VAD Model**: FireRedVAD folder
> <span style="color: red;">**Important**</span>:  All models can be taken from the cache folder and placed into the project folder.  


### 4. Hardware Acceleration

#### 🎮 NVIDIA GPU (CUDA):

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 💻 CPU Only:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Run the Application

```bash
uv run main.py
```

## 📁 Project Structure

```
FireRed/
├── 📄 main.py                 # Main program
├── 📄 ctc.py                  # CTC module
├── 📄 model.py                # Fun-ASR model code
├── 📄 requirements.txt        # Python dependencies
├── 📁 asr/                    # ASR package sources
├── 📁 FireRedVAD/             # Voice activity detection model files
├── 📁 FunAudioLLM/            # Fun-ASR local model folders
├── 📁 ibm-granite/            # Granite Speech model files
├── 📁 iic/                    # Additional speech model folders
├── 📁 locales/                # Language translations
├── 📁 tools/                  # Utility scripts and tools
├── 📁 utils/                  # Translation utilities
├── 📁 ZhipuAI/                # Additional local model folders
├── 📄 README.md               # This file
├── 📄 README_ZH.md            # Chinese version
└── 📄 README_JA.md            # Japanese version
```

## 🎯 Usage Tips

> <span style="color: red;"> **Important**</span>: When performing batch transcription, always try single-file transcription first to find the optimal VAD parameters and ensure accurate sentence segmentation.  Please prioritize adjusting the VAD parameter Min Silence Frames to get better sentence segmentation.

## 🤝 Contributing

We welcome all kinds of contributions! Feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 🌍 Add translations
- 🔧 Improve code

## 🙏 Acknowledgements

This project relies on the following open-source projects:

- **[Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)** — ASR model pipeline, Apache 2.0

See [NOTICE.md](NOTICE.md) for full license texts and copyright notices.

## 📄 License

This project is open-sourced under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📸 Interface Preview

![Main Interface](main.png)
