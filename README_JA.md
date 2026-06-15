# <div align=center>FireRedVAD-ASR-SRT 🎙️🎬</div>

<div align=center>

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align=center>

**高度な音声認識および字幕生成ツール**

</div>

FireRedVADを複数のASRモデルと統合し、単一ファイルまたはバッチでのSRT字幕生成をサポートします。

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## 🌟 主な特徴

- 🎯 **マルチモデル対応**: SenseVoiceSmall、Paraformer、Fun-ASR-Nano、Fun-ASR-MLT-Nano、Granite-Speech、GLM-ASR-Nano
- 🎭 **多言語インターフェース**: 英語、中国語、日本語（拡張が容易）
- 📝 **バッチ処理**: 単一ファイルまたはバッチでの文字起こしに対応
- ⚡ **高性能**: CPUおよびGPUの両方に最適化
- 🎛️ **柔軟な設定**: VADパラメータとモデル設定を調整可能
- 📊 **豊富な出力**: タイムスタンプ付きSRT字幕形式

## 🚀 クイックスタート

### 1. 環境設定

`uv` を使用して仮想環境を作成します（推奨）：

```bash
uv venv --python 3.12
```

### 2. 依存関係のインストール

`uv` を使ってインストールします：

```bash
uv pip install -r requirements.txt
```

### 3. モデル設定

#### モデルのダウンロードと設定：

- **SenseVoiceSmall**: 自動ダウンロード
- **Fun-ASR-Nano**: 自動ダウンロード
- **GLM-ASR-Nano**: 自動ダウンロード
- **Granite-Speech**: 対応言語：英語、フランス語、ドイツ語、スペイン語、ポルトガル語、日本語。ダウンロード：[modelscope](https://www.modelscope.cn/models/ibm-granite/granite-speech-4.1-2b) または [Huggingface](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- **VADモデル**: FireRedVAD フォルダー

### 4. ハードウェアアクセラレーション

#### 🎮 NVIDIA GPU（CUDA）：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126（システムにインストールされているCUDAバージョンに合わせて、適切なバージョンをインストールしてください。）
```

#### 💻 CPUのみ：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. アプリケーションの実行

```bash
uv run main.py
```

## 📁 プロジェクト構造

```
FireRed/
├── 📄 main.py                 # メインプログラム
├── 📄 ctc.py                  # CTC モジュール
├── 📄 model.py                # Fun-ASR モデルコード
├── 📄 requirements.txt        # Python 依存
├── 📁 asr/                    # ASR パッケージソース
├── 📁 FireRedVAD/             # 音声活動検出モデルファイル
├── 📁 FunAudioLLM/            # Fun-ASR ローカルモデルフォルダ
├── 📁 ibm-granite/            # Granite Speech モデルファイル
├── 📁 iic/                    # 追加の音声モデルフォルダ
├── 📁 locales/                # 言語翻訳
├── 📁 tools/                  # ユーティリティスクリプトとツール
├── 📁 utils/                  # ヘルパーモジュール
├── 📁 ZhipuAI/                # 追加ローカルモデルフォルダ
├── 📄 README.md               # このファイル
├── 📄 README_ZH.md            # 中国語版
└── 📄 README_JA.md            # 日本語版
```

## 🎯 使用のヒント

> <span style="color: red;">**重要**</span>: バッチ文字起こしを行う場合は、まず単一ファイルでテストを実行し、最適なVADパラメータを見つけて文章区切りを正確にしてください。文章の区切りを改善するために、VADパラメータ：Min Silence Frames の調整を優先してください。

## 🤝 貢献ガイド

さまざまな貢献を歓迎します！お気軽に：

- 🐛 バグを報告
- 💡 機能を提案
- 🌍 翻訳を追加
- 🔧 コードを改善

## 🙏 謝辞

本プロジェクトは以下のオープンソースプロジェクトに依存しています：

- **[Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)** — ASR モデルパイプライン，Apache 2.0

完全なライセンステキストと著作権表示については [NOTICE.md](NOTICE.md) をご覧ください。

## 📄 ライセンス

このプロジェクトは Apache 2.0 ライセンスの下でオープンソース化されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 📸 インターフェースプレビュー

![メイン画面](main.png)
