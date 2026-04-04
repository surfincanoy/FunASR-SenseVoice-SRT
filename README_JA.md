# <div align="center">FireRedVAD-ASR-SRT 🎙️🎬</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align="center">

**高度な音声認識および字幕生成ツール**

</div>

FireRedVADを複数のASRモデルと統合し、単一ファイルまたはバッチでのSRT字幕生成をサポートします。

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## 🌟 主な特徴

- 🎯 **マルチモデル対応**: SenseVoiceSmall、Whisper、Paraformer、Fun-ASR-Nano、Fun-ASR-MLT-Nano
- 🎭 **多言語インターフェース**: 英語、中国語、日本語（簡単に拡張可能）
- 📝 **バッチ処理**: 単一ファイルまたはバッチでの文字起こし機能
- ⚡ **高性能**: CPUとGPUアクセラレーションの両方に最適化
- 🎛️ **柔軟な設定**: VADパラメータとモデル設定を調整可能
- 📊 **豊富な出力**: タイムスタンプ付きSRT字幕フォーマット

## <div align="center">🚀 クイックスタート</div>

### 1. 環境設定

**uv** を使用して仮想環境を作成（推奨）：

```bash
uv venv --python 3.12
```

### 2. 依存関係のインストール

uvを使用してインストール：

```bash
uv pip install -r requirements.txt
# または
uv sync
```

### 3. モデル設定

#### モデルのダウンロードと設定：

- **SenseVoiceSmall**: `disable_update=True` で自動ダウンロード
- **Whisper**: 追加インストールが必要：`uv pip install -U openai-whisper`
- **VADモデル**: FireRedVAD、デフォルトで `.pretrained_models/` に配置

### 4. ハードウェアアクセラレーション

#### 🎮 NVIDIA GPU（CUDA）：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 💻 CPUのみ：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. アプリケーションの実行

```bash
uv run main.py
```

## 🌐 多言語サポート

アプリケーションはシステム言語を自動検出します：

- 🇺🇸 **英語**: 自動的に英語インターフェースに切り替え
- 🇨🇳 **中国語**: 自動的に中国語インターフェースに切り替え
- 🇯🇵 **日本語**: 自動的に日本語インターフェースに切り替え
- ➕ **簡単に拡張**: `locales/` にJSONファイルを追加するだけで新言語をサポート

### 言語オプション

#### 言語を強制指定：

```bash
uv run main.py --lang=en    # 英語インターフェース
uv run main.py --lang=zh    # 中国語インターフェース
uv run main.py --lang=ja    # 日本語インターフェース
```

## 📁 プロジェクト構造

```
FireRed/
├── 📄 main.py                 # メインプログラム
├── 📁 utils/                  # ユーティリティモジュール
│   └── 🌐 translator.py        # 多言語サポート
├── 📁 locales/                # 言語翻訳
│   ├── 🇺🇸 en.json          # 英語
│   ├── 🇨🇳 zh.json          # 中国語
│   └── 🇯🇵 ja.json          # 日本語
├── 📁 tools/                  # Fun-ASRツール
├── 📄 model.py                # Fun-ASRモデルコード
├── 📄 ctc.py                  # CTCモジュール
├── 📄 pyproject.toml          # プロジェクト設定
├── 📄 README.md              # このファイル
├── 📄 README_ZH.md           # 中国語版
└── 📄 README_JA.md           # 日本語版
```

## 🎯 使用のヒント

> **重要**: バッチ文字起こしを行う際は、必ず単一ファイルでの文字起こしを先に行い、最適なVADパラメータを見つけて、正確な文章区切りを確認してください。

### ワークフロー：

1. **単一ファイルテスト**: 最適な設定を見つける
2. **バッチ処理**: 設定を複数のファイルに適用
3. **品質チェック**: 生成された字幕を確認
4. **エクスポート＆保存**: 指定された場所に保存

## 🤝 貢献ガイド

あらゆる種類の貢献を歓迎します！お気軽に：

- 🐛 バグを報告
- 💡 機能を提案
- 🌍 翻訳を追加
- 🔧 コードを改善

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下でオープンソース化されています - 詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 🙏 謝辞

- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) - 音声活動検出
- [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - 音声認識
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 音声認識
- [OpenAI Whisper](https://github.com/openai/whisper) - 音声認識
- [Gradio](https://gradio.app/) - Webインターフェースフレームワーク


---

<div align="center">

**❤️ 音声認識コミュニティのために作られました**

[![Star History Chart](https://api.star-history.com/svg?repos=FunAudioLLM/SenseVoice&type=Date)](https://star-history.com/#FunAudioLLM/SenseVoice&Date)

</div>
