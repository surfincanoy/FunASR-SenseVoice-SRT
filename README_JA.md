# <div align="center">SenseVoice-SRT 🎙️🎬</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/FunAudioLLM/SenseVoice)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/FunAudioLLM/SenseVoice)

</div>

<div align="center">

**高度な音声認識と字幕生成ツール**

</div>

公式SenseVoice WebUIをベースに、単一ファイルまたは一括でのSRT字幕出力をサポートし、選択可能なASRモデルに対応しました。

[中文](README_ZH.md) | [English](README.md) | [日本語](README_JA.md)

---

## <div align="center">🌟 主要機能</div>

- 🎯 **マルチモデル対応**: SenseVoiceSmall、Whisper、Paraformer、Fun-ASR-Nano-2512、Fun-ASR-MLT-Nano-2512
- 🎭 **多言語インターフェース**: 英語、中国語、日本語（簡単拡張）
- 📝 **一括処理**: 単一ファイルまたは一括変換機能
- ⚡ **高性能**: CPUとGPUアクセラレーションを最適化
- 🎛️ **柔軟な設定**: 無音しきい値とモデル設定を調整可能
- 📊 **豊富な出力**: タイムスタンプ付きのSRT字幕フォーマット

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
uv add -r requirements.txt
```

### 3. モデル設定

#### モデルのダウンロードと設定：

- **SenseVoiceSmall**: `disable_update=False` を設定して自動ダウンロード
- **Whisperモデル**: 追加インストールが必要: `uv pip install -U openai-whisper`
- **VADモデル**: `fsmn_vad` パスを設定

ダウンロード完了後、`disable_update=True` を設定して起動時間を短縮。

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

## <div align="center">🌐 多言語サポート</div>

アプリケーションはシステム言語を自動検出：

- 🇺🇸 **英語**: 自動的に英語インターフェースに切り替え
- 🇨🇳 **中国語**: 自動的に中国語インターフェースに切り替え
- 🇯🇵 **日本語**: 自動的に日本語インターフェースに切り替え
- ➕ **拡張可能**: `locales/` にJSONファイルを追加するだけで新言語をサポート

### 言語オプション

#### 特定言語を指定：

```bash
uv run main.py --lang=en    # 英語インターフェース
uv run main.py --lang=zh    # 中国語インターフェース
uv run main.py --lang=ja    # 日本語インターフェース
```

## <div align="center">📁 プロジェクト構造</div>

```
SenseVoice-SRT/
├── 📄 main.py                 # メインプログラム
├── 📁 utils/                  # ユーティリティモジュール
│   └── 🌐 translator.py        # 多言語サポート
├── 📁 locales/                # 言語翻訳
│   ├── 🇺🇸 en.json          # 英語
│   ├── 🇨🇳 zh.json          # 中国語
│   └── 🇯🇵 ja.json          # 日本語
├── 📄 requirements.txt         # 依存関係リスト
├── 📄 README.md              # 英語版
├── 📄 README_ZH.md           # 中国語版
└── 📄 README_JA.md           # 日本語版（このファイル）
```

## <div align="center">⚙️ 設定説明</div>

### 対応モデル：

- **SenseVoiceSmall**: 高速、正確な多言語ASR
- **Whisper-large-v3-turbo**: 最適化バージョン（デフォルト）

### VAD設定：

- **無音しきい値**: 調整可能（デフォルト：800ms）
- **セグメント長**: 音声認識向けに最適化

## <div align="center">🎯 使用のコツ</div>

> **重要提示**: 一括変換を行う際は、まず単一ファイルでテストし、最適な無音しきい値を見つけて、正確な文の分割を確実にしてください。

### ワークフロー：

1. **単一ファイルテスト**: 最適な設定を検索
2. **一括処理**: 複数ファイルに設定を適用
3. **品質チェック**: 生成された字幕を確認
4. **エクスポート**: 指定場所に保存

## <div align="center">🤝 コントリビューション</div>

様々なコントリビューションを歓迎します！：

- 🐛 問題の報告
- 💡 機能の提案
- 🌍 翻訳の追加
- 🔧 コードの改善

## <div align="center">📄 オープンソースライセンス</div>

本プロジェクトはMITライセンスに基づきオープンソースです - 詳細は [LICENSE](LICENSE) ファイルを確認してください。

## <div align="center">🙏 謝辞</div>

- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - コア音声認識
- [OpenAI Whisper](https://github.com/openai/whisper) - Whisperモデルサポート
- [Gradio](https://gradio.app/) - Webインターフェースフレームワーク

---

<div align="center">

**❤️ 音声認識コミュニティのために作成**

[![Star History Chart](https://api.star-history.com/svg?repos=FunAudioLLM/SenseVoice&type=Date)](https://star-history.com/#FunAudioLLM/SenseVoice&Date)

</div>
