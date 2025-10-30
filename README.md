# MurmurNet: 小規模言語モデル群による協調的創発知能システム

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**MurmurNet**は、複数の小規模言語モデル（SLM）が協調して動作することで、単一のSLMや場合によっては大規模言語モデル（LLM）にも見られないような「創発的な能力」が観測されるかを検証する研究実験システムです。

## 🎯 プロジェクト概要

### 背景

ミンスキーの「心の社会」理論に着想を得て、本システムは以下の仮説を検証します：

1. 異なる役割を持つ複数のSLMエージェントがブラックボードシステムを介して協調することで、単一のSLMでは解決できない複雑なタスクを遂行できる
2. この協調プロセスを通じて、スケーリング則だけでは説明できない非線形な能力向上が観測される

### システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (API)                      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│               Blackboard System (中央制御)                   │
│          (エージェント間の共有ワークスペース)                  │
└─────┬──────────────────────┬──────────────────────┬─────────┘
      │                      │                      │
┌─────▼──────┐    ┌─────────▼────────┐    ┌───────▼─────────┐
│ Knowledge  │    │  Memory System   │    │  Agent Swarm   │
│ Base (RAG) │    │(Long/Short Term) │    │ (6 Specialized)│
│ Wikipedia  │    │   Experience     │    │     Agents)    │
└────────────┘    └──────────────────┘    └────────────────┘
```

### 主要コンポーネント

#### 1. ブラックボードシステム
- 全エージェントが情報を共有・読み書きする中央データストア
- タスクの現在状態を保持するワーキングメモリ
- タイムスタンプ付きエントリー管理

#### 2. エージェント群（6種類）

| エージェント | 役割 | 出力 |
|------------|------|-----|
| **入力分析官** | ユーザー入力の解析、キーワード抽出 | タスク要約、キーワード |
| **計画立案官** | ステップバイステップの計画立案 | 実行計画 |
| **知識検索官** | RAGシステムとメモリから関連知識を検索 | 検索結果 |
| **回答形式指定官** | 最適な回答形式を定義 | 形式仕様 |
| **統合官** | 全情報を統合し最終回答を生成 | 最終回答 |
| **監督官** | ブラックボード全体を監視・調整 | 指示 |

#### 3. 知識ベース（RAG）
- WikipediaのZIMファイルからオフライン知識検索
- FAISS/ChromaDBによるベクトル検索
- `multilingual-e5-large`による埋め込み

#### 4. メモリシステム
- **長期記憶**: 過去の対話の重要な結論・事実を保存
- **経験メモリ**: 成功/失敗したタスク遂行プロセスを記録

## 🚀 セットアップ

### 必要要件

- Python 3.10以上
- CUDA対応GPU（推奨、CPUでも動作可能）
- 8GB以上のRAM
- Docker & Docker Compose（オプション）

### インストール

#### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/MurmurNet.git
cd MurmurNet
```

#### 2. 環境変数の設定

```bash
cp .env.example .env
# .envファイルを編集して設定をカスタマイズ
```

#### 3. Dockerを使用する場合（推奨）

```bash
# イメージのビルド
docker-compose build

# サービスの起動
docker-compose up -d

# ログの確認
docker-compose logs -f
```

#### 4. ローカル環境で実行する場合

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# サーバーの起動
python -m src.main
```

### Wikipedia ZIMファイルの準備（オプション）

RAG機能を使用する場合は、WikipediaのZIMファイルをダウンロードしてください：

```bash
# 日本語Wikipediaの例
mkdir -p data/zim
cd data/zim
wget https://download.kiwix.org/zim/wikipedia/wikipedia_ja_all_maxi_latest.zim

# .envファイルでパスを設定
# ZIM_FILE_PATH=./data/zim/wikipedia_ja_all_maxi_latest.zim
```

ZIMファイルをインデックス化：

```bash
# APIエンドポイント経由
curl -X POST http://localhost:8000/knowledge/index?max_articles=1000
```

## 📖 使用方法

### API経由でクエリを送信

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "量子コンピュータの基本原理を説明してください"}'
```

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "量子コンピュータの基本原理を説明してください"}
)

result = response.json()
print(result["answer"])
```

### タスク履歴の取得

```bash
curl http://localhost:8000/task/{task_id}/history
```

### システム統計の確認

```bash
curl http://localhost:8000/statistics
```

## 🧪 評価

### ベンチマークの実行

システムは以下の3つのベンチマークで評価されます：

1. **GSM8K**: 算数文章題（多段階推論）
2. **MT-Bench**: マルチターン対話能力
3. **TruthfulQA**: 幻覚抑制・事実性

```bash
# 評価スクリプトの実行
python evaluation/run_evaluation.py
```

評価結果は`evaluation/results/`ディレクトリに保存されます。

### カスタム評価

```python
from src.orchestrator import Orchestrator
from evaluation.gsm8k_benchmark import GSM8KBenchmark

orchestrator = Orchestrator()
benchmark = GSM8KBenchmark(orchestrator, output_dir="./results")
results = await benchmark.run_evaluation(max_questions=100)
```

## 📊 ログと可視化

### ログファイル

- `logs/murmurnet_YYYY-MM-DD.log`: 全体ログ
- `logs/murmurnet_errors_YYYY-MM-DD.log`: エラーログ

### ブラックボード履歴の可視化

```python
from src.orchestrator import Orchestrator

orchestrator = Orchestrator()
result = await orchestrator.process_query("質問内容")

# タスク履歴の取得
history = await orchestrator.get_task_history(result["task_id"])

# 詳細な分析
import json
print(json.dumps(history, indent=2, ensure_ascii=False))
```

## ⚙️ 設定

主要な設定は`.env`ファイルまたは環境変数で行います：

### モデル設定

```bash
DEFAULT_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
MODEL_DEVICE=cuda
MODEL_QUANTIZATION=none  # none, int8, int4
```

### エージェント設定

```bash
MAX_AGENT_RETRIES=3
AGENT_TIMEOUT_SECONDS=60
MAX_PARALLEL_AGENTS=3
```

### ベクトルDB設定

```bash
VECTOR_DB_TYPE=faiss  # faiss or chroma
VECTOR_DB_TOP_K=5
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

## 🔬 研究のポイント

### 創発的能力の観測方法

1. **定量的評価**: ベンチマークスコアの比較
   - 単一SLM vs MurmurNetシステム
   - 小規模構成 vs 大規模LLM

2. **定性的評価**: ブラックボードログの分析
   - エージェント間の情報伝播パターン
   - 計画の進化過程
   - 知識の統合プロセス

3. **成功事例の抽出**
   - 単一SLMが失敗しMurmurNetが成功したケースを特定
   - エージェント協調がどのように問題解決に寄与したかを分析

### データ収集

```python
# 詳細なタスク履歴をエクスポート
history = await orchestrator.get_task_history(task_id)

# 各エージェントの寄与を分析
for entry in history["entries"]:
    print(f"{entry['agent_id']}: {entry['entry_type']}")
```

## 📝 開発ガイドライン

### コード品質

- すべての設定値は`.env`または`src/config/`で管理
- マジックナンバーは使用しない
- 明晰で読みやすいコードを心がける
- ダミー実装やモックは使用しない

### 新しいエージェントの追加

1. `src/config/agent_config.py`に定義を追加
2. `src/agents/specialized_agents.py`に実装を追加
3. `src/orchestrator/orchestrator.py`で初期化

### 新しいベンチマークの追加

1. `evaluation/base_benchmark.py`を継承
2. `load_dataset()`と`evaluate_answer()`を実装
3. `evaluation/run_evaluation.py`に追加

## 🤝 貢献

本プロジェクトは研究目的で開発されています。バグ報告や改善提案は歓迎します。

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 📚 参考文献

1. Minsky, M. (1986). "The Society of Mind"
2. Scaling Laws for Neural Language Models (Kaplan et al., 2020)
3. Retrieval-Augmented Generation (Lewis et al., 2020)

## 🔗 関連リンク

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**MurmurNet** - Small Models, Big Intelligence Through Collaboration
