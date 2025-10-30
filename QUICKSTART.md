# MurmurNet Quick Start Guide

## クイックスタート（3ステップ）

### 1. セットアップ

```bash
# セットアップスクリプトの実行
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. 設定

`.env`ファイルを編集して、使用するモデルとデバイスを設定：

```bash
# 推奨設定（小規模モデル）
DEFAULT_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
MODEL_DEVICE=cuda  # GPUがない場合はcpu
MODEL_QUANTIZATION=int8  # メモリ節約のため

# 埋め込みモデル
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

### 3. 起動

```bash
# サーバーの起動
./scripts/start.sh

# または Docker で
docker-compose up -d
```

## 基本的な使い方

### APIエンドポイントの確認

```bash
curl http://localhost:8000/health
```

### クエリの送信

```bash
# スクリプトを使用
python scripts/client.py query --text "量子コンピュータとは何ですか？"

# または直接curlで
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "量子コンピュータとは何ですか？"}'
```

### タスク履歴の確認

```bash
# タスクIDを使って履歴を取得
python scripts/client.py history --task-id <TASK_ID> --output history.json

# 可視化
python scripts/visualize_history.py history.json
```

## 評価の実行

```bash
# 全ベンチマークの実行
./scripts/run_evaluation.sh

# 結果の確認
cat evaluation/results/evaluation_report.txt
```

## Wikipedia知識ベースの準備（オプション）

```bash
# 日本語Wikipedia（約90GB）のダウンロード
wget -P data/zim https://download.kiwix.org/zim/wikipedia/wikipedia_ja_all_maxi_latest.zim

# インデックス化（サーバー起動後）
curl -X POST "http://localhost:8000/knowledge/index?max_articles=10000"
```

## トラブルシューティング

### GPUメモリ不足

```bash
# .envで量子化を有効化
MODEL_QUANTIZATION=int8  # または int4
```

### モデルのダウンロードが遅い

モデルは初回起動時に自動ダウンロードされます。`models/`ディレクトリにキャッシュされます。

### ポートが使用中

```bash
# .envでポートを変更
API_PORT=8001
```

## より詳しい情報

詳細は[README.md](README.md)を参照してください。

---

質問や問題がある場合は、GitHubのIssuesで報告してください。
