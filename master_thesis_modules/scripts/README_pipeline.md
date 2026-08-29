# 修士論文評価パイプライン

## 正本経路

修士論文で使う risk 評価の正本経路は以下です。

1. 実データ前処理、またはシミュレーションデータ生成で入力特徴量を作る。
2. `scripts.master_v5.Master.evaluate()` が、`scripts.risk.schema.EVALUATION_STEPS` に記載した順序で risk ノードを計算する。
3. `scripts.fuzzy.fuzzy_reasoning_v5.FuzzyReasoning` が fuzzy 推論による risk 統合を行う。
4. `scripts.notification.rank_utils.get_risk_rank_by_patient()` が表示・保存用の順位を割り当てる。順位 `0` が最も高 risk の対象を表す。
5. `scripts.visualize.visualizer_v5.Visualizer.plot_matplotlib()` が内部 CSV の risk 値を描画する。平滑化には `w=6` の rolling window を使い、risk 値の上下反転は行わない。

`10000000`, `40000010`, `40000110`, `40000111` などの risk ノードは、すべて「値が大きいほど危険」として扱う。

## 正本ファイル

- `scripts/master_v5.py`: 論文評価の正本エントリポイント。
- `scripts/fuzzy/fuzzy_reasoning_v5.py`: v5 パイプラインの fuzzy 推論実装。
- `scripts/risk/schema.py`: ノード定義と評価順序の記録。
- `scripts/preprocess/preprocess_objects_snapshot.py`: 実データの物体座標特徴量生成。
- `scripts/preprocess/staff_watch.py`: 実データの見守り特徴量生成。
- `scripts/notification/rank_utils.py`: 患者ごとの順位割当て。
- `scripts/visualize/visualizer_v5.py`: 論文用プロット生成。

## 参照用・legacy ファイル

以下は参照用として残す。明示的な目的がない限り、論文評価の正本経路としては使わない。

- `scripts/master_v3.py`
- `scripts/master_v4.py`
- `scripts/master_v6.py`
- `scripts/realtime/debug_realtime.py`
- `scripts_202511/` 以下の古い検証・前処理スクリプト

legacy ファイルは import しているコードが残っている可能性があるため、移動・削除する前に利用箇所を確認する。

## プロット生成

代表的な論文図は以下で再生成できる。

```bash
python master_thesis_modules/scripts/visualize/visualizer_v5.py
```

現状、このスクリプトの `__main__` 付近には複数の trial name が直書きされている。実行前に対象 trial を確認すること。`plot_matplotlib()` は `data_*_eval.csv` を読み込み、可能な列には `rolling(w=6).mean()` を適用し、y 軸を `Risk value` として `result_<node>.pdf` を保存する。

## テスト

risk の向き・順位・入力契約を確認する focused regression suite:

```bash
pytest -q -rx master_thesis_modules/scripts/tests/test_risk_direction_regression.py
```

現在の論文評価パイプライン用テスト:

```bash
pytest -q -rx \
  master_thesis_modules/scripts/tests/test_risk_direction_regression.py \
  master_thesis_modules/scripts/tests/test_simulation_ideal_regression.py \
  master_thesis_modules/scripts/tests/test_real_data_ideal_regression.py
```

全テスト:

```bash
pytest -q
```

## 既知 TODO

- `00021` の T=75 秒以後について、`40000111` と最終 risk 列を再生成した実データ評価を実行する。
- 実データの正本 artifact を決めたら、fixture ベースの実データ理想解テストを read-only の実データ確認テストに置き換える。
- 通知文生成の妥当性確認は、risk・順位の正しさを確認するテストとは分離して扱う。
- 図のデザイン変更は、risk 値の検証とは分離して扱う。
