# risk_core

`risk_core` は、正本である `scripts/master_v5.py` の横に追加した、読みやすいリスク評価コアです。`master_v5.py` を置き換えるものではなく、主要なノード契約と評価順序を新しい構成で再現し、固定外部パスに依存せずにシミュレーション検証・実データ検証を実行するための土台です。

## 主なクラス

- `FeatureFrame`: 1人物・1時刻分の特徴量です。
- `FeatureFrameSequence`: 1人物の時系列特徴量です。
- `RiskEngine`: 1つの `FeatureFrame` を評価します。
- `BatchRiskEngine`: 複数人物・複数時刻をまとめて評価します。
- `RiskConfig`: モデル種別、重み、部屋サイズ、Fuzzy設定を保持します。

`RiskConfig.model_type` では次を切り替えられます。

- `action_only`: `40000010`-`40000016` の動作リスクのみを使います。
- `action_attribute`: 属性と動作リスクを使います。
- `spatial_context`: 属性、動作、周辺物体、スタッフ見守りを使います。

## ノード番号との対応

既存ノード番号は `schema/node_ids.py` に集約しています。新しいコード内では `10000000` や `40000110` を直接書かず、`TOTAL_RISK` や `STAFF_DISTANCE_RISK` のような定数名を使います。

評価順序は `schema/evaluation_order.py` にあり、`scripts/risk/schema.py` と同じ流れを明示しています。

## 旧実装との関係

現在の互換範囲は次の通りです。

- 動作リスクは `master_v5.py` の姿勢テンプレート類似度を使います。
- `40000010` は `height_max` に有限値がある場合、既存と同じシグモイド式で上書きします。`60010002` 列が存在しても全て欠損値の場合は、姿勢特徴量ベースの立ち上がりリスクを維持します。
- 物体リスクとスタッフリスクは、既存と同じ距離・cos類似度の式を使います。
- Fuzzy集約は、`fuzzy_reasoning_v5.py` の既定ルールと同等の軽量実装を持ちます。
- AHP重みは既定値を持ちますが、`legacy_ahp_adapter.py` と `profile_config.py` から既存CSVを読み込めます。

## AHP / Fuzzy プロファイル

旧 `master_v5.py` の次のような総当たり検証に対応するため、`profile_config.py` を追加しています。

```python
staff_names = ["中村", "百武"]
for staff_name_ahp in staff_names:
    for staff_name_fuzzy in staff_names:
        ...
```

新実装では `make_profile_risk_config()` を使います。

```python
from master_thesis_modules.risk_core.engine.profile_config import make_profile_risk_config

config = make_profile_risk_config(
    ahp_profile_name="中村",
    fuzzy_profile_name="百武",
    common_dir="master_thesis_modules/database/common",
)
```

AHPは `comparison_mtx_30000001_<名前>.csv` と `comparison_mtx_30000010_<名前>.csv` を読みます。Fuzzyは `TFN_<名前>.csv` があれば読みます。TFN CSVが見つからない場合は、旧検証で使った `questionaire_1b.csv` を読み、`S202_Fuzzy推論結果の記録.py` と同じ `5 -> 1.0`, `4 -> 0.75`, `3 -> 0.5`, `2 -> 0.25`, `1 -> 0.0` の変換でルール出力値を作ります。

現在固定している代表値は次の通りです。

- 中村 Fuzzy の総合危険度 `10000000`: `(内部高・外部高=1.0, 内部高・外部低=0.0, 内部低・外部高=0.75, 内部低・外部低=0.25)`
- 百武 Fuzzy の総合危険度 `10000000`: `(内部高・外部高=1.0, 内部高・外部低=0.5, 内部低・外部高=0.25, 内部低・外部低=0.0)`
- 百武 AHP の動作重みは、`40000010=0.148`, `40000013=0.248`, `40000016=0.390` です。

## 通知ログ

`scenario_sim.runner._outputs.save_evaluation_outputs()` は、評価済みCSV、順位、説明文に加えて `notification_log.csv` を保存します。通知判定は `notification.NotificationPolicy` で行い、通知側には AHP/Fuzzy の profile 名や `RiskConfig` は渡しません。profile の差は、すでに `RiskEngine` が出力する `10000000` と factor risk の値に反映されている前提です。

`notification_type=notice` は、各時刻の `RiskResult` だけを使って次の条件で生成します。

- 対象人物の `10000000` が `0.30` 以上である。
- いずれかの動作リスク `40000010`-`40000016` が、前時刻の `0.8` 未満から現時刻の `0.8` 以上へ立ち上がる。
- さらに次のどちらかを満たす。
  - `10000000` が前時刻から `0.15` 以上上昇する。
  - スタッフ見守り系リスク `40000110` または `40000111` が `0.5` 以上で、かつ `10000000` が前時刻から `0.01` 以上上昇する。
- notice 同士は `2.0` 秒以上空ける。
- 1時刻で最大 `3` 件まで通知する。

このため、通知は `10000000` の絶対値だけではなく、「動作が新しく発生したか」と「その動作によって総合危険度がどれだけ上がったか」を見ます。スタッフ見守り系リスクは profile 分岐ではなく、状況特徴量として扱います。

`notification_type=help` は応援要請です。現状では notice とは独立に、`10000000 >= 0.45` の人物数が `max(staff_count, 1)` を超え、前回 help から `2.0` 秒以上経過した場合に固定文を出します。`run_real_data_eval.py` では `--staff-count` でこの人数条件を変更できます。
