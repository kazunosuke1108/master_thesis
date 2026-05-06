# リノベ前後比較レポート

## 目的

このレポートは、既存の修論コード `master_thesis_modules/scripts/master_v5.py` と、リノベ後に追加した `risk_core` / `scenario_sim` / `real_data` の対応関係を、元のコード作者が追いやすい形で整理したものです。

今回のリノベは、`master_v5.py` を置き換える作業ではありません。`master_v5.py` は修論再現用の正本として保持し、その横に、検証や拡張を行いやすい新構成を追加しています。

## リノベ前の構成

旧正本は `master_v5.py` です。主な特徴は次の通りです。

- 人物IDごとの `pandas.DataFrame` を `data_dicts` として受け取る。
- 各DataFrameは `timestamp` と8桁ノード番号列を持つ。
- 第5・6層特徴量から第4層リスクを計算する。
- 第4層リスクから、AHP、Fuzzy推論、単純重み和で第3・2・1層へ集約する。
- 最終的に `10000000` が総合危険度になる。
- 実行・保存・外部ストレージ参照・AHP/Fuzzyプロファイル読込が `Master` クラス周辺にまとまっている。

旧実装の評価順序は、`scripts/risk/schema.py` の `EVALUATION_STEPS` と `Master.evaluate()` が対応しています。

## リノベ後の構成

新構成は責務ごとに分割しています。

| 新ディレクトリ | 責務 |
|---|---|
| `risk_core/schema` | ノードID、ラベル、評価順序、旧schema adapter |
| `risk_core/features` | `FeatureFrame`、時系列、旧DataFrame変換 |
| `risk_core/factors` | 属性、動作、物体、スタッフ関連の第4層リスク |
| `risk_core/aggregators` | 重み和、AHP、Fuzzy、旧CSV adapter |
| `risk_core/engine` | `RiskEngine`、`BatchRiskEngine`、プロファイル設定 |
| `risk_core/explanation` | 第4層リスクからの説明文生成 |
| `risk_core/notification` | 通知・応援要請ログ |
| `scenario_sim` | YAMLシナリオ、イベント、シミュレーションrunner、可視化 |
| `real_data` | `data_dicts.pickle` / CSV入力、実データ評価、旧出力比較 |

## 評価順序の対応

| 旧処理 | 新処理 |
|---|---|
| `fuzzy_logic()` | `risk_core/factors/attribution_risk.py` |
| `pose_similarity()` | `risk_core/factors/action_risk.py` |
| `object_risk()` | `risk_core/factors/object_risk.py` |
| `staff_risk()` | `risk_core/factors/staff_risk.py` |
| `fuzzy_multiply()` | `RiskEngine` 内の `internal_static` 計算 |
| `AHP_weight_sum(..., 30000001)` | `RiskConfig.action_weights` / `legacy_ahp_adapter.py` |
| `AHP_weight_sum(..., 30000010)` | `RiskConfig.object_weights` / `legacy_ahp_adapter.py` |
| `fuzzy_reasoning_master(..., 30000011)` | `LegacyLikeFuzzyAggregator(EXTERNAL_DYNAMIC_RISK)` |
| `simple_weight_sum(..., 20000000, [0.1, 0.9])` | `RiskEngine` 内の `internal` 計算 |
| `fuzzy_reasoning_master(..., 20000001)` | `LegacyLikeFuzzyAggregator(EXTERNAL_RISK)` |
| `fuzzy_reasoning_master(..., 10000000)` | `LegacyLikeFuzzyAggregator(TOTAL_RISK)` |

評価順序自体は `risk_core/schema/evaluation_order.py` に定義しています。

## 第4層リスクの一致点

### 動作リスク

`40000010`-`40000016` は、`master_v5.py` の `risky_motion_dict` と `pose_similarity()` に合わせています。

```text
similarity = 1 - mean(abs(reference_pose - observed_pose))
risk = similarity ** 4
```

`40000010` は、`60010002` に相当する `height_max` がある場合、旧実装と同じ式で上書きします。

```text
risk = 1 / (1 + exp(-5 * (height_max - 1)))
```

### 物体リスク

点滴 `40000100` と車椅子 `40000101` は近いほど高リスクです。手すり `40000102` は遠いほど高リスクです。

正規化係数は旧実装と同じ `sqrt(2) * 6` を既定値にしています。ただし、新実装では `RiskConfig.room_diagonal_m` で変更できます。

### スタッフ関連リスク

スタッフ距離 `40000110` は遠いほど高リスクです。

スタッフ見守り喪失 `40000111` は、患者への相対位置ベクトルとスタッフ速度ベクトルのcos類似度から計算します。スタッフが患者へ向かうほど低リスク、患者から外れる方向へ動くほど高リスクです。

旧実装は速度ゼロなどで `NaN` になり得ます。新実装ではバッチ評価を止めないため、ゼロ速度・欠損時は中立値 `0.5` として扱います。この点は意図的な差分です。

## AHP / Fuzzyプロファイルの対応

旧実装では次のように、AHP担当者とFuzzy担当者の組み合わせを総当たりしていました。

```python
staff_names = ["中村", "百武"]
for staff_name_ahp in staff_names:
    for staff_name_fuzzy in staff_names:
        cls = Master(
            trial_name=trial_name,
            strage="NASK",
            AHP_array_type=staff_name_ahp,
            staff_name_ahp=staff_name_ahp,
            staff_name_fuzzy=staff_name_fuzzy,
        )
        cls.evaluate()
        cls.save_session()
```

新実装では `run_profile_sweep` が対応します。

```bash
python -m master_thesis_modules.scenario_sim.runner.run_profile_sweep \
  --scenario master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml \
  --staff-names 中村 百武 \
  --common-dir master_thesis_modules/database/common \
  --output outputs/thesis_4_5_profile_sweep \
  --visualize
```

出力は次のように分かれます。

```text
outputs/thesis_4_5_profile_sweep/
  ahp_中村__fuzzy_中村/
  ahp_中村__fuzzy_百武/
  ahp_百武__fuzzy_中村/
  ahp_百武__fuzzy_百武/
```

AHPは次のCSVを読みます。

```text
comparison_mtx_30000001_<名前>.csv
comparison_mtx_30000010_<名前>.csv
```

Fuzzyは `TFN_<名前>.csv` が存在すれば読みます。TFN CSVが見つからない場合は、旧検証で使っていた `questionaire_1b.csv` を読み、`S202_Fuzzy推論結果の記録.py` と同じ変換でルール出力値を作ります。

```text
5 -> 1.0
4 -> 0.75
3 -> 0.5
2 -> 0.25
1 -> 0.0
```

この対応により、代表的なルールは次の値になります。

| Fuzzy担当 | 外的リスク `20000001` | 総合危険度 `10000000` |
|---|---|---|
| 中村 | `(高高=0.75, 高低=0.0, 低高=0.25, 低低=0.0)` | `(高高=1.0, 高低=0.0, 低高=0.75, 低低=0.25)` |
| 百武 | `(高高=1.0, 高低=0.75, 低高=0.75, 低低=0.5)` | `(高高=1.0, 高低=0.5, 低高=0.25, 低低=0.0)` |

`questionaire_1b.csv` も見つからない場合のみ、上表と同じ値を最終フォールバックとして使います。旧環境の `TFN_中村.csv` と `TFN_百武.csv` がある場合は、`--common-dir` に置くことでTFNを優先できます。

プロファイル総当たりの横断可視化は、後からでも実行できます。

```bash
python -m master_thesis_modules.scenario_sim.runner.visualize_profile_sweep \
  --input outputs/thesis_4_5_profile_sweep
```

主な出力は、プロファイル別総危険度グリッド、首位患者リスク比較、通知件数比較、要約CSVです。

## シミュレーション検証の対応

追加済みシナリオは次の通りです。

| シナリオ | 目的 |
|---|---|
| `reach_object_context_demo.yaml` | 周辺物体・スタッフ文脈により順位が変わることを確認 |
| `staff_nearby_suppression_demo.yaml` | スタッフ近接により危険度が抑制されることを確認 |
| `priority_reversal_by_context_demo.yaml` | 動作のみと空間文脈込みで順位が変わることを確認 |
| `thesis_4_3_risk_fps_demo.yaml` | 危険度時系列と通知・優先度変化の土台 |
| `thesis_4_4_position_grid_demo.yaml` | 患者・スタッフ位置関係の検証土台 |
| `thesis_4_5_multi_patient_action_demo.yaml` | A/B/Cの複数患者が時間差で危険動作を起こす検証 |

時系列シナリオでは次のイベントを使えます。

- `set_action`
- `move_patient`
- `move_staff`
- `set_pose_features`
- `set_object_position`

レビュー時に、イベント時刻がフレーム刻みと完全一致しない場合に取りこぼす不備を確認しました。現在は、前フレーム時刻から現在フレーム時刻までの間に入ったイベントを次フレームで適用するよう修正済みです。

## 実データ検証の対応

新実装は次の入力を扱えます。

- `data_dicts.pickle`
- `data_<patient>_raw.csv`
- `data_<patient>_eval.csv`

実行例:

```bash
python -m master_thesis_modules.real_data.runner.run_real_data_eval \
  --input /path/to/data_dicts.pickle \
  --output outputs/real_data_eval_new
```

旧出力との比較:

```bash
python -m master_thesis_modules.real_data.runner.compare_real_data_with_legacy \
  --new outputs/real_data_eval_new \
  --legacy /path/to/legacy_eval_csv_dir \
  --output outputs/real_data_compare
```

比較では次を出力します。

- `10000000`, `20000000`, `20000001`, 第4層リスクのMAE
- 最大誤差
- 順位一致率
- 1位一致率

## 出力ファイル

シナリオ・実データ評価では、主に次を出力します。

- `data_<patient>_eval.csv`
- `risk_timeseries.csv`
- `ranking.csv`
- `notification_log.csv`
- `explanations.json`
- `risk_timeseries.png`
- `ranking.png`
- `notification_log.png`
- `visualization/profile_summary.csv`
- `visualization/profile_ranking_summary.csv`
- `visualization/profile_plot_labels.csv`
- `visualization/profile_total_risk_grid.png`
- `visualization/profile_top_risk_comparison.png`
- `visualization/profile_notification_counts.png`

`notification_log.csv` は、音声生成ではなく、通知・応援要請の検証ログです。

## 確認済み項目

今回の確認で実施した内容は次の通りです。

- 新規モジュールの構文確認
- `pytest -q`
- `run_thesis_simulation`
- `compare_models`
- `run_profile_sweep`
- ダミー `data_dicts.pickle` による `run_real_data_eval`
- 新旧比較runnerの出力確認
- `60010002` 列が全て欠損値の場合でも、`40000010` が欠損値にならず姿勢特徴量ベースで計算されること
- `中村` / `百武` のFuzzyプロファイルが `questionaire_1b.csv` 由来の値になること

## 一致している点

- ノードIDの意味と主要な評価順序
- 動作リスクのテンプレートと `similarity ** 4`
- `height_max` による `40000010` 上書き
- 物体リスクの方向
- スタッフ距離・見守り喪失リスクの方向
- 既定Fuzzyルールの形
- AHP比較行列CSVから重みを計算する経路
- `中村` / `百武` のAHP/Fuzzyプロファイル総当たり実行経路

## 意図的に差分が残っている点

- `master_v5.py` は外部固定パス `/media/hayashide/MasterThesis` に依存しますが、新実装はCLI引数でパスを渡します。
- 旧実装の通知は音声ファイルを生成しますが、新実装はCSVログに限定しています。
- スタッフ速度ゼロ時、旧実装は `NaN` になり得ますが、新実装は中立値 `0.5` にします。
- TFN CSVがリポジトリに存在しない場合、新実装は `questionaire_1b.csv` 由来のルール出力値を使います。
- 修論4.5の厳密なイベント時刻は `master_v5.py` 内で完全には明示されていないため、新YAMLでは検証意図を再現するイベント列として記述しています。

## 今後の課題

1. 旧環境の `TFN_中村.csv`, `TFN_百武.csv` をリポジトリ管理または入力データとして整理する。
2. 旧 `master_v5.py` の出力済みCSVと、新実装の比較結果を実データで蓄積する。
3. AHP/Fuzzyプロファイルごとの順位差・通知差を論文図表向けに整形する。
4. 通知条件を旧 `notification_generator_v5.py` にさらに寄せるか、検証用に簡略化した仕様として固定するかを決める。
5. `thesis_4_3`, `thesis_4_4`, `thesis_4_5` のイベント内容を、本文・図表番号と対応する形でコメント補強する。
