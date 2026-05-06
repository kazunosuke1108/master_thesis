# scenario_sim

`scenario_sim` は、意味名で書いたYAMLシナリオを `risk_core` に流して評価するためのモジュールです。シナリオYAMLでは `40000010` のようなノード番号を直接書かず、`action_label`, `iv_pole`, `wheelchair`, `handrail`, `staff` などの意味名を使います。

## 基本実行

修論4.5相当の時系列シミュレーションを実行します。

```bash
python -m master_thesis_modules.scenario_sim.runner.run_thesis_simulation \
  --scenario master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml \
  --model spatial_context \
  --output outputs/thesis_4_5_new
```

## 比較モデル

動作のみ、属性+動作、空間的文脈込みの3モデルを比較します。

```bash
python -m master_thesis_modules.scenario_sim.runner.compare_models \
  --scenario master_thesis_modules/scenario_sim/scenarios/reach_object_context_demo.yaml \
  --models action_only action_attribute spatial_context \
  --output outputs/reach_context_comparison
```

## AHP / Fuzzy プロファイル総当たり

旧 `master_v5.py` の `staff_names = ["中村", "百武"]` ループに相当する実行は、`run_profile_sweep` で行います。

```bash
python -m master_thesis_modules.scenario_sim.runner.run_profile_sweep \
  --scenario master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml \
  --staff-names 中村 百武 \
  --common-dir master_thesis_modules/database/common \
  --output outputs/thesis_4_5_profile_sweep \
  --visualize
```

出力先には次のようなサブディレクトリが作られます。

```text
outputs/thesis_4_5_profile_sweep/
  ahp_中村__fuzzy_中村/
  ahp_中村__fuzzy_百武/
  ahp_百武__fuzzy_中村/
  ahp_百武__fuzzy_百武/
```

各ディレクトリには、評価済みCSV、順位、通知ログ、説明文が保存されます。

既に計算済みの `run_profile_sweep` 出力を後から可視化する場合は、次を実行します。

```bash
python -m master_thesis_modules.scenario_sim.runner.visualize_profile_sweep \
  --input outputs/thesis_4_5_profile_sweep
```

このコマンドは、既定では `outputs/thesis_4_5_profile_sweep/visualization/` に、プロファイル横断の図と要約CSVを保存します。

## 生成されるファイル

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

## 対応イベント

YAMLの `events` では次のイベントを使えます。

- `set_action`: 患者の動作ラベルを変更します。
- `move_patient`: 患者位置を変更します。
- `move_staff`: スタッフ位置と速度を変更します。
- `set_pose_features`: 4次元姿勢特徴量を直接変更します。
- `set_object_position`: 物体位置を変更します。
