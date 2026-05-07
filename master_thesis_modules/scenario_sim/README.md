# scenario_sim

`scenario_sim` は、意味名で書いたYAMLシナリオを `risk_core` に流して評価するためのモジュールです。シナリオYAMLでは `40000010` のようなノード番号を直接書かず、`action_label`, `iv_pole`, `wheelchair`, `handrail`, `staff` などの意味名を使います。

## シナリオYAML仕様

生成AIにシミュレーションシナリオを作らせる場合は、この節のフォーマットに従った YAML ファイルを出力させてください。ファイルは `master_thesis_modules/scenario_sim/scenarios/<scenario_name>.yaml` に置く想定です。

### 全体構造

```yaml
scenario_name: example_scenario
time_range:
  duration_s: 10.0
  step_s: 1.0
patients:
  - id: A
    is_patient_label: yes
    age_group_label: old
    position: {x: 2.0, y: 5.0}
    action_label: neutral_sitting
objects:
  - id: wheelchair_a
    type: wheelchair
    position: {x: 2.0, y: 5.0}
staff:
  - id: NS
    position: {x: 5.0, y: 5.0}
    velocity: {vx: 0.0, vy: 0.0}
events:
  - time_s: 2.0
    event_type: set_action
    target_id: A
    action_label: standing_up
```

### ルート項目

| 項目 | 必須 | 型 | 説明 |
|---|---:|---|---|
| `scenario_name` | 推奨 | 文字列 | シナリオ名。生成AIには英数字と `_` の名前を推奨します。`thesis_4_5_multi_patient_action_demo` は `run_profile_sweep` で特別扱いされるため、新規シナリオでは使わないでください。 |
| `time_range.duration_s` | 必須 | 数値 | シミュレーション長さ。秒単位。 |
| `time_range.step_s` | 必須 | 数値 | 評価間隔。秒単位。`1.0` なら 0, 1, 2, ... 秒で評価します。 |
| `patients` | 必須 | 配列 | 評価対象者の初期状態。1人以上を推奨します。 |
| `objects` | 任意 | 配列 | 周辺物体の初期状態。無い種類の物体はその種類のリスクが 0 になります。 |
| `staff` | 任意 | 配列 | スタッフの初期状態。無い場合はスタッフ距離リスクが最大になります。 |
| `events` | 任意 | 配列 | 時刻に応じた状態変更。空配列でも実行できます。 |

### 患者 `patients`

```yaml
patients:
  - id: A
    is_patient_label: yes
    age_group_label: old
    position: {x: 2.0, y: 5.0}
    action_label: neutral_sitting
    height_max: 0.8
```

| 項目 | 必須 | 値 |
|---|---:|---|
| `id` | 必須 | 患者ID。`A`, `B`, `C` など。イベントの `target_id` から参照します。 |
| `is_patient_label` | 任意 | `yes` または `no`。省略時は `yes`。 |
| `age_group_label` | 任意 | `young`, `middle`, `old`。省略時は `middle`。 |
| `position` | 必須 | `{x: 数値, y: 数値}`。2次元平面上の位置。 |
| `action_label` | 任意 | 下記の動作ラベル。省略時は `neutral_sitting` 相当。 |
| `height_max` | 任意 | 高さ最大値。指定した場合、立ち上がりリスク `40000010` は高さベースのシグモイド値になります。姿勢テンプレートで評価したい場合は省略してください。 |
| `pose_features` | 任意 | 4次元姿勢特徴量を直接指定します。指定時は `action_label` より優先されます。 |

`pose_features` を使う場合は次の形式です。

```yaml
pose_features:
  standing_degree: 0.0
  trunk_tilt: 0.0
  wrist_distance_from_hip: 0.0
  ankle_spread: 0.2
```

### 動作ラベル `action_label`

使える `action_label` は次の通りです。

| ラベル | 意味 |
|---|---|
| `neutral_sitting` | 通常の座位 |
| `standing_up` | 立ち上がり |
| `release_brake` | 車椅子ブレーキ解除 |
| `move_wheelchair` | 車椅子を動かす |
| `lose_balance` | バランスを崩す |
| `raise_hands` | 手を上げる/動かす |
| `cough_up` | せき込む |
| `touch_face` | 顔を触る |
| `reach_floor` | 床方向へ手を伸ばす |
| `reach_table` | テーブル方向へ手を伸ばす |
| `reach_iv_pole` | 点滴方向へ手を伸ばす |

### 物体 `objects`

```yaml
objects:
  - id: iv_a
    type: iv_pole
    position: {x: 2.2, y: 5.0}
  - id: wheelchair_a
    type: wheelchair
    position: {x: 2.0, y: 5.0}
  - id: handrail_a
    type: handrail
    position: {x: 0.0, y: 5.0}
```

| 項目 | 必須 | 値 |
|---|---:|---|
| `id` | 必須 | 物体ID。イベントの `target_id` から参照します。 |
| `type` | 必須 | `iv_pole`, `wheelchair`, `handrail` のいずれか。 |
| `position` | 必須 | `{x: 数値, y: 数値}`。 |

各患者に対して、種類ごとの最近傍物体だけが特徴量として使われます。たとえば `wheelchair` が複数ある場合、各患者に最も近い車椅子が `40000101` の計算に使われます。

### スタッフ `staff`

```yaml
staff:
  - id: NS
    position: {x: 5.0, y: 5.0}
    velocity: {vx: 0.0, vy: 0.0}
```

| 項目 | 必須 | 値 |
|---|---:|---|
| `id` | 必須 | スタッフID。イベントの `target_id` から参照します。 |
| `position` | 必須 | `{x: 数値, y: 数値}`。 |
| `velocity` | 任意 | `{vx: 数値, vy: 数値}`。省略時は `{vx: 0.0, vy: 0.0}`。 |

スタッフ距離リスク `40000110` は、患者から最も近いスタッフの距離で計算されます。スタッフ見守りリスク `40000111` は、患者への相対位置ベクトルとスタッフ速度ベクトルから計算されます。速度がゼロの場合は方向が定義できないため、中立値 `0.5` として扱われます。

### イベント `events`

イベントは `time_s` の時刻で状態を変更します。状態変更は次のイベントが来るまで持続します。通常の `scenario_sim` ではイベント間の位置補間は行わず、指定時刻で状態が切り替わります。連続移動を表したい場合は、複数の `move_patient` / `move_staff` イベントを細かく並べてください。

#### `set_action`

患者の動作ラベルを変更します。

```yaml
- time_s: 2.0
  event_type: set_action
  target_id: A
  action_label: standing_up
```

#### `set_pose_features`

患者の4次元姿勢特徴量を直接変更します。

```yaml
- time_s: 3.0
  event_type: set_pose_features
  target_id: A
  standing_degree: 0.8
  trunk_tilt: 0.2
  wrist_distance_from_hip: 0.1
  ankle_spread: 0.7
```

#### `move_patient`

患者位置を変更します。

```yaml
- time_s: 4.0
  event_type: move_patient
  target_id: A
  position: {x: 2.5, y: 4.5}
```

#### `move_staff`

スタッフ位置と速度を変更します。

```yaml
- time_s: 5.0
  event_type: move_staff
  target_id: NS
  position: {x: 2.0, y: 3.0}
  velocity: {vx: -0.5, vy: -0.5}
```

#### `set_object_position`

物体位置を変更します。

```yaml
- time_s: 6.0
  event_type: set_object_position
  target_id: wheelchair_a
  position: {x: 2.1, y: 4.8}
```

### 生成AIへの作成ルール

別のAIにシナリオを作らせる場合は、次の条件を指示してください。

- YAMLだけを出力し、Markdownの説明文を混ぜない。
- `scenario_name` は英数字と `_` のみで作る。
- `patients` は最低1人、可能なら `A`, `B`, `C` のように短いIDにする。
- `position`, `velocity`, `time_s`, `duration_s`, `step_s` はすべて数値にする。
- `events[*].target_id` は、必ず `patients`, `staff`, `objects` のいずれかに存在する `id` を参照する。
- `event_type` と `action_label` はこのREADMEにある許可値だけを使う。
- `time_s` は `0.0 <= time_s <= duration_s` に収める。
- `events` は時刻順に並べる。
- 連続移動を表す場合は複数イベントで近似する。
- `height_max` は高さベースの立ち上がり評価を明示したい場合だけ使う。通常は省略する。

### 生成AI向け最小テンプレート

```yaml
scenario_name: generated_example
time_range:
  duration_s: 10.0
  step_s: 1.0
patients:
  - id: A
    is_patient_label: yes
    age_group_label: old
    position: {x: 2.0, y: 5.0}
    action_label: neutral_sitting
  - id: B
    is_patient_label: yes
    age_group_label: middle
    position: {x: 4.0, y: 2.0}
    action_label: neutral_sitting
objects:
  - id: wheelchair_a
    type: wheelchair
    position: {x: 2.1, y: 5.0}
  - id: iv_b
    type: iv_pole
    position: {x: 4.2, y: 2.0}
  - id: handrail_wall
    type: handrail
    position: {x: 0.0, y: 3.0}
staff:
  - id: NS
    position: {x: 5.0, y: 5.0}
    velocity: {vx: 0.0, vy: 0.0}
events:
  - time_s: 2.0
    event_type: set_action
    target_id: A
    action_label: standing_up
  - time_s: 3.0
    event_type: move_staff
    target_id: NS
    position: {x: 3.0, y: 4.0}
    velocity: {vx: -1.0, vy: -0.5}
  - time_s: 5.0
    event_type: set_action
    target_id: B
    action_label: reach_iv_pole
  - time_s: 7.0
    event_type: set_action
    target_id: A
    action_label: neutral_sitting
```

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

旧 `master_v5.py` の `staff_names = ["中村", "百武"]` ループに相当する実行は、`run_profile_sweep` で行います。`thesis_4_5_multi_patient_action_demo` は旧 `master_v5.py` の既定シミュレーション入力（20Hz、姿勢・スタッフ軌道の補間、物体配置）を互換生成します。一方で危険度計算は現行 `risk_core` を使うため、立ち上がりリスクやスタッフ速度ゼロ時の扱いは旧 `master_v5.py` と意図的に異なる場合があります。

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

旧出力との差分確認は次のように行えます。差分がある場合、このコマンドは差分列を出力して非ゼロ終了します。

```bash
python -m master_thesis_modules.scenario_sim.runner.compare_profile_sweep_with_legacy \
  --legacy-dir /media/hayashide/MasterThesis/20260506_Simで動作確認_中村 \
  --renovated-dir outputs/thesis_4_5_profile_sweep/ahp_中村__fuzzy_中村 \
  --output-csv outputs/thesis_4_5_profile_sweep/compare_中村.csv
```

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
