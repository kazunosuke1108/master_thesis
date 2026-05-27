# master_thesis

修士論文用のリスク評価コード一式です。人物ごとの時系列特徴量を入力し、内的要因・外的要因・周辺物体・スタッフの見守り状態を階層的に統合して、最終的な総合危険度 `10000000` を算出します。

現状の正本は `master_thesis_modules/scripts/master_v5.py` です。`master_v2.py`, `master_v3.py`, `master_v4.py`, `master_v6.py` などは過去検証・派生実装として残っていますが、論文評価の説明では `v5` 系を中心に見るのが安全です。

## 目的

このリポジトリは、病院・介護施設のような空間で、複数人物のうち「今もっとも注意すべき対象者」を推定するための実験コードです。

主な入力は、人物の属性、姿勢、位置、周辺物体、スタッフ位置・移動方向です。これらをノード番号で管理し、次の階層で危険度へ変換します。

1. 入力特徴量を作る。
2. 特徴量から第4層の要因別リスクを計算する。
3. AHP、Fuzzy 推論、単純重み和で上位リスクへ統合する。
4. 患者ごとの総合危険度 `10000000` を降順に並べ、通知・可視化へ渡す。

## ディレクトリ構成

```text
master_thesis_modules/
  scripts/
    master_v5.py                         # 現行の論文評価エントリポイント
    README_pipeline.md                   # 正本パイプラインの短い運用メモ
    risk/schema.py                       # 主要ノード定義と評価順序
    network/graph_manager_v3.py          # 評価グラフとノード接続
    fuzzy/fuzzy_reasoning_v5.py          # Fuzzy 推論
    AHP/get_comparison_mtx_v3.py         # AHP 一対比較行列と重み
    preprocess/                         # 実データ用の特徴量生成
    pseudo_data/                        # シミュレーションデータ生成
    notification/                       # リスク順位と通知文生成
    visualize/                          # グラフ・動画・論文用プロット
    tests/                              # リスク方向や順位の回帰テスト
  scripts_202511/                       # 2025年11月頃の検証・集計スクリプト
docker/                                 # Docker 実行補助
```

## 正本パイプライン

`scripts.master_v5.Master` は、入力済みの `data_dicts` を受け取り、人物 ID ごとの `pandas.DataFrame` に計算済みリスク列を追加していきます。各 DataFrame は `timestamp` とノード番号列を持ちます。

正本の評価順序は `scripts/risk/schema.py` の `EVALUATION_STEPS` に記録されています。実際の処理は `master_v5.py` の `evaluate()` が呼び出します。

| 段階 | メソッド | 入力 | 出力 | 内容 |
|---|---|---|---|---|
| 5 -> 4 | `fuzzy_logic()` | `50000000`, `50000010` | `40000000`, `40000001` | 患者判別・年齢を三角型ファジィ数へ変換 |
| 5 -> 4 | `pose_similarity()` | `50000100`-`50000103`, `60010002` | `40000010`-`40000016` | 姿勢特徴量と危険動作テンプレートの類似度 |
| 5/6 -> 4 | `object_risk()` | 物体座標, 本人座標 | `40000100`-`40000102` | 点滴・車椅子・手すりとの距離リスク |
| 5/6 -> 4 | `staff_risk()` | スタッフ座標・移動量, 本人座標 | `40000110`, `40000111` | スタッフ不在・視線喪失に相当するリスク |
| 4 -> 3 | `fuzzy_multiply()` | `40000000`, `40000001` | `30000000` | 内的・静的リスク |
| 4 -> 3 | `AHP_weight_sum()` | `40000010`-`40000016` | `30000001` | 内的・動的リスク |
| 4 -> 3 | `AHP_weight_sum()` | `40000100`-`40000102` | `30000010` | 外的・静的リスク |
| 4 -> 3 | `fuzzy_reasoning_master()` | `40000110`, `40000111` | `30000011` | 外的・動的リスク |
| 3 -> 2 | `simple_weight_sum()` | `30000000`, `30000001` | `20000000` | 内的リスク。現状重みは `[0.1, 0.9]` |
| 3 -> 2 | `fuzzy_reasoning_master()` | `30000010`, `30000011` | `20000001` | 外的リスク |
| 2 -> 1 | `fuzzy_reasoning_master()` | `20000000`, `20000001` | `10000000` | 総合危険度 |

リスク値は原則 `0.0` から `1.0` の連続値で、`10000000`, `40000010`, `40000110`, `40000111` などは「値が大きいほど危険」です。

## 特徴量・リスクのナンバリング

ノード番号は 8 桁の整数で管理されています。厳密なフォーマット定義ファイルはありませんが、現行コードでは次の規則で運用されています。

| 先頭桁 | 層 | 役割 |
|---|---:|---|
| `1` | 第1層 | 最終出力。総合危険度 |
| `2` | 第2層 | 内的リスク・外的リスク |
| `3` | 第3層 | 内的/外的 × 静的/動的の中間リスク |
| `4` | 第4層 | 解釈可能な要因別リスク |
| `5` | 第5層 | センサ・画像・前処理から得る入力特徴量 |
| `6` | 第6層 | 人物の幾何情報など、リスク変換前の基本状態 |
| `7` | 第6層相当 | 背景差分などの実験的特徴量 |

番号の中盤はカテゴリを表します。たとえば `40000010` は第4層の内的・動的リスク、`50001110` は第5層のスタッフ関連特徴量です。

### 上位リスクノード

| ノード | 名前 | 意味 | 計算 |
|---:|---|---|---|
| `10000000` | `total_risk` | 総合危険度 | `20000000`, `20000001` を Fuzzy 統合 |
| `20000000` | `internal_risk` | 内的危険度 | `30000000`, `30000001` の重み和 |
| `20000001` | `external_risk` | 外的危険度 | `30000010`, `30000011` を Fuzzy 統合 |
| `30000000` | `internal_static_risk` | 内的・静的危険度 | 患者判別と年齢のファジィ統合 |
| `30000001` | `internal_dynamic_risk` | 内的・動的危険度 | 危険動作リスクの AHP 重み和 |
| `30000010` | `external_static_risk` | 外的・静的危険度 | 周辺物体リスクの AHP 重み和 |
| `30000011` | `external_dynamic_risk` | 外的・動的危険度 | スタッフ距離・見守り喪失の Fuzzy 統合 |

### 第4層: 解釈可能な要因別リスク

| ノード | 意味 | 入力 | 値の向き |
|---:|---|---|---|
| `40000000` | 患者である | `50000000`, `50000001` | 三角型ファジィ数 |
| `40000001` | 高齢である | `50000010`, `50000011` | 三角型ファジィ数 |
| `40000010` | 立ち上がり動作 | `50000100`-`50000103` または `60010002` | 高いほど危険 |
| `40000011` | 車椅子ブレーキ解除 | `50000100`-`50000103` | 高いほど危険 |
| `40000012` | 車椅子を動かす | `50000100`-`50000103` | 高いほど危険 |
| `40000013` | バランスを崩す | `50000100`-`50000103` | 高いほど危険 |
| `40000014` | 手を挙げる/動かす | `50000100`-`50000103` | 高いほど危険 |
| `40000015` | せき込む | `50000100`-`50000103` | 高いほど危険 |
| `40000016` | 顔を触る | `50000100`-`50000103` | 高いほど危険 |
| `40000100` | 点滴の近くにいる | `50001000`, `50001001`, `60010000`, `60010001` | 近いほど高い |
| `40000101` | 車椅子に乗っている/近い | `50001010`, `50001011`, `60010000`, `60010001` | 近いほど高い |
| `40000102` | 手すりから離れている | `50001020`, `50001021`, `60010000`, `60010001` | 遠いほど高い |
| `40000110` | スタッフが近くにいない | `50001100`, `50001101`, `60010000`, `60010001` | 遠いほど高い |
| `40000111` | スタッフが見ていない | `50001100`, `50001101`, `50001110`, `50001111`, `60010000`, `60010001` | スタッフ移動方向が対象から外れるほど高い |

`master_v5.py` の `40000010` は `60010002` が存在する場合、対象者高さ最大値からシグモイドで立ち上がり度を計算し、`60010002` がない場合は `50000100` をそのまま使います。`risk_core` では、`60010002` がない場合も他の動作リスクと同じく 4 次元姿勢テンプレート類似度で評価します。

### 第5層: 入力特徴量

| ノード | 意味 | 主な生成元 |
|---:|---|---|
| `50000000` | 対象が患者かどうか。`yes` / `no` | `preprocess_zokusei_snapshot.py`, `preprocess_blip.py` |
| `50000001` | 患者判別の信頼度 | BLIP または色判定 |
| `50000010` | 年齢カテゴリ。`young` / `middle` / `old` | BLIP または固定値 |
| `50000011` | 年齢推定の信頼度 | BLIP または固定値 |
| `50000100` | 立位・座位の度合い | YOLO pose、bbox 縦横比、擬似データ |
| `50000101` | 体幹の傾き | YOLO pose |
| `50000102` | 手首の動き/腰からの距離 | YOLO pose |
| `50000103` | 足首の開き/下肢姿勢 | YOLO pose |
| `50001000` | 最近傍点滴の x 座標 | 施設構造または BLIP/アノテーション |
| `50001001` | 最近傍点滴の y 座標 | 施設構造または BLIP/アノテーション |
| `50001002` | 点滴存在ポテンシャル | `PreprocessObject.gauss_func()` など |
| `50001003` | 点滴存在ポテンシャルの補助列 | 現状は `50001002` と同値で扱われることが多い |
| `50001010` | 最近傍車椅子の x 座標 | 施設構造または BLIP/アノテーション |
| `50001011` | 最近傍車椅子の y 座標 | 施設構造または BLIP/アノテーション |
| `50001012` | 車椅子存在ポテンシャル | `PreprocessObject.gauss_func()` など |
| `50001013` | 車椅子存在ポテンシャルの補助列 | 現状は `50001012` と同値で扱われることが多い |
| `50001020` | 最近傍手すり/壁の x 座標 | 施設構造から最近傍壁を選択 |
| `50001021` | 最近傍手すり/壁の y 座標 | 施設構造から最近傍壁を選択 |
| `50001100` | 最近傍スタッフの x 座標 | 実データのスタッフ位置、またはスタッフステーション |
| `50001101` | 最近傍スタッフの y 座標 | 実データのスタッフ位置、またはスタッフステーション |
| `50001110` | 最近傍スタッフの x 方向変化 | 現在位置 - 前フレーム位置 |
| `50001111` | 最近傍スタッフの y 方向変化 | 現在位置 - 前フレーム位置 |

`network/graph_manager_v3.py` には `50001022`, `50001023` も残っていますが、現行の `master_v5.py` と `risk/schema.py` では正本入力として使われていません。互換性のために残っている予約・過去実験用列と見なすのが妥当です。

### 第6層・実験的特徴量

| ノード | 意味 | 補足 |
|---:|---|---|
| `60010000` | 対象者 x 座標 | 距離リスク計算の基準 |
| `60010001` | 対象者 y 座標 | 距離リスク計算の基準 |
| `60010002` | 対象者高さ最大値 | 存在する場合は `40000010` の立ち上がり判定に使う |
| `70000000` | 背景差分値 | `pseudo_data_generator.py` などに残る実験用特徴量 |

## リスク計算の要点

### 姿勢リスク

`50000100`-`50000103` は、次の 4 次元姿勢特徴量として扱われます。

```text
[立位度, 体幹傾き, 手首特徴, 足首特徴]
```

`master_v5.py` の `risky_motion_dict` は、この 4 次元ベクトルに対する危険動作テンプレートを持ちます。各リスクは、テンプレートとの差分平均から類似度を計算し、`similarity ** 4` で強調されます。ただし、`master_v5.py` の立ち上がり `40000010` は `60010002` がある場合に高さベースのシグモイド値で、ない場合に `50000100` で上書きされます。`risk_core` は `60010002` がある場合だけ高さベースのシグモイド値を使い、ない場合はテンプレート類似度を維持します。

## 新規MVP: risk_core / scenario_sim

`master_v5.py` を直接置き換えず、読みやすいリスク評価コアとシナリオ実行基盤を `master_thesis_modules/risk_core/` と `master_thesis_modules/scenario_sim/` に追加しています。

主な入口は次の通りです。

```bash
python -m master_thesis_modules.scenario_sim.runner.run_scenario \
  --scenario master_thesis_modules/scenario_sim/scenarios/reach_object_context_demo.yaml
```

このMVPでは、YAML内に `40000010` などのノード番号を書かず、`action_label`, `position`, `iv_pole`, `wheelchair`, `handrail`, `staff` のような意味名から `FeatureFrame` を生成します。`RiskEngine` は `FeatureFrame` を受け取り、第4層リスク、簡易上位リスク、`total_risk`、説明文を返します。

生成AIにシミュレーションシナリオを作らせる場合の YAML フォーマット、許可される `action_label`、イベント種別、作成ルールは `master_thesis_modules/scenario_sim/README.md` の「シナリオYAML仕様」にまとめています。

動作リスク `40000010`-`40000016` は、`master_v5.py` の `risky_motion_dict` と `pose_similarity()` を踏襲しています。

```text
similarity = 1 - mean(abs(reference_pose - observed_pose))
risk = similarity ** 4
```

`40000010` は `height_max` がある場合、既存実装と同じく次の式で上書きします。

```text
risk = 1 / (1 + exp(-5 * (height_max - 1)))
```

テストは次で実行できます。

```bash
pytest tests/risk_core tests/scenario_sim
```

### 周辺物体リスク

物体リスクは人物座標と最近傍物体座標のユークリッド距離から計算します。正規化係数は `sqrt(2) * 6` です。

点滴 `40000100` と車椅子 `40000101` は、対象に近いほど危険です。

```text
risk = 1 - clip(distance / (sqrt(2) * 6), 0, 1)
```

手すり `40000102` は、対象が手すりから遠いほど危険です。

```text
risk = clip(distance / (sqrt(2) * 6), 0, 1)
```

### スタッフ関連リスク

スタッフ距離 `40000110` は、スタッフが遠いほど大きくなります。

```text
risk = clip(distance / (sqrt(2) * 6), 0, 1)
```

スタッフ見守り喪失 `40000111` は、対象者への相対位置ベクトルとスタッフ移動ベクトルの cos 類似度から計算します。

```text
cos_theta = dot(patient - staff, staff_velocity) / (|patient - staff| * |staff_velocity|)
risk = 1 - (cos_theta / 2 + 0.5)
```

スタッフが対象者へ向かうほど低リスク、対象者から外れる方向に動くほど高リスクになります。`risk_core` ではスタッフ位置がない場合は最大リスク `1.0`、スタッフ速度がない/ゼロで方向を定義できない場合は中立値 `0.5` とします。スタッフが検出されない場合、`preprocess/staff_watch.py` は施設構造の `staff_station` 座標と `direction` を代替値として入れます。

## Fuzzy 推論と個人差

Fuzzy 推論は `scripts/fuzzy/fuzzy_reasoning_v5.py` が担当します。基本の membership は `low`, `middle`, `high` です。

`Master.__init__()` は `/media/hayashide/MasterThesis/common/TFN_<staff_name_fuzzy>.csv` を読み込み、`define_custom_rules()` でスタッフごとの判断傾向を反映します。たとえば、内的リスクを重く見る評価者と外的リスクを重く見る評価者で、同じ入力に対する `10000000` が変わります。

AHP 重みは `scripts/AHP/get_comparison_mtx_v3.py` の `getConsistencyMtx().get_all_comparison_mtx_and_weight()` から読みます。`master_v5.py` では `staff_name_ahp` を `AHP_array_type` として渡し、`中村` / `百武` などの評価者別に実行する想定です。

## データ入出力

`Manager.get_database_dir()` は、既定で外部ストレージ `/media/hayashide/MasterThesis` を使います。ローカル実行では、このパスに実データ・共通 CSV・出力 trial ディレクトリがある前提です。

主な入力・出力は次の通りです。

| 種類 | 形式 | 例 |
|---|---|---|
| 入力特徴量 | `data_dicts.pickle` または `data_<patient>_raw.csv` | `/media/hayashide/MasterThesis/20251211_MasterThesisData/data_dicts.pickle` |
| 評価済みデータ | `data_<patient>_eval.csv` | `data_00021_eval.csv` |
| 評価グラフ | `graph_dicts.pickle`, `graph_dicts.json` | ノード接続と重み |
| 図 | `result_<node>.pdf` など | `visualizer_v5.py` が生成 |

`master_v5.py` の `__main__` は、`中村` と `百武` の AHP/Fuzzy 組み合わせを総当たりし、`20251211_MasterThesisData_<AHP評価者頭文字><Fuzzy評価者2文字目>` の trial 名で保存するように書かれています。

## 実行例

リポジトリ直下から実行する想定です。

```bash
python master_thesis_modules/scripts/master_v5.py
```

論文用プロットの代表的な再生成は次です。

```bash
python master_thesis_modules/scripts/visualize/visualizer_v5.py
```

ただし、どちらも入力データパスや trial 名がスクリプト内に直書きされています。実行前に `__main__` 付近の `trial_name`, `data_dicts` 読み込み元、`staff_names` を確認してください。

## テスト

正本パイプラインの回帰テストは `master_thesis_modules/scripts/tests/` にあります。

```bash
pytest -q -rx \
  master_thesis_modules/scripts/tests/test_risk_direction_regression.py \
  master_thesis_modules/scripts/tests/test_simulation_ideal_regression.py \
  master_thesis_modules/scripts/tests/test_real_data_ideal_regression.py
```

特に `test_risk_direction_regression.py` は、リスク値の向き、スタッフ見守りリスク、物体座標の扱い、順位付けを確認します。

新構成のテストは次で実行できます。

```bash
pytest tests/risk_core tests/scenario_sim tests/real_data
```

## 新構成の実行例

修論4.5相当の時系列シミュレーション:

```bash
python -m master_thesis_modules.scenario_sim.runner.run_thesis_simulation \
  --scenario master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml \
  --model spatial_context \
  --output outputs/thesis_4_5_new
```

比較モデル実行:

```bash
python -m master_thesis_modules.scenario_sim.runner.compare_models \
  --scenario master_thesis_modules/scenario_sim/scenarios/reach_object_context_demo.yaml \
  --models action_only action_attribute spatial_context \
  --output outputs/reach_context_comparison
```

旧 `master_v5.py` の `staff_names = ["中村", "百武"]` ループに相当する、AHP/Fuzzyプロファイル総当たり。入力シナリオは旧シミュレーションに合わせますが、`risk_core` の立ち上がりリスクとスタッフ速度ゼロ時の扱いは現行ロジックを優先するため、旧CSVと完全一致しない場合があります。

```bash
python -m master_thesis_modules.scenario_sim.runner.run_profile_sweep \
  --scenario master_thesis_modules/scenario_sim/scenarios/thesis_4_5_multi_patient_action_demo.yaml \
  --staff-names 中村 百武 \
  --common-dir master_thesis_modules/database/common \
  --output outputs/thesis_4_5_profile_sweep \
  --visualize
```

計算済みのプロファイル総当たり結果だけを可視化する場合:

```bash
python -m master_thesis_modules.scenario_sim.runner.visualize_profile_sweep \
  --input outputs/thesis_4_5_profile_sweep
```

実データ評価:

```bash
python -m master_thesis_modules.real_data.runner.run_real_data_eval \
  --input /path/to/data_dicts.pickle \
  --output outputs/real_data_eval_new \
  --visualize
```

新旧比較:

```bash
python -m master_thesis_modules.real_data.runner.compare_real_data_with_legacy \
  --new outputs/real_data_eval_new/ahp_中村__fuzzy_中村 \
  --legacy /path/to/legacy_eval_csv_dir \
  --output outputs/real_data_compare
```

`run_thesis_simulation` は `risk_timeseries.csv`, `ranking.csv`,
`notification_log.csv`, `explanations.json` と、対応するPNG可視化を出力します。

## レガシーファイルの扱い

過去実装は多く残っています。削除されていない理由は、過去の結果再現、実験比較、または参照用です。現行説明では詳述しなくて構いません。

| ファイル/ディレクトリ | 位置づけ |
|---|---|
| `scripts/master.py` | 初期版の統合実装 |
| `scripts/master_v2.py` | シミュレーション・実験・basic check を内包した旧系統 |
| `scripts/master_v3.py`, `scripts/master_v4.py` | `graph_manager_v3` や旧 Fuzzy 実装を使う中間版。文字列ノード番号が混じる |
| `scripts/master_v6.py` | `v5` に近い派生版。現状は `EVALUATION_STEPS` との対応記録がないため正本扱いしない |
| `scripts/master_basic_check*.py` | 入力・推論の簡易検証用 |
| `scripts/master_multiple_risks.py`, `scripts/master_comprehensive_analysis.py` | シミュレーション条件を変えた包括評価用 |
| `scripts_202511/` | 2025年11月頃のアンケート集計、データ復旧、クレンジング、予備分析 |
| `scripts/depreciated/` | 古い前処理・スロットリング検証 |

正本に関係する変更をする場合は、まず `master_v5.py`, `risk/schema.py`, `graph_manager_v3.py`, `fuzzy_reasoning_v5.py`, `tests/` を見るのが最短です。

## ChatGPT に説明するときの要約

このリポジトリは、人物ごとの時系列特徴量から転倒・見守りリスクを階層的に推定する修論用コードである。特徴量とリスクは 8 桁のノード番号で管理され、先頭桁が階層を表す。第5・6層の入力特徴量、たとえば姿勢 `50000100`-`50000103`、物体座標 `50001000` 系、スタッフ座標・速度 `50001100` 系、本人座標 `60010000` 系を、第4層の解釈可能なリスク `40000010`, `40000100`, `40000110`, `40000111` などへ変換する。その後、AHP、Fuzzy 推論、重み和で `300000xx`, `200000xx` へ集約し、最終的に総合危険度 `10000000` を出す。現行の正本は `master_v5.py` で、評価順序と主要ノード仕様は `risk/schema.py` に記録されている。
