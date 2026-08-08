# real_data

`real_data` は、既存の実データ入力を新しい `risk_core` パイプラインで評価するためのモジュールです。

## 対応入力

- `data_dicts.pickle`: `dict[patient_id, pandas.DataFrame]` 形式
- `data_<patient>_raw.csv` を含むディレクトリ
- `data_<patient>_eval.csv` を含むディレクトリ

外部ストレージ `/media/hayashide/MasterThesis` には依存せず、入力パスはCLI引数で指定します。

## 実データ評価

```bash
python -m master_thesis_modules.real_data.runner.run_real_data_eval \
  --input /path/to/data_dicts.pickle \
  --output outputs/real_data_eval_new \
  --visualize
```

既定では `["中村", "百武"]` のAHP/Fuzzyプロファイル直積を評価し、次のようにプロファイル別ディレクトリへ保存します。

```text
outputs/real_data_eval_new/
  ahp_中村__fuzzy_中村/
  ahp_中村__fuzzy_百武/
  ahp_百武__fuzzy_中村/
  ahp_百武__fuzzy_百武/
  visualization/
```

プロファイル名を変更する場合は `--staff-names`、通知ログ生成に使うスタッフ数を変更する場合は `--staff-count` を指定します。

```bash
python -m master_thesis_modules.real_data.runner.run_real_data_eval \
  --input /path/to/data_dicts.pickle \
  --output outputs/real_data_eval_new \
  --staff-names 中村 百武 \
  --staff-count 1
```

`--staff-names all` を指定すると、`--common-dir` 内で動作AHP・物体AHP・Fuzzyの3ファイルが揃った全スタッフを使います。AHP を山口に固定し、Fuzzy を全スタッフで掃引する場合は次のように実行します。

```bash
python -m master_thesis_modules.real_data.runner.run_real_data_eval \
  --input /path/to/data_dicts.pickle \
  --output outputs/real_data_eval_ahp_yamaguchi \
  --staff-names all \
  --ahp-staff-names 山口 \
  --common-dir master_thesis_modules/database/common \
  --model spatial_context \
  --action-aggregation weighted_max \
  --notification-message-style legacy \
  --visualize
```

`--model` で文脈の使い方を切り替えます。`spatial_context` は患者属性・年齢・動作に加えて、周辺物体とスタッフ見守りも使います。`patient_context` は患者属性・年齢・動作だけを使い、空間的文脈を総合危険度に入れない比較手法です。

```bash
python -m master_thesis_modules.real_data.runner.run_real_data_eval \
  --input /path/to/data_dicts.pickle \
  --output outputs/real_data_patient_context \
  --staff-names 山口 百武 \
  --model patient_context \
  --visualize
```

既に計算済みの出力を後から可視化する場合は、次を実行します。

```bash
python -m master_thesis_modules.real_data.runner.visualize_profile_sweep \
  --input outputs/real_data_eval_new
```

## Fuzzyプロファイルと患者危険度の後解析

実データのプロファイル掃引出力にも、シナリオシミュレーションと同じFuzzyプロファイル後解析を適用できます。`C_i` と患者別の正規化平均総リスクの関係を折れ線グラフとCSVへ出力します。AHPプロファイルが複数ある出力では、比較対象を `--ahp-profile` で指定してください。

```bash
python -m master_thesis_modules.real_data.runner.analyze_fuzzy_profile_rankings \
  --input outputs/20260726_realdata \
  --ahp-profile 山口
```

既定では `<input>/analysis/` に、`C_i` と `D_i` の各グラフおよびCSVを保存します。`D_i = c(10行目) - c(7行目)` です。

## 旧実装との比較

```bash
python -m master_thesis_modules.real_data.runner.compare_real_data_with_legacy \
  --new outputs/real_data_eval_new/ahp_中村__fuzzy_中村 \
  --legacy /path/to/legacy_eval_csv_dir \
  --output outputs/real_data_compare
```

比較では、ノードごとのMAE、最大誤差、順位一致率、1位一致率を出力します。

## 既知の差分

新実装は `master_v5.py` と完全な数値一致を保証する段階ではありません。特に、評価者別TFN CSVと全AHPプロファイルを常に自動読込するわけではないため、旧出力との差分は `compare_real_data_with_legacy` で定量確認してください。
