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
  --output outputs/real_data_eval_new
```

## 旧実装との比較

```bash
python -m master_thesis_modules.real_data.runner.compare_real_data_with_legacy \
  --new outputs/real_data_eval_new \
  --legacy /path/to/legacy_eval_csv_dir \
  --output outputs/real_data_compare
```

比較では、ノードごとのMAE、最大誤差、順位一致率、1位一致率を出力します。

## 既知の差分

新実装は `master_v5.py` と完全な数値一致を保証する段階ではありません。特に、評価者別TFN CSVと全AHPプロファイルを常に自動読込するわけではないため、旧出力との差分は `compare_real_data_with_legacy` で定量確認してください。

