# REQ-001 内的リスクのFuzzy化

日付: 2026-10-12。対象: [要求書 revision 2](../requests/REQ-001-useFuzzy.md)、R-01〜R-03／AC-01〜AC-04。Q-01のユーザー回答「OK。」に基づいて実装。

## 変更と影響

新risk_coreのFuzzy有効時、内的静的・内的動的から内的リスクをFuzzyで集約する。スタッフTFNの先頭4中心値をHH／HL／LH／LLへ接続する。TFNがない場合は質問票へ代替し、不正TFN・指定スタッフの取得不能・不正な内的後件はエラーとする。スタッフ未指定時は旧既定1／1／0.5／0を使う。Fuzzy無効時の内的加重和とaction_onlyの総合値は維持する。

内的・総合の値と、その後段の順位・通知は変わり得る。実データの性能改善は未検証。スタッフ回答、下位要因、外的・総合Fuzzy式、旧実装、評価指標、データ分割、通知閾値は変更していない。

## 成果物と担当

- [Fuzzy集約](../master_thesis_modules/risk_core/aggregators/fuzzy.py)、[設定読込](../master_thesis_modules/risk_core/engine/profile_config.py)、[評価エンジン](../master_thesis_modules/risk_core/engine/risk_engine.py): implementer。
- `tests/risk_core`: evaluator。
- [アーキテクチャ](../docs/system_architecture.md)、要求・報告: PM。
- 独立レビュー: 実装担当とは別スレッドのreviewer。

## 再現条件

開始時HEAD: `6cb411bfd54775df9c64d8cc3d7669a089574917`。コミット・pushはしていない。
環境: Python 3.11.7、pytest 7.4.0、pandas 2.1.4、numpy 1.26.3、PyYAML 6.0.1、matplotlib 3.8.0。
検証入力: 既存スタッフTFN・質問票、テスト用の一時CSVとFeatureFrame。模擬入力は実データ性能の証拠として扱わない。乱数実験・seed変更なし。

## 検証

作業ディレクトリ: `/home/hayashide/kazu_ws/master_thesis`。

```bash
python -m pytest tests/risk_core tests/scenario_sim tests/real_data -q
git diff --check
```

PM実行結果: **107 passed in 26.15s**。差分検査正常。evaluatorの対象実行は62 passed、関連全体の重複実行は107 passed in 25.59s。対象テスト初回は作業ディレクトリの誤りでno tests ran、訂正後に成功。全体実行の重複は連絡到着前に双方が開始したためであり、今後は実行担当を一本化する。

既存 `database/common/TFN_*.csv` 14件について、`make_profile_risk_config` と `RiskEngine` の生成、内的後件4件の検証が成功。runnerの設定伝達はコード読取で確認: `scenario_sim/runner/run_profile_sweep.py:107`、`real_data/runner/run_real_data_eval.py:59` が同factoryで設定を生成してRiskEngineへ渡す。通常シミュレーションは `run_thesis_simulation.py:38` で未指定RiskConfigを使用する。runner全体の接続は上記既存テストでも検証した。

## 独立レビューと受入条件

別スレッドreviewerの最終読取レビュー: 承認範囲からの逸脱・実装を止める指摘なし。要求書への完了追記、質問票不正回答・TFN列数不正のテスト補完を低重要度で推奨。前者はPMが反映、後者はevaluatorが22件追加し、`python -m pytest -q tests/risk_core/test_profile_config.py` で49 passed in 0.33sを確認。質問票の先頭4位置それぞれの4.5／NaN／範囲外回答、TFNの2列・4列と正常質問票への非代替を検証した。追加後に全体テストは再実行していない（本体コードの変更なし）。

- AC-01: 四隅・中間点の独立期待値、後件差がテスト成功。
- AC-02: TFN・質問票の順序、未指定既定、欠損・不正値エラーがテスト成功。
- AC-03: Fuzzy無効式、action_only総合値、モデル別接続、関連全体テスト成功。
- AC-04: アーキテクチャ・本報告に影響と再現条件を記録。実データ性能向上は未検証。

## 残る課題

広範囲のリノベーションは保留。質問票原本の意味との照合、実データにおける変更後の順位・通知・性能の評価は今回の実装検証に含めていない。

