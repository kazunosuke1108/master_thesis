# REQ-003 手法・実装修正計画を設ける段階と監査項目

日付：2026-10-17。担当：architect。対象：[REQ-003 revision 2](../requests/REQ-003-evaluationDesign.md) R-03 / AC-03。今回の承認範囲は読取調査・エージェント間の議論・報告であり、以下の計画候補の採用や数式・実装修正は未承認である。

## 1. 計画を作る段階の提案

実装監査は評価方法の議論と並行して今開始し、修正計画の詳細化は主要シナリオ・入力要件・比較条件を仮合意した後に行うことを提案する。先に全コードを整理すると、今回の本人順位との一致の検証に必要な修正と、保守上の整理が混在する。一方、本人順位を取得してから算出方法を修正すると、評価対象と評価基準の独立性が失われやすい。

| 段階 | 作成・確認するもの | 次工程へ進む条件の提案 |
| --- | --- | --- |
| 現在：評価の議論と並行 | 原稿・現コード・既存出力の差分、今回使用する入口の候補、未確認事項 | 既修正と未修正を区別し、研究目的に関係する項目を選ぶ |
| 評価条件の仮合意後 | 対象シナリオ、本人へ提示する情報、知識入力先、特徴量定義、修正理由・出力影響・受入条件を含む要求案 | 研究責任者が算出仕様・修正範囲を承認する |
| 承認後、予行前 | 最小限の修正、独立期待値による計算検証、同一入力での新旧差の記録 | 入力・知識・出力の接続が検証される。予行の結果による追加変更は再度差分管理する |
| スタッフ順位の取得前 | コード版、入力、プロファイル、提示資料、順位の時点・区間、同順位処理、分析手順の固定 | 評価で使う版と算出条件を追跡できる |

AI回答は評価設計の予行用とし、算出式をAI順位へ合わせ込む根拠にしない。実スタッフ評価後に修正が必要になった場合も、その評価に使った元の版と結果は保持し、修正版の扱いを研究責任者が決める。ここでいう条件は提案であり、主要指標やデータ分割の確定ではない。

## 2. 調査時点のコード版と変更状況

- master_thesis：branch `202610`、HEAD `4673175fe8e0fb3a3c3b1b055e69b10846bb46a4`。最新commitは2026-10-12の「2000/0000をFuzzyに戻した」。調査開始時の `git status --short` に追跡済みファイルの変更はなく、要求書、既存調査報告、`paper/latest/` などが未追跡だった。他担当の作業と並行しているため、この状態は調査開始時の記録である。
- master_thesis_journal：branch `main`、HEAD `0bdbb6745995ee246f19154dcd263e364f13b62c`。調査時の `git status --short` は出力なし。読取のみ。
- 旧調査 [current_implementation_review_20261012.md](../docs/current_implementation_review_20261012.md) は変更前の証拠である。特にK01の「内的統合へ回答が反映されない」という指摘は、現行risk_coreへそのまま適用できない。

## 3. ユーザーが述べた既修正事項の確認

ユーザー原文の「もう修正したけど20000000の算出がαの重み和だったりね」は、**現行risk_coreのFuzzy有効経路についてコード・履歴・既存要求書で裏付けられる**。

| 経路・資料 | 確認した事実 | 解釈の範囲 |
| --- | --- | --- |
| [risk_engine.py:97](../master_thesis_modules/risk_core/engine/risk_engine.py#L97) | `use_legacy_like_fuzzy=True` では内的静的・動的リスクを `LegacyLikeFuzzyAggregator` で統合 | 現行の既定経路ではα固定和を修正済み |
| [risk_config.py:25](../master_thesis_modules/risk_core/engine/risk_config.py#L25) | Fuzzy有効が既定。False時に用いる0.1/0.9も残る | 別経路の固定係数の存在だけでは今回の不整合と断定できない |
| [profile_config.py:95](../master_thesis_modules/risk_core/engine/profile_config.py#L95) | TFN CSVの先頭4行の中心値を内的後件HH/HL/LH/LLへ接続。TFNがない場合は質問票先頭4回答へ接続 | 回答者名の指定時に内的後件を取得できなければエラーとなる。旧K08の「Fuzzy取得不能を常に黙って代替」も現状に合わせて限定が必要 |
| [REQ-001](../requests/REQ-001-useFuzzy.md)、[実装報告](2026-10-12-REQ-001-implementation.md) | REQ-001 revision 2の承認と実装・検証完了が記録され、commit `4673175` に変更が含まれる | 既存報告は関連107件成功等を記録。今回は再実行していない。実データ性能改善は同報告でも未検証 |
| [master_v5.py:457](../master_thesis_modules/scripts/master_v5.py#L457) | 旧実装の通常evaluateには0.1/0.9加重和が残る | 旧実装全体も修正済みとは言えない。今回これを使うか先に確認する |
| v18 PDF、journal [access.tex:330](/home/hayashide/kazu_ws/master_thesis_journal/access.tex:330) | 内的統合はα式、同350行は全評価α=0.1との説明 | 現行risk_coreと原稿に版の差がある。原稿の旧数値を新式で得た結果と扱えない |

Fuzzy式の現実装は、有限な入力 `x,y∈[0,1]` について

```text
F(x,y) = cHH*x*y + cHL*x*(1-y)
       + cLH*(1-x)*y + cLL*(1-x)*(1-y)
```

に対応する。根拠は [fuzzy.py:65](../master_thesis_modules/risk_core/aggregators/fuzzy.py#L65) の積の適合度と中心値による正規化、同99行のhigh/low会員関数である。TFNの幅を含む一般的な出力集合の面積重心と自動的に同一視しない。これは実装から読める式の記述であり、今回の数式変更提案ではない。

## 4. 評価に使用し得る入口と原稿との差

| 対象 | 現行コードの接続 | 監査上の扱い |
| --- | --- | --- |
| シナリオの個人知識評価 | [run_profile_sweep.py:94](../master_thesis_modules/scenario_sim/runner/run_profile_sweep.py#L94) → YAML入力生成 → `make_profile_risk_config` → `RiskEngine` | AHPとFuzzyを直積で実行できる。本人のAHPとFuzzyの両方を反映する条件では、両者を同じ本人にすることが基本候補。異なる回答者の組合せは感度解析等として区別する |
| シナリオの簡易実行 | [run_thesis_simulation.py:38](../master_thesis_modules/scenario_sim/runner/run_thesis_simulation.py#L38) → 未指定 `RiskConfig` → `RiskEngine` | 個人名付き知識読込と同一条件ではない |
| 実データの個人知識評価 | [run_real_data_eval.py:52](../master_thesis_modules/real_data/runner/run_real_data_eval.py#L52) → 保存DataFrameのadapter → 同factory → `RiskEngine` | 個人の判断知識を計算へ反映する処理はコード上で確認。bagから生成された具体的入力と原稿出力の対応は今回確認していない |
| legacy | `scripts/master_v5.py` | READMEが「正本」と呼んでいても、新runnerがこのevaluateを呼ぶ証拠にはならない。再現用経路と今回の評価経路を分けて記録する |

動作集約は、現行runnerとfactoryの既定が `weighted_sum`、v18・journal本文は `max_j(w_j*r_j)` である。ただし [README.md:370](../README.md#L370) 以降には `--action-aggregation weighted_max` を明示したシナリオ・実データのコマンドが残る。したがって、CLI既定値から過去の実験条件を決めつけられない。新しい評価では方式を明示し、過去出力は実行版・設定の証拠を別途確認する。和とmaxの切替はスコアと順位を変え得るため、修正や採用は研究責任者の承認対象とする。

現行 `patient_context` は空間要因を0にし、最終値として内的リスクを直接返す（[risk_engine.py:129](../master_thesis_modules/risk_core/engine/risk_engine.py#L129)）。v18の比較説明とjournalも内的リスクを直接返す記述であり、旧v12に対するK02の指摘をそのまま「現原稿と不一致」とは言えない。ただし通常階層の空間入力だけ0にした条件とは算出構造が違う。あり／なし比較を採用する場合の意味はPM・researcherとの議論で決める。

## 5. 研究の意味を確認する監査項目

優先対象は、今回の本人順位との照合に使う経路である。未使用のlegacyや通知機構の全面整理は先行条件にしない。

| 確認対象 | 現行根拠と問題になり得る点 | 修正計画化の前に必要な確認 |
| --- | --- | --- |
| 回答→設定→算出 | `profile_config.py:41–81,95–119,139–170`。AHPの不足時は既定重みが残り、ファイル解決には旧共通ディレクトリへの代替がある | 使用した回答、項目順、TFN後件順、CSVの実体と解決先、AHP CI/CRを確認。回答者名だけで全入力が本人由来と断定しない |
| 固定係数と個人知識の分担 | 距離の正規化、動作テンプレート、スタッフ統合規則等は共通。内的・外的・総合の後件やAHPは本人入力を使える | どこを共通設計、どこを本人入力とするか明記。固定値の全面排除は求めない。共通とした理由と感度を必要範囲で記録する |
| 見守り状況の意味 | [staff_risk.py:26](../master_thesis_modules/risk_core/factors/staff_risk.py#L26) は移動方向を代理にする。停止・方向不明は0.5。最近傍スタッフ1人を使用 | アニメで注視を表現してもシステムには速度方向しか入らない可能性がある。スタッフへ提示する状況と計算入力の差を点検。視線導入や複数人統合は追加の研究設計変更として扱う |
| 隠れた入力切替 | `run_profile_sweep.py:95–98` はシナリオ名 `thesis_4_5_multi_patient_action_demo` のとき互換固定DataFrameを使用 | 今回流用するシナリオが該当するか確認。該当時はYAMLの位置・動作編集が評価入力へ反映されると仮定しない |
| 特徴量・単位・欠損 | [feature_encoder.py:22](../master_thesis_modules/scenario_sim/encoder/feature_encoder.py#L22) は位置と速度を生成し、[dataframe_adapter.py:92](../master_thesis_modules/risk_core/features/dataframe_adapter.py#L92) は姿勢欠損や本人位置欠損を0へ代替する | m、s、方向の定義、物体不在と未観測の違い、姿勢・高さの適用分岐を保存入力と照合。欠損方針の変更は結果への影響を提示して承認を受ける |
| 順位の照合単位 | [_outputs.py:73](../master_thesis_modules/scenario_sim/runner/_outputs.py#L73) はtimestamp完全一致で集約し、同スコアも連番にする | スタッフ回答が静止時点か動画区間か、対象患者、同順位・判断不能の許容と採点を決める。ID順等による同点時の並びを本人不一致と扱わない |
| 再現と原稿 | 出力保存関数はCSV・説明等を保存するが、同関数でcommit・全設定・入力hashのmanifest保存は確認できない | 各結果の生成版・引数・回答ファイル・入力来歴・使用期間を対応付ける。原稿の旧図表は再評価済みと混同しない |

AHPの不整合やFuzzy後件の非単調性を診断しても、回答を自動修正しない。回答の転記・項目順の誤りと、本人の判断を表す回答内容を分ける。後者を除外・補正する基準は、本人知識を反映するという研究目的に影響するため研究責任者が決める。

## 6. 計画候補を採用した後の検証案

計算の検証では内的・外的・総合Fuzzyの四隅と中間値を独立計算し、AHPの項目順と重みを手計算可能な入力で確認する。空間条件を比較する場合は、変更対象以外の入力・知識設定が同じことを入力ダンプで確認する。本人プロファイルではAHPとFuzzyの回答者IDと出所の一致を検査する。

接続の検証では、提示シナリオ、保存特徴量、中間ノード、総合値、対象時点の順位を1組の識別子で対応付ける。実データについては、既存の保存特徴量を利用できるか先に確認し、bagから全再構築することを無条件の前提にしない。式や入力定義を変えた場合は、変更前後の差を記録し、旧出力との一致を受入条件にしない。

これらは将来の検証候補である。今回はコードのimport、実験、テスト、データ変換、環境変更を行っていない。

## 7. 実施記録と未確認事項

読取には `git status --short`、`git branch --show-current`、`git rev-parse HEAD`、`git log`、`git show --stat 4673175`、`rg -n`、`nl -ba`、`pdftotext -layout paper/latest/paper_full_draft_v18.pdf -` を使用した。PDFは抽出時に数式・英数字の一部が文字化けするため、読み取れる説明とjournalのTeXを照合した。v18とjournalが同じ版であることを保証したものではない。

報告の論証は `japanese-tech-writing` Skillの規範で確認し、事実・提案・未確認を区別した。外部Web調査は行わず、ローカルのコード・履歴・文書を根拠とした。今回の書込は本報告のみで、ソース・データ・数式・原稿・外部リポジトリを変更していない。commit/pushも未実施。

AC-03に対して、計画を設ける段階と監査項目を提案し、20000000の既修正を経路別に確認した。残る未決事項は、今回使う評価入口と算出条件、原稿図表の生成版・実引数、本人提示情報と入力特徴量の対応、具体的な修正対象・受入条件である。計画自体の承認と実装・再評価は今回の範囲外である。
