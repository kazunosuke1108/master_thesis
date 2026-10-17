---
request_id: REQ-003
request_revision: 7
updated_at: "2026-10-17T22:04:48+09:00"
status: prototype-verified
issued_at: "2026-10-17T21:48:22+09:00"
---

# 試作・AI予行の実行報告

研究の課題・解決策に対する結論は[総括015](2026-10-17-015-REQ-003-research-assessment.md)にまとめた。本書は実行内容と技術的な検証の記録である。

## 到達点と成果物

承認済み計画の最初の段階として、指定瞬間の配置図、知識36判断と場面順位の回答フォーム、本人知識を使う解析、AI予行を実行した。研究全体と実スタッフ調査は未完了である。算出式、主要指標、原データ、既存出力、原稿は変更していない。commit・pushは行っていない。

| 成果物 | 参照・用途 |
| --- | --- |
| 場面の根拠 | [調査011](2026-10-17-011-REQ-003-scene-inventory.md) |
| 試作仕様 | [仕様012](2026-10-17-012-REQ-003-prototype-specification.md) |
| 回答フォーム | [questionnaire.html](../outputs/REQ003_pilot_20261017T165939/bundle-v4/participant/questionnaire.html) |
| CLIと使用説明 | [usage](../master_thesis_modules/evaluation_study/2026-10-17-001-usage.md) |
| 3場面と時点 | [manifest](../studies/REQ-003/prototype.yaml)。C01/C02は既存、C03はスタッフ位置だけを変えた派生 |
| フォーム回答の取込 | [import_questionnaire.py](../studies/REQ-003/import_questionnaire.py)。両回答種別を保持して本人解析へ接続 |
| AI知識変換 | [convert_pilot_knowledge.py](../studies/REQ-003/convert_pilot_knowledge.py) |
| 独立査読 | [査読014](2026-10-17-014-REQ-003-prototype-review.md) |
| 最終再現結果 | [final-reproduction-check.json](../outputs/REQ003_pilot_20261017T165939/final-reproduction-check.json) |

回答者へ渡すのは `bundle-v4/participant/` のみ。解析側の知識源、元ID、算出値、個人名を含み得る対応表は渡さない。生成物は専用ディレクトリ配下の.gitignoreで除外する。過去のbundle・回答・解析は残し、最終版には別の出力先を使用した。

## 実装した接続

manifest → ScenarioLoader → 指定時刻以下のイベント適用 → 同一状態の提示・特徴量化 → 本人のAHP/Fuzzyを使う既存RiskEngine → 本人順位との行単位の照合、という接続を試作した。全回答の不正検出、同順位・判断不能、入力不足時の拒否、既存出力の上書き拒否を含む。主要評価指標や統計検定は実装していない。

フォームは動作AHP21・物体AHP3・Fuzzy12の36判断と3場面の順位を含む。開始・完了時刻と単調時計による経過時間を記録し、完了後は固定する。初期回答は空欄で、未回答のまま完了できない。JSON・順位CSV・知識CSVをローカル保存できる。Googleフォーム配布は未実施である。

最終査読で、人の全回答JSONから本人解析への取込が未完と分かったため、専用importerを追加した。既存のAI専用converterの制限は緩めていない。専用importerはschema、回答種別、項目順、知識値、場面hash、時刻、全順位を検証してから新規保存する。合成・人試行の両ラベルで、実Chrome出力から既存profile・評価まで自動fixtureで確認した。人の実回答を取得した検証ではない。

物体の同一座標ラベルが重なっていたため、ラベルだけを別行にした。位置マーカー、元座標、算出入力は保持した。Chrome画像で車椅子と点滴台の両ラベルをPMも確認した。

## AI予行の結果と限界

別スレッドの合成スタッフ役2名が、それぞれ知識回答と3場面の順位を回答した。回答役にはシステム出力や実スタッフの知識を渡していない。回答は保存し、合うように修正していない。患者順はP01、P02、P03、P04である。

| 回答役 | 場面 | 本人役の順位 | 算出順位（sum/max共通） | 診断 |
| --- | --- | --- | --- | --- |
| S01 | C01 | 1,3,2,4 | 1,3,2,4 | 一致 |
| S01 | C02 | 1,3,1,3 | 1,2,3,4 | 相違。本人役は同順位 |
| S01 | C03 | 1,3,2,4 | 1,3,2,4 | 一致 |
| S02 | C01 | 1,3,2,4 | 1,2,3,4 | 相違 |
| S02 | C02 | 1,2,3,4 | 1,2,3,4 | 一致 |
| S02 | C03 | 1,3,2,4 | 1,2,3,4 | 相違 |

各集約で6回答中3場面の順位ベクトルが一致し、3場面は相違した。これは接続確認の診断であり、研究の主指標や成功率として採用していない。両集約で順位は同じでもスコアは異なる。この結果だけで最終集約を選ばない。

最初のS01回答は通常回答のstatusが `answered` で、validatorが拒否した。元回答と失敗記録を残し、本人役が `ranked` とした形式のみの再回答を採用した。順位・理由は変えていない。生成案内にも正常時のstatusを明記し、テストを追加した。

回答役からは、スタッフが誰を見ているか、物体の使用・接続・ブレーキ状態、立ち上がりの不安定さが分からないとの指摘があった。現在の入力が示すのは位置や姿勢presetであり、これらの状態を推定して補ってはいない。提示説明と、実映像を採用するときの観測範囲を今後整理する。

実スタッフ14名分の完全な既存知識についても、3場面×4患者を両集約で出力した（各168行）。本人順位を新たに取得した結果ではない。既存回答を現在の本人判断と同一とは扱わない。

AI予行は全フォーム実装前の知識回答・順位回答である。後からフォームで回答時間を測ったことにはしない。モデル詳細・生成seedは取得できておらず、保存した入力からの算出は再現できても、AI回答の再生成は保証できない。AI予行は論文の実スタッフ評価の証拠に使わない。

## 再現条件と検証

Git HEADは `8cad4a3a10aee4196cdc75de59bcec4fa6177dc7`、branchは `202610`。作業開始時のHEADと異なるため再確認した。作業ツリーは未コミットで、各生成物のprovenanceに入力・ソースSHA256、設定、日時、引数を記録している。最終study.pyのSHA256は `fbdea536183e743397a395528dbc2527e1ab3e61195608bf897d01bee5b2b2fd`。

環境はPython3.11.7、numpy1.26.3、pandas2.1.4、PyYAML6.0.1。算出は乱数を使わずseedはnull。フォームの自動入力時間を人間の所要時間として扱わない。

```bash
python -m master_thesis_modules.evaluation_study bundle studies/REQ-003/prototype.yaml outputs/REQ003_pilot_20261017T165939/bundle-v4
python -m master_thesis_modules.evaluation_study validate studies/REQ-003/prototype.yaml outputs/REQ003_pilot_20261017T165939/responses/all.csv
python -m master_thesis_modules.evaluation_study evaluate studies/REQ-003/prototype.yaml outputs/REQ003_pilot_20261017T165939/synthetic-profiles.yaml outputs/REQ003_pilot_20261017T165939/synthetic-weighted_sum-final --action-aggregation weighted_sum --responses outputs/REQ003_pilot_20261017T165939/responses/all.csv
python -m pytest tests/evaluation_study tests/studies tests/scenario_sim tests/risk_core -q
```

上記は実行したコマンドの記録である。再実行時は出力先を新規パスにする。weighted_maxも明示して別出力先で実行した。既存14名の解析は専用の非公開profilesを使用した。

| 確認 | 結果 |
| --- | --- |
| 再開後の回帰 | 173 passed in 16.99s。接続追加後の最終回帰は204 passed in 21.41s。Chrome実DOMを使うフォーム・取込検証を含む |
| 最終ソースでの再評価 | 合成各24行、既存各168行。両集約の4件のsystem-ranks.csvが以前のv1とバイト一致 |
| 独立再計算 | reviewerが384件のスコア・順位一致を確認 |
| 追跡 | 最終bundleの88ソースhash、4解析のprovenanceが現行と一致とreviewerが確認 |
| 人間の5分達成 | 未検証。知識36判断を含むので順位だけの所要時間で代用しない |

フォームJSONの取込と解析は次の形で実行する。元JSONと来歴を保持し、出力先は新規に限る。

```bash
python studies/REQ-003/import_questionnaire.py SOURCE.json studies/REQ-003/prototype.yaml NEW_IMPORT_DIR
python -m master_thesis_modules.evaluation_study evaluate studies/REQ-003/prototype.yaml NEW_IMPORT_DIR/profiles.yaml NEW_ANALYSIS_DIR --action-aggregation weighted_sum --responses NEW_IMPORT_DIR/responses.csv
```

## 次に判断する事項

1. まず試作フォームを使い、人が説明・知識36判断・順位を通して答える負担と時間を測る。5分を超えても、項目を無断削除しない。
2. 本調査の主要指標・同順位の評価規則・合否、sum/maxの採用を具体化して研究責任者と合意する。予行不一致を埋めるために算出式を調整しない。
3. 位置と姿勢だけで判断できる場面の説明を整え、実画像を使う場合は保存データとの時刻・人物対応を確認する。

## 実行・変更・引継ぎ履歴

| ID | 発生日時 | 記録日時 | 担当 | 内容・結果 |
| --- | --- | --- | --- | --- |
| E01 | 2026-10-17（実装・受信秒未記録） | 2026-10-17T21:48:22+09:00 | implementer・PM | 試作実装、AI2名予行、既存14名出力。詳細は保存回答・provenance |
| E02 | 2026-10-17T17:08:15+09:00 | 2026-10-17T21:48:22+09:00 | implementer | SVGラベル修正、41試作テストと全回帰167件合格を返却 |
| E03 | 2026-10-17T21:47:00+09:00 | 2026-10-17T21:48:22+09:00 | PM | 中断位置を照合し、要求された引継ぎをcheckpointへ保存 |
| E04 | 2026-10-17（再開後、検証終了秒未記録） | 2026-10-17T21:48:22+09:00 | PM | bundle-v4生成、173件回帰と最終4条件再計算・旧出力不変を確認 |
| E05 | 2026-10-17T21:50:02+09:00 | 2026-10-17T21:50:02+09:00 | implementer→PM | フォーム取込接続を追加。専用31件合格。human_trialも自動fixtureであり実回答ではない |
| E06 | 2026-10-17T21:50:28+09:00 | 2026-10-17T21:50:28+09:00 | PM | 接続追加後の204件回帰合格。独立査読のRV03解消と最終hashを確認 |
| E07 | 2026-10-17T21:51:28+09:00 | 2026-10-17T21:51:28+09:00 | PM | 要求書・報告011〜014・usageの日本語校正、参照リンク存在、原文不変、Git差分を確認。結果を下記へ記録 |

## 最終文書確認

`yomiyasu`による静的検査と差分確認を行い、要求書の原文部分が再開時の保存版と一致することを確認した。全対象Markdownの相対リンク先は存在する。校正で算出式・数値・採否は変更していない。

静的検査の候補のうち、要求書の矢印はユーザー原文のため保持した。仕様の「所有者ではなく位置」はモデルの誤解を防ぐ説明として保持した。usageの同一語尾は手順と動作の説明に必要な自然な敬体として保持した。報告011・013・014の静的検査は指摘なし。検査結果は[document-lint.txt](../outputs/REQ003_pilot_20261017T165939/document-lint.txt)、最終テスト・ソースhashは[final-verification.json](../outputs/REQ003_pilot_20261017T165939/final-verification.json)へ保存した。

| 発生日時 | 記録日時 | 担当 | 追記 |
| --- | --- | --- | --- |
| 2026-10-17T22:04:48+09:00 | 2026-10-17T22:04:48+09:00 | PM | 冒頭へ研究上の総括015の参照を追加。過去の結果・記録は保持 |
