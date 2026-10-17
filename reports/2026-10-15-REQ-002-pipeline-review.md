# REQ-002 revision 2：パイプライン仕様の独立レビュー

2026-10-15。対象は[要求書](../requests/REQ-002-simplifyDataFormat.md) revision 2のR-04〜06 / AC-05〜07。調査担当とは別スレッドで、[bag担当報告](2026-10-15-REQ-002-bag-investigation.md)、[特徴量担当報告](2026-10-15-REQ-002-feature-investigation.md)、[PM統合仕様](2026-10-15-REQ-002-pipeline-specification.md)をレビューした。以前のデータ形式変更案は実装未承認である。

## 判定と確認範囲

確認した範囲では、統合仕様の主要な接続について重大度「高」の誤りは見つからなかった。コード上の実装経路を記述し、特定bagから最終入力までの実使用来歴を未確認とする結論は妥当である。NAS実物や動作を検証済みとする記述も見つからなかった。

静的確認の対象版は、bag側 `a98fceb135f77bce9a031ae91f37dbbbd2bfd6a3`、handheld側 `aa6d11923069ff3ea51b74b86297b86e13ee79c5`、master_thesis側 `4673175fe8e0fb3a3c3b1b055e69b10846bb46a4` と調査時の作業ツリー。以下のHはhandheldの `ytlab_handheld_sensoring_system_modules/`、Mはmaster_thesisの `master_thesis_modules/` を表す。根拠行は確認時点の作業ツリーであり、文書追記で報告側の行番号は変わり得る。

| 照合した境界 | 独立確認した根拠 | 判定 |
| --- | --- | --- |
| bag選択・再生とCSV抽出の役割 | bag側のHEAD再生launchを `git show`、作業ツリーlaunch 2〜5行、TF診断5〜25行で照合 | bag側の確認範囲を再生・記録・診断とし、CSV抽出はhandheld側として分ける記述は妥当 |
| ROS処理とCSV出力 | H:launch/tracking/csvExporter_left.launch:17〜29、38〜65、80〜103。H:scripts/tracking/csvExporter_left.py:55〜90、162〜273 | 独自DualSemanticRGBDと左CompressedImageの購読、20×11スロット、別画像時刻、最新map TF、座標中央値は実装に対応 |
| message schemaと同期 | H:msg/DualSemanticRGBD.msg、上記exporter:84〜90 | トップレベルHeaderなし。nested画像headerに基づく厳密同期や実時刻整合を保証しないという限定は妥当 |
| rawラベルと注釈 | H:scripts/tracking/M_20240228_analysisManagement.py:448〜470、annotator.py:129〜172 | 0→NaN、全欠損列削除、線形補間・ID別bfill・画像最近傍対応の存在を確認。実データに適用された規則は別途確認が必要 |
| 特徴量算出 | M:scripts/preprocess/preprocess_blip.py:112〜154、preprocess_yolo.py:121〜183、preprocess_others.py:29〜99 | 姿勢式、物体への人物位置代入、固定スタッフID、差分の単位、壁投影の説明は実装に対応 |
| 人物別CSVから最終入力 | M:real_data/loader/raw_csv_loader.py:12〜18、risk_core/features/dataframe_adapter.py:18〜47、92〜158、real_data/runner/run_real_data_eval.py:52〜67 | ノードID正規化、timestamp引継ぎ、欠損処理、FeatureFrameSequenceとBatchRiskEngineへの接続を確認 |
| NASルート対応 | H:管理クラス194〜206行と `git show 55f0717aba1d4eaf069618dae1c4ccf03c25069a:.../M_20240228_analysisManagement.py` の該当関数 | 同じNASK設定に対するUNCサブディレクトリと旧Linuxパスの対応がある。common全体への対応拡大は支持されない |
| 実使用を示す履歴の扱い | handheld `31512c50...`、master_thesis `eca1261a...` のcommit文を独立に照合 | 開発意図・作業履歴の証拠として扱う記述は妥当。実行完走・生成物同一性の証明にはならない |

20 Hz統合・追跡の全内部関数、全ブランチ、全履歴、依存パッケージの全版を独立に再調査したわけではない。担当報告の引用を読み、工程間の接続に関係する上記コードを優先して確認した。

## 指摘と反映確認

### RV-01：スタッフ差分の計算順序（中、反映確認済み）

初読時の特徴量報告81行と統合表の「スタッフx/yの1行差分」だけでは、欠損補填後の位置列から差分を取ると読めた。M:preprocess_others.py:69〜77では、annotation座標の転記、diff、座標fillna、差分fillnaの順に処理する。欠損区間や復帰直後には両解釈の値が一致しない。

修正文案は「annotation転記直後・座標fillna前の差分。座標欠損は固定地点、差分欠損は固定directionで別々に補填」。最新の担当報告・統合表でこの順序が明記されたことを確認した。コード修正は提案していない。

### RV-02：保存物体confidenceと最終入力の境界（低、反映確認済み）

初読時の特徴量表では、物体位置のconfidence列 `50001002/03` と `50001012/13` が省略されていた。M:preprocess_blip.py:136〜151は、VQA confidenceをx/y用の2列へ同値で保存する。一方、dataframe_adapter.py:110〜119は座標のみを取り込む。

修正文案は「この4列は保存されるが現行FeatureFrame変換では未使用。位置測定の不確かさとは区別」。最新の両表への反映を確認した。これにより保存CSVと最終入力のschemaの差が明確になった。

### RV-03：Headerなし同期の保証範囲（低、反映確認済み）

H:csvExporter_left.py:90の `ApproximateTimeSynchronizer(...,100,10,allow_headerless=True)` とDualSemanticRGBDのトップレベルHeader欠如を併記するだけでは、nested画像header同士を合わせると誤解される余地があった。slop=10も含め、厳密同期を主張する根拠にはならない。

修正文案は「nested画像headerを基準に厳密同期する契約ではなく、実行時のmessage_filters実装と対応画像を別途確認する」。最新の担当報告・統合仕様で注意が明示された。今回、message_filters実行環境の挙動を検証したわけではない。

## 未確認事項と残るリスク

NASは本レビューで接続・列挙していない。旧パスから `FY2024/09_MobileSensing/<相対パス>` への対応はコード上の候補であり、現在の実在・内容・過去のマウント実績は未検証である。要求書とbag報告の追加UNC原文については、実文字の先頭backslash 2個・区切り1個を読取確認した。

bag記録時刻、画像header、画像ファイル名時刻、20 Hz統合時刻、最終time_sは別の契約である。コード上の区別が文書化されても、実データのずれ・重複・時刻原点の妥当性を確認したことにはならない。実bagのmetadata、TF、CSVのtimestampと画像名を対応付ける必要がある。

人手ID修正とfixposition生成、前処理間の行順、画像解像度、人物IDの一致は未確認である。パス二重付加や固定試行設定もあるため、掲載コマンドを現HEADで完走可能な手順と判断できない。これは統合仕様でも未解決として区別されており、調査事実の過大な断定には当たらない。

bag直読の画像抽出器 `H:scripts/tracking/T_readRosbag.py:13〜75` も読取確認した。左右compressed画像をheader時刻で期間選択し、fps間引きしてJPGを保存する別経路である。topic購読のleft exporterと同一工程とは扱えず、最終特徴量への実接続も未確認。担当の追加補足と統合時にはこの限定を維持する必要がある。

## 実施した検証と権限

ファイルの読取、`nl`、`sed`、`rg`、`git show`、`git log`、`git status`、原文文字数の読取確認のみを行った。実装のimport、モデル起動、変換、評価、書込を伴うテスト、checkout、fetch、マウント・認証、commit・pushは行っていない。reviewerの書込は本報告のみ。文章の論理確認にはjapanese-tech-writingの手順を用い、未確認事項を断定へ置換していない。

本レビューはAC-07の独立確認に対応する。AC-05の実装仕様記述とAC-06の未接続時の限定は確認範囲で支持されるが、実データの来歴確定や動作再現の達成を意味しない。
