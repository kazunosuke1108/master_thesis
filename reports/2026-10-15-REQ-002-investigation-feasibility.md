# REQ-002 追加調査の実施可否

[要求書](../requests/REQ-002-simplifyDataFormat.md) の追加原文（rosbagから特徴量算出まで）を対象に、2026-10-15に読取確認を実施した。今回は実施可否の判定のみ。本調査・変換実行・コード変更は行っていない。

## 判定

コードに基づくパイプライン仕様の調査は実施可能。実際に使用したbag・ブランチ・生成物を結び付けるには追加の証拠確認が必要であり、再実行による検証の可否は未判定。

- `/home/hayashide/ytlab_ros_ws/ytlab_rosbag/ytlab_rosbag_modules` は存在。現在は `hayashide` ブランチ。launchの未コミット変更あり。
- `/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system` は存在。現在は `feature/implement_tracking`。tracking配下4ファイルに未コミット変更あり。
- ローカルremote-tracking refsに抽出・姿勢推定・追跡等の関連ブランチ候補がある。履歴調査はcheckoutせず `git show` / `git log` で可能。refsの最新性は未確認。
- `tracking/T_20240822_extractbag.py`、`get_position_features.py`、`M_20240228_analysisManagement.py` 等の調査候補が存在。名前だけでは実際の使用を確定できない。
- `/media/hayashide/MobileSensing` は存在するが、今回の `findmnt -T` ではルートファイルシステムが返り、独立したNASマウントは確認できなかった。NASへの直接接続やマウントは試していない。
- tmuxとCodex CLIは利用可能。CLIは0.162.1、`multi_agent` はstable/true。公式ドキュメントでも明示依頼によるサブエージェント調査を案内している。

## 引継ぎ条件

外部2リポジトリは読取のみ。ブランチ切替、fetch、入力変換、環境変更は行わず、未コミット変更も区別して調査する。NAS未接続でもコード調査を進め、実データ対応は未確認と明記する。PMが要求書の追加原文に対応する調査範囲を別版として追記し、今回の追加調査の直接指示と以前の実装候補の承認を区別する。

researcherをbag抽出側と特徴量側に分けて並列調査し、両結果が揃ってから別reviewerが入出力の接続と根拠を確認する。全員が同じファイルを書かず、PMがreports配下へ統合する。

成果はrosbagのtopic・型・時刻・座標系、抽出物、追跡・特徴量、master_thesis入力までの対応表、実行条件と依存、ブランチ/commit/未コミット差分の識別、確定/推定/未確認の区別を含む。生データや個人情報を報告へ転記しない。

確認コマンド：`git status --short --branch`、`git branch -a`（外部2リポジトリ）、`rg --files`、`findmnt -T /media/hayashide/MobileSensing`、`codex --version`、`codex --help`、`codex features list`。seedは使用していない。外部コード・データを変更していない。


## 2026-10-15 本調査の開始追記

上記は実施可否確認時点の記録として保持する。その後のユーザーの直接指示により、REQ-002 revision 2 R-04〜06 / AC-05〜07の読取調査と仕様文書作成を開始した。要求書の追加原文は保持し、調査範囲・禁止事項・実行根拠を[要求書のrevision 2欄](../requests/REQ-002-simplifyDataFormat.md)へ追記した。以前のデータ形式変更案の実装承認は得ていない。

追加のNAS所在情報はユーザーの記憶として記録した。`//192.168.1.5/common` の `FY2024/09_MobileSensing` と旧 `/media/hayashide/MobileSensing` の対応を候補として検証し、common全体とは対応付けない。開始時の `findmnt -rn -t cifs,smb3,fuse.gvfsd-fuse` は該当接続なし、旧パスの `findmnt -T` はext4のルートを返し、現在ユーザーのGVfsディレクトリも存在しなかった。新規接続は実施していない。


## 本調査の成果への参照

[統合仕様書](2026-10-15-REQ-002-pipeline-specification.md)、[bag側](2026-10-15-REQ-002-bag-investigation.md)、[追跡・特徴量側](2026-10-15-REQ-002-feature-investigation.md)、[独立レビュー](2026-10-15-REQ-002-pipeline-review.md)へ調査結果を保存した。コードからの処理仕様は復元したが、NAS未接続のため実bag・中間生成物・最終入力の同一来歴は未確認。実装変更・変換・環境変更は実施しておらず、以前のデータ形式変更案も実装未承認のままである。
