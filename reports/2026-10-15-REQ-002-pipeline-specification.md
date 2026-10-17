# REQ-002 rosbagから特徴量・最終入力までの調査仕様

対象は[要求書revision 2](../requests/REQ-002-simplifyDataFormat.md)のR-04〜06 / AC-05〜07。2026-10-15の直接指示に基づく読取調査であり、以前のデータ形式変更案の実装承認は含まない。実装やデータ変換を実行せず、コード上の仕様と実使用を裏付ける記録を分ける。

## 調査版と変更の保持

| 対象 | 現在ブランチ | HEAD | 既存未コミット変更 |
| --- | --- | --- | --- |
| ytlab_rosbag_modules | hayashide | a98fceb135f77bce9a031ae91f37dbbbd2bfd6a3 | launch/rosbag_play.launch |
| ytlab_handheld_sensoring_system | feature/implement_tracking | aa6d11923069ff3ea51b74b86297b86e13ee79c5 | tracking配下の管理・画像処理・注釈4ファイル。担当調査に内訳を記載 |
| master_thesis | 202610 | 4673175fe8e0fb3a3c3b1b055e69b10846bb46a4 | 開始時はREQ-002要求書と既存報告3件が未追跡。追跡コードの差分なし |

remote-tracking refはローカル保存時点の内容であり、fetchしていない。実際の採用ブランチを現在ブランチだけで断定しない。

## NASパス対応の根拠と制約

追加情報はユーザーの記憶である。原文は要求書に保存した。接続先共有は `//192.168.1.5/common`、調査対象はその中の `FY2024/09_MobileSensing` とする。

対応候補は次のとおりである。

| 旧絶対パス | SMB共有内のパス | 根拠と確度 |
| --- | --- | --- |
| `/media/hayashide/MobileSensing/<相対パス>` | `FY2024/09_MobileSensing/<相対パス>` | ユーザー記憶と下記コードの対照。コード上の対応は確認、実マウント記録・現物は未確認 |
| `/media/hayashide/MobileSensing/Nagasaki20240827/jpg/<sensor>/<file>` | `FY2024/09_MobileSensing/Nagasaki20240827/jpg/<sensor>/<file>` | visualizer_v5.pyのIP UNC書換えはこの試行に固定。他試行へ一般化できない |

master_thesis HEADの `master_thesis_modules/scripts/visualize/visualize_video.py:284–287` には、同じ `fullrgb_imagePath` に旧マウント先を付加する実装と、`//NASK/common/FY2024/09_MobileSensing` を付加するコメントが併存する。導入履歴候補は `git log --all -S '//NASK/common/FY2024/09_MobileSensing' -- <file>` の `1ec692c`。NASKとIPの同一性はDNSや共有実物で未確認である。

同HEADの `master_thesis_modules/scripts/visualize/visualizer_v5.py:566–572` には `//192.168.1.5/common/FY2024/09_MobileSensing/Nagasaki20240827/jpg/{sensor}/` と画像basenameからパスを構成する処理がある。導入履歴候補は `git log --all -S '//192.168.1.5/common/FY2024/09_MobileSensing/Nagasaki20240827/jpg/' -- <file>` の `25f93c7`。これはパスを構成する実装の証拠であり、NASに同名ファイルが存在する証拠ではない。

調査開始時、`findmnt -rn -t cifs,smb3,fuse.gvfsd-fuse` に該当マウントはなく、`findmnt -T /media/hayashide/MobileSensing` はルートext4を返した。旧ディレクトリは空で、現在ユーザーのGVfsディレクトリも存在しない。NAS実データは未照合。共有ルートcommon全体を旧マウント先とは対応付けていない。

新規接続が必要なため、マウントを作らず `smbclient //192.168.1.5/common -U <ユーザー名>` で対話認証し、対象サブディレクトリの一覧だけを確認する案をユーザーへ提示した。許可・認証方法の確認なしには実行しない。パスワードをコマンド引数、文書、履歴へ保存しない。NAS接続を待たずにコード調査を完了する。

## 調査方法と再現条件

調査はファイル読取とローカルGit参照に限定した。`git status --short --branch`、`git rev-parse HEAD`、`git branch -a`、`git show <ref>:<path>`、`git log --all`、`git diff`、`rg -n`、`rg --files` を用いる。履歴検索の各対象は担当報告に記載する。リポジトリ全体のあらゆる過去版について、文字列の不存在を証明する調査ではない。

読取ホストはLinux 6.8.0-136-generic、既定Pythonは3.11.7。PATH上のrosbagと `/opt/ros` は確認できなかった。このホスト情報を過去の実行環境と同一視しない。依存関係はDockerfile・package.xml・requirements等から記述し、インストールの完全性やモデル重みの有無は実行検証していない。seedは使用していない。後述の処理コマンドはコードから整理した候補であり、今回の実行記録ではない。

PMは調査途中に外部2リポジトリのHEAD、ブランチ、status、`git diff HEAD --binary` のSHA-256を控え、終了時と照合する。外部の未コミット変更は仕様の一部として識別するが、調査中に変更しない。報告・要求書以外のコード・設定・原データは書き換えず、commit・pushもしない。

## bag側の処理契約

詳細根拠は[bag担当調査](2026-10-15-REQ-002-bag-investigation.md)に記載した。指定されたytlab_rosbag_modulesは主に再生・記録・TF診断を担い、確認範囲ではCSV・画像抽出の実装を確認できなかった。bag再生を必須工程とするかは下流の入力方式で異なる。

| 工程 | 入出力 | 時刻・座標系 | 仕様上の注意 |
| --- | --- | --- | --- |
| GUI・shellから再生 | 選択bagの絶対パス → roslaunchのfilename | HEADは `rosbag play --clock`。座標変換なし | 現在の未コミットlaunchは `--pause -l` と `use_sim_time=false` を追加。過去実行版は未確定 |
| TF診断 | 固定bagの `/tf` → frame名を標準出力 | bag時刻もtransform時刻も利用しない。親子frame名のみ | `tf2_msgs/TFMessage` を想定するコード。実bagの型確認ではない。特徴量やCSVの出力なし |
| 別refのcompressed再配信 | `/usb_cam/image_raw` のcompressed → `/image_raw` | 再生＋image_transport | `origin/devel`の仕様。下流での採用未確認 |

コード上のbag名は `_2022-05-05-17-42-27.bag`、`_2023-06-12-15-05-55.bag`、`_2023-12-02-21-52-57.bag`。defaultまたはTF診断の固定値であり、今回のMobileSensingデータと結び付く証拠はない。共有内のbag相対パスをこれらから推測しない。

ROS Noetic / Ubuntu 20.04を意図したDocker構成があるが、GUIとlaunchの `/rosbag` に対し、Docker bindは `/catkin_ws/src/rosbag` で不一致がある。実行候補は `roslaunch ytlab_rosbag_modules rosbag_play.launch filename:=<bag絶対パス>`。環境・実データ・使用launchが未確認のため、今回動作を確認したコマンドではない。関連refsのcommitとtopic一覧、依存、履歴検索範囲は担当報告を参照する。

## 抽出から最終入力までの接続仕様

以下は[追跡・特徴量担当調査](2026-10-15-REQ-002-feature-investigation.md)を統合した仕様である。根拠のファイル行、特徴式の詳細、起動形式は同報告を参照する。全工程を一つの実データで実行した証拠は得ていない。

| 工程 | 入力 → 出力・引継ぎ | 時刻・座標・重要条件 |
| --- | --- | --- |
| bag直読の画像抽出（別経路） | `T_readRosbag.py` → 左右compressed画像のJPG | 画像header時刻で期間選択しfps間引き。最終特徴量への実接続は未確認。根拠は独立レビューの同スクリプト13〜75行確認 |
| bag直読の区間抽出 | `T_20240822_extractbag.py` → rosbag filterによる部分bag | filterはbag記録時刻の範囲を使う。必ずこの区間抽出が前段に使われたとは未確認。topic購読exporterの画像header時刻と区別 |
| handheld側の認識・抽出launch | bag → `/rgbd` → `/semantic_rgbd` → `/semantic_rgbd_with_cloud` → raw CSV・人物crop・全景JPG | `csvExporter_left.launch`は再生も含む。bag側の単独再生launchとの直列実行を必須としない。検出はYOLO、点群はbboxへ付与 |
| left exporter | 独自 `DualSemanticRGBD` と左 `CompressedImage` → 1時刻＋20スロット×11項目＋全景pathのヘッダなしraw CSV | CSV時刻はleft rect画像headerをfloat秒化。全景名は別同期画像時刻の小数3桁。同期queue=100、slop=10。点群frame→mapは最新TF、各軸中央値 |
| rawの列名付与 | raw → host別スロット名、0→NaN、全欠損列削除 | 0が座標値として意味を持つ場合も、この処理で欠損化される仕様。変更はしていない |
| センサー統合 | raw群 → 20 Hz最近傍merge → HDBSCAN cluster CSV | 同期間隔0.05秒、採用差<0.05秒、x/yでクラスタリング。元行再利用と欠損あり |
| 時間追跡 | cluster/raw様式 → ID付きreid CSV | Munkres割当＋距離補正＋Kalman、dt=0.05。YOLO tracker IDと後段IDは別契約 |
| 注釈・補間 | ID付きCSV → annotation、全景/差分画像path列 | bbox/x/y補間とID別bfill、画像最近傍対応が存在。人手ID修正・fixposition生成の手順と実使用は未特定 |
| 特徴量作成 | annotation＋全景/差分画像 → BLIP、YOLO pose、位置処理 → 人物別 `data_<id>_raw.csv` | 前処理間で入力名が異なる。転記は行番号対応でありtimestamp joinではない |
| 現行最終入力 | CSVディレクトリ／`dict[str, DataFrame]` pickle → FeatureFrameSequence → BatchRiskEngine | timestampをそのままtime_sへ。危険度・順位算出は今回実行も変更もしていない |

rawスロットの11項目は `x,y,z,imagePath,object_id,bbox_lowerX,bbox_higherX,bbox_lowerY,bbox_higherY,bbox_value,bbox_label`。人物がいる行だけを保存する。crop pathは抽出正規表現のリスト表現、全景pathは絶対パスで、ファイル内のパス形式も統一されていない。

画像topicは `/<hostname>/elp/{left,right}/usb_cam/image_rect_color`（Image）、CameraInfo、および左の `.../image_raw/compressed`（CompressedImage）がコード上の入力候補である。合成・検出・点群付与のdual modeは独自 `DualSemanticRGBD` を用い、左右画像・CameraInfo・range/reflec画像・PointCloud2・メタデータ・左右BoundingBoxArrayを含む。トップレベルHeaderがないため、単一の共通stampがあるとはみなさない。`allow_headerless=True` の同期器がnested画像headerを基準に厳密同期する実装とは確認できず、slop=10を含む設定の実挙動は未検証である。LiDARと左右画像の対応はtime_shiftを考慮した最近傍で、双方の誤差が1/60秒未満という採用条件がある。実bagにこれらの型・topicが存在するかは未確認。

map座標の原点・方角・校正結果は未確認。追跡の距離ログにはm表記があるが、実データの単位検証はしていない。画像時刻のTF指定がなく最新TFを使うこと、CSV時刻と全景画像名の時刻が別であることから、全工程で厳密同期したという主張はできない。merge期間を作る `datetime.timestamp()` もホストtimezoneに依存する。

## 特徴量と現行入力の対応

| 人物別CSVの列 | 意味・算出元 | 単位と引継ぎの注意 |
| --- | --- | --- |
| timestamp | annotation時刻 | float秒をtime_sへ。相対原点の減算なし |
| 50000000/1、50000010/11 | BLIP2の患者識別／年齢とconfidence | 赤シャツ判定反転という実験固有の患者規則。confidenceは最後の生成トークンのsoftmax最大値 |
| 50000100〜103 | bbox縦横比の立位度、体幹傾斜、手首と股関節の距離、足首と股関節の距離を正規化 | 0〜1へclipする無次元量。3D関節距離ではなくcrop画素座標。式と分母は担当報告に記載 |
| 70000000 | bbox内背景差分の画素平均/255 | 無次元。現adapterの直接入力外 |
| 60010000/1 | annotationの人物x/y | map座標を継承 |
| 50001000/1、50001010/11 | 点滴・車椅子がyesのとき人物x/yを代入 | 物体固有の3D中心ではない |
| 50001002/03、50001012/13 | BLIPの点滴・車椅子判定confidenceをx/y両軸に同値で保存 | 現adapterでは読まない。位置測定の不確かさとは別 |
| 50001100/1 | 固定人物ID_00003のスタッフx/y、欠損は固定地点 | 最も近いスタッフを選ぶ実装ではない |
| 50001110/11 | スタッフx/yの1行差分 | annotation転記直後・座標fillna前の差分を取り、差分欠損は固定directionで埋める。dtで割らないためm/sとは扱えず、追跡KF速度とも別 |
| 50001020/21 | 矩形境界に対する最短の壁投影点 | adapterでは手すり側の入力に渡る。前処理の定義と区別する |

現loaderは数値ノードIDのヘッダをint列名に正規化する。adapterは欠損姿勢・人物位置を0、欠損confidenceを1、任意座標・速度の非有限値をNoneへ変換する。timestamp欠損時の連番補完も存在する。CSV上の欠損状態とFeatureFrame上の値を同一視しない。

`get_position_features.py` はコンストラクタの引数保存のみで算出未実装だった。実装された特徴量経路はmaster_thesisの `preprocess_blip.py`、`preprocess_yolo.py`、`preprocess_others.py` にある。位置補正済みCSVの生成手順や行順整合が未確認のため、この3本だけで現在の最終入力を再現できるとは判定しない。

## NAS対応の追加証拠とbag候補

handheldの `scripts/tracking/M_20240228_analysisManagement.py:194–206` は同じNASKストレージに対しWindowsでは `//192.168.1.5/common/FY2024/09_MobileSensing`、Linuxでは `/media/hayashide/MobileSensing` を選ぶ。HEADと当該関数の作業ツリーは同じで、`55f0717aba1d4eaf069618dae1c4ccf03c25069a` にも対応が残る。これはユーザー記憶を支持するコード証拠であり、OSの過去のマウント記録ではない。

共有内パス候補には `FY2024/09_MobileSensing/Nagasaki20241205193158/csv/annotation/`、同試行の `jpg/diff/`、`PullWheelchairObaachan/csv/annotation/`、`TeaChair/csv/cluster/`、`Nagasaki20240827/csv/raw/` がある。具体名と根拠は担当報告の対応表を参照する。NAS上の存在・内容は未確認。

handheldには `ytpc2024j_2024-12-05-19-31-58_1787.bag`（コメント）と `ytnpc2021h_2024-08-27-19-06-53_51_short.bag`（実行設定）がローカルcatkin_wsを指す例、`/media/hayashide/*HDD*/rosbag/*.bag` を探索する実装がある。bag名時刻と試行名の一致は候補探索に使えるが、NASにbagが保管されていたことも、bagと最終入力の同一来歴も確定しない。NASでのbag相対パスは未特定。

## 実使用の証拠と再現上の未解決点

| 証拠・問題 | 判定と確認方法 |
| --- | --- |
| handheldの `31512c50`、`1cf7f87`、`51d0f154`、master_thesisの `eca1261a` の変更とcommit文 | 動画・差分・reid・特定試行の前処理を開発した証拠。実行完走やbag hashと入力pickleの対応は証明しない。詳細なcommitと日付は担当報告 |
| 現在のブランチと未コミット設定 | 現在のコード状態。過去に使用した版はログ・保存設定・生成物の来歴と照合が必要 |
| Managerと旧前処理のパス二重付加、試行名の不一致 | 現HEADの既定設定をそのまま再現手順として示せない。意図した試行・当時版を確認する。修正しない |
| BLIPのfullimagePath入力とYOLO/othersのfixposition入力 | 手動修正・生成コマンド、人物ID・行順・行数・timestampの同一性を読取照合する必要がある |
| bag再生・抽出launchのtopic供給 | カメラ補正画像・CameraInfo・LiDAR・TFの供給元をbag metadataとlaunchで照合する。bag_pathだけで十分とは断定しない |
| bbox縮小、ID固定の補間・遮蔽処理 | 実際に採用した解像度・規則・対象IDを中間生成物と保存設定で確認する |
| 依存と重み | Dockerfileは構成意図。実image digest、ROS/Python依存版、GPU、BLIP/YOLO重みhash、seedは不明 |

handheldはROS Noetic、Ubuntu20.04/CUDA12.1/ZED4.0系Docker構成、OpenCV、PCL、tf/tf2、numpy/pandas、ultralytics、hdbscan、munkres等を使用するコードを持つ。旧特徴量処理はPyTorch/transformersのBLIP2とultralyticsのYOLO pose等に依存する。検出ノードはファイル名yolov8に対して実際の指定はyolo11x-seg、姿勢はyolo11x-poseであり、ファイル名からモデル版を判断しない。

コードから得たコマンド形式・固定main設定・上書き動作は担当報告に列挙した。実行時にはデータ生成・上書きが生じるため、今回それらのコマンド、モデルimport、コンストラクタの実行はしていない。今回の完了対象は仕様の復元と未確認箇所の特定である。

## 成果と受入条件の確認

| 条件 | 結果 | 根拠・限界 |
| --- | --- | --- |
| AC-05：パイプライン仕様 | コードに基づく仕様化を実施 | [bag調査](2026-10-15-REQ-002-bag-investigation.md)、[追跡・特徴量調査](2026-10-15-REQ-002-feature-investigation.md)と本書。実行版・実bag・生成物の同一来歴は未確認 |
| AC-06：NAS対応と未確認事項 | パス対応のコード根拠と候補を整理 | 実マウント記録、共有上のファイル存在、bag相対パスは未確認。新規接続にはユーザー確認が必要で、未実施 |
| AC-07：独立レビューと統合 | 別スレッドのreviewerによる確認を実施 | [独立レビュー](2026-10-15-REQ-002-pipeline-review.md)。中1件・低2件の指摘は担当調査と本書へ反映 |

PMの照合では3リポジトリのHEAD・ブランチ・追跡ファイルの `git diff HEAD --binary` SHA-256が記録時と終了確認時で一致し、外部2リポジトリのstatusも一致した。master_thesisへの書込はREQ-002要求書の追記とreports/のMarkdownのみ。既存原文は保持した。実装テスト、bag再生、抽出、特徴量算出、危険度評価は実施していない。

残る確認は、許可後のNAS読取照合、bag metadataとTF・画像stampの比較、中間CSVのヘッダ・行順・ID・timestamp・画像pathの連続性、実行ログとcommit・環境の照合である。必要な修正や再実行は今回の承認範囲に含めない。以前の保存形式変更案も実装未承認のままである。

文章はjapanese-tech-writingの論証確認手順に沿い、コード仕様から実使用へ断定を広げていないか、不確実性・単位・時刻・来歴の区別が保たれるかを点検した。
