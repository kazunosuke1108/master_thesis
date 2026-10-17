# REQ-002 revision 2：追跡・特徴量・master_thesis入力の読取調査

調査日：2026-10-15。担当：researcher（追跡・特徴量側）。対象要求は[REQ-002 revision 2](../requests/REQ-002-simplifyDataFormat.md) R-04〜06 / AC-05〜07。ユーザーの「今回、調査と仕様文書作成を直接指示します。実装変更は含みません。」を実行根拠とする。以前の保存形式変更案は実装未承認のまま。

## 問いと対象範囲

外部の `ytlab_handheld_sensoring_system` に残るbagから位置・人物ID・画像を得る処理と、`master_thesis` に渡る特徴量の生成・読込契約を調べた。`~/.codex/AGENTS.md`、要求書、[実施可否報告](2026-10-15-REQ-002-investigation-feasibility.md)を確認した。外部コードとデータは読取のみで、checkout、fetch、変換、環境変更、NAS接続、commit、pushは行っていない。スクリプトのimportも行っていない。管理クラスの初期化やパス取得自体が `os.makedirs` を実行するためである。

根拠の略記は次の通り。`H:` は `/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/`、`M:` は `/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/`。行番号は調査時の作業ツリー。Git履歴は `git show <commit>:<path>` で参照可能。

## 版と変更状況

| 対象 | 調査時の版 | 確認した状態 |
|---|---|---|
| handheld | `feature/implement_tracking`、HEAD `aa6d11923069ff3ea51b74b86297b86e13ee79c5` | `origin/feature/implement_tracking` も同じ。fetch未実施でリモート最新性は不明 |
| master_thesis | `202610`、HEAD `4673175fe8e0fb3a3c3b1b055e69b10846bb46a4` | 作業開始時はREQ-002要求書・関連報告4ファイルが未追跡。調査中はPM等の文書追記が並行 |

handheldの既存未コミット変更は `scripts/tracking/M_20240228_analysisManagement.py`、`T_elpJPG2mp4.py`、`T_getImagePathBystamp.py`、`annotator.py` の4ファイル。前者はmain節の試行名・hostnameと実行呼出しの変更（570行以降）で、194〜206行のNASルート分岐はHEADと同じ。annotatorは描画対象IDの選択とmain節の入力名・annotationモードを変更している。本報告の補間処理129〜172行はその変更部分ではない。差分は保持した。

全ローカルremote-tracking refを `git ls-tree` で検索した結果、`scripts/tracking/get_position_features.py` と `csvExporter_left.py` があるrefは `origin/feature/implement_tracking` だけだった。`origin/feature/implement_poseEstimation` (`ad29232`)、`origin/feat/estimate_camera_pose` (`5d47cc7`)、`origin/devel` (`dd2a051`) も存在するが、名前だけを根拠に今回の入力の使用ブランチとはしない。削除済みref・未取得ブランチは調査できていない。

## コードから復元できる処理

次表は実装として存在する経路であり、一つの実データに対して全工程が実行された証明ではない。

| 段階 | 入力 → 処理 → 出力 | 根拠 |
|---|---|---|
| bag再生と検出 | bag → カメラ/LiDAR合成 `/rgbd` → YOLO検出 `/semantic_rgbd` → 点群付与 `/semantic_rgbd_with_cloud` | `H:launch/tracking/csvExporter_left.launch:17–65,97–103`、`H:scripts/yolov8_node.py:35–46`、`H:src/pointcloud_segmentator.cpp:184–197` |
| 位置・bbox・画像抽出 | DualSemanticRGBDと左ELP圧縮画像 → bbox点群をmapへ変換、座標中央値、人物crop、全景JPG、raw CSV | `H:scripts/tracking/csvExporter_left.py:45–90,162–186,218–273` |
| rawラベル化 | ヘッダなしCSV → host別検出スロットの列名付与、0をNaNへ、全欠損列を削除 | `H:scripts/tracking/M_20240228_analysisManagement.py:448–472` |
| センサー同期・統合 | raw CSV群 → 20 Hz最近傍同期merge CSV → x,yのHDBSCAN → cluster CSV | `H:scripts/tracking/A_20240909_clusterExporter.py:17–79,175–224,268–298` |
| 時間方向の追跡 | clusterまたはraw様式のCSV → Munkres割当・距離補正・Kalman処理 → ID付きreid CSV | `H:scripts/tracking/A_20240909_reidExporter.py:27–81,187–237,273–308,352–431` |
| annotation・画像の補間 | ID付きCSV → bbox/x/y線形補間、ID別bfill、全景/差分画像の最近傍対応 → 同じCSVへ保存 | `H:scripts/tracking/annotator.py:129–172`。人手ID修正そのものを行う工程・操作履歴は未特定 |
| 背景差分 | 全景JPG → グレースケール、指数移動平均背景との差分 → `jpg/diff` | `H:scripts/tracking/eventcamera.py:63–86` |
| master_thesisの属性・物体 | annotation CSV＋全景画像 → BLIP2 VQA → 人物別 `data_<id>_raw.csv` | `M:scripts/preprocess/preprocess_blip.py:32–38,71–77,79–154` |
| master_thesisの姿勢・動き | 上記raw CSV＋annotation＋全景/差分 → YOLO poseと画像統計 → 同raw CSVを更新 | `M:scripts/preprocess/preprocess_yolo.py:40–54,185–278` |
| master_thesisの位置・周辺 | 上記raw CSV＋annotation → 患者/スタッフ/壁座標を転記・算出 → 同raw CSVを更新 | `M:scripts/preprocess/preprocess_others.py:45–103` |
| 最終入力 | 人物別raw CSVディレクトリまたはpickle → dict[str, DataFrame] → FeatureFrameSequence → BatchRiskEngine | `M:real_data/loader/trial_loader.py:9–18`、`raw_csv_loader.py:12–18`、`data_dicts_loader.py:13–19`、`M:real_data/runner/run_real_data_eval.py:52–67` |

`get_position_features.py` は1〜28行を全件確認した。`PositionFeatures.__init__` は引数を保存するだけで、読込・特徴量算出・保存は実装されていない。この名前のスクリプトを完成した特徴量抽出器とみなすことはできない。

`T_20240822_extractbag.py:14–38` はbagファイルに `rosbag filter` を実行する別工程である。`sources/20240822_trial1/rosbag/{yt,hm}*.bag` からhostname=`ytjs2022c`だけを選び、2024-08-22 10:45〜10:50の `t.to_sec()`（bag記録時刻）で両端を含む時間切出しを行う。出力名は `bag_path[:-5]+'1.bag'`。topic条件はなく、CSV・特徴量抽出も行わない。これに対して `csvExporter_left.py` はbagを直接読まず、再生されたtopicを購読する。フィルタのrecord timeと、CSVの画像header時刻は別の時刻である。切出しスクリプトの日時はnaive datetimeから作るため、実行ホストのtimezoneが必要で、今回のデータに適用されたかは未確認。

`A_20241114_csvExporter.py` は左・右bboxを扱う別の抽出器で、left専用とは異なる。launch末尾のbag速度は0.5、left専用launchは0.1であり、ファイル名の類似だけで同一条件とはしない。`P_20241030_postProcess01_local.py:17–41` は `person_or_obj="left"` として `csvExporter_left.launch` を選び、bag名からhostnameを再決定する。main節の `hostname="ytpc2021h"` とbag由来の `ytnpc2021h` の綴りの差もある。

## topic・型・時刻・座標と抽出形式

| 境界 | コード上の契約 | 限界・注意 |
|---|---|---|
| 左右画像 | `/<hostname>/elp/{left,right}/usb_cam/image_rect_color`、`sensor_msgs/Image`。camera_infoは `CameraInfo` | launchの既定値。実bag内の有無は未確認 |
| 元の左画像 | `/<hostname>/elp/left/usb_cam/image_raw/compressed`、`sensor_msgs/CompressedImage` | left exporterがDualSemanticRGBDとApproximateTimeSynchronizer、queue=100、slop=10、allow_headerless=Trueで合わせる (`csvExporter_left.py:82–90`)。DualSemanticRGBDにはトップレベルHeaderがないため、nestedのleft_rect_image.header.stamp同士を厳密同期する契約ではない。Headerなしメッセージの扱いは実行時のmessage_filters実装確認が必要 |
| 合成/検出/点群付与 | `/rgbd` → `/semantic_rgbd` → `/semantic_rgbd_with_cloud`、dual modeでは全て独自 `DualSemanticRGBD` | `msg/DualSemanticRGBD.msg` は左右Image/CameraInfo、range/reflec Image、PointCloud2、Stringメタデータ、左右BoundingBoxArrayを含む。単一のトップレベルHeaderはない |
| bbox | `BoundingBox.msg`：uint32 id/上下左右画素、float32 value、string label、UInt32MultiArray mask、PointCloud2 cloud | idはYOLO trackerのidがある場合に転記 (`yolov8_node.py:130–163`)。後段のID_番号と同一視しない |
| 元画像とLiDARの対応 | dual publisherはLiDAR時刻からtime_shiftを引いた時刻に最も近い左右画像を選び、双方の誤差が1/60 s未満のとき採用 | `rgbd_publisher.py:423–478`。`enable_timeshift`、関連launchの設定に依存し、厳密同時刻ではない |
| raw CSVの時刻 | `left_rect_image.header.stamp`のsecs/nsecsをfloat秒に変換 | `csvExporter_left.py:168–171`、`M_20240228_expCommons.py:60–74`。bagのrecord timeではない |
| 全景JPG時刻 | 別に同期したCompressedImage.header.stampを小数3桁に丸めてファイル名へ | `csvExporter_left.py:181–186`。CSV先頭の時刻と一致する保証はない |
| 空間 | bbox.cloudのframeから `map` への最新TF `lookupTransform(..., rospy.Time(0))` で点群変換し、NaNを除いた点の各軸中央値をx,y,zへ | `csvExporter_left.py:218–251`。画像時刻のTFを明示指定していない。後段reidのログは距離をmと表示 (`A_20240909_reidExporter.py:195–199`) するが、map原点・方角・校正精度を実データで未確認 |
| raw行 | 1時刻＋20スロット×11項目＋全景path。各スロットは x,y,z,imagePath,object_id,bbox_lowerX,bbox_higherX,bbox_lowerY,bbox_higherY,bbox_value,bbox_label | `csvExporter_left.py:166,248–273`、`M_20240228_analysisManagement.py:56`。未使用スロット初期値0、人物がいる行だけ保存。crop pathは `re.findall("database/(.*)", ...)` のリスト表現、全景pathは絶対パスで、形式が混在 |
| 同期CSV | `universal_timestamp=np.arange(start,end,0.05)`、センサーごと最近傍を差<0.05 sで割当 | `A_20240909_clusterExporter.py:64–79`。再利用される元行や欠損を許す。開始終了日時生成は `datetime(...).timestamp()` (`analysisManagement.py:481–485`) でホストtimezone依存 |
| reid | timestampまたはuniversal_timestampをtimestampへ改名、範囲は厳密不等号。20 Hz固定dt、距離2を閾値。観測activeBinaryの合計が40未満かつ末尾5行の合計が0のIDを除去 | `A_20240909_reidExporter.py:41–81,315–334`。2秒相当を意図したパラメータだが厳密には左記条件。実際に除去されたIDや使用パラメータは未確認 |

merge/cluster経路では同一フレーム内の複数センサーの点を統合し、reid経路では時間方向に追跡する。reid割当コストは `||diag(1,1,0.01,0.01)(観測状態−既存状態)||`、ダミーコスト99、Munkresで最小化し、位置のユークリッド距離が2以上なら新IDへ補正する。KFは状態 `[x,y,vx,vy]`、dt=0.05、観測 `[x,y]`。このKF速度と、master_thesisへ入るスタッフの1行差分は別物である。

## 特徴量の意味・式・単位と最終入力

下表のclipは0〜1への切詰め、L/Rは左右、hip/shoulderは両側の平均点。画像点は人物crop内の画素座標で、3D座標ではない。

| 保存列 | コード上の算出 | 単位・根拠 |
|---|---|---|
| timestamp | annotationのtimestampをコピー | 秒のfloat。`preprocess_blip.py:38`。この段階で相対時刻へ変換しない |
| 50000000 / 50000001 | 赤いシャツかVQAし、yesをno、その他をyesへ反転／confidence | ラベル／無次元。患者識別のこの実験固有規則。`preprocess_blip.py:73,112–122` |
| 50000010 / 50000011 | old/middle/youngのVQA／confidence | カテゴリ／無次元。`preprocess_blip.py:74,123–127` |
| 50001000/1、50001010/11 | IV poleまたはwheelchairがyesなら対象人物のannotation x,y、他はNaN | map座標の継承。物体固有の3D中心を検出しているわけではない。`preprocess_blip.py:128–151` |
| 50001002/03、50001012/13 | IV pole／wheelchairのVQA confidenceを各物体のx軸用・y軸用の2列へ同じ値で保存 | 無次元。物体位置の測定不確かさではない。`preprocess_blip.py:138–139,150–151`。現行 `dataframe_adapter.py:110–119` は物体座標のみを取り込み、この4列はFeatureFrame変換に未使用 |
| 50000100 | `clip(((b−t)/(r−l)−2)/(3.5−2))` | 無次元の立位度。コードのstand_ratio=3.5を採用（コメントの3とは異なる）。`preprocess_yolo.py:121–129,249–260` |
| 50000101 | `clip(abs(atan2(abs(shoulder_x−hip_x),abs(shoulder_y−hip_y)))/(π/4))` | 無次元の体幹傾斜。角度そのもののrad保存ではない。`preprocess_yolo.py:131–138` |
| 50000102 | `clip(max(||hip_L−wrist_L||,||hip_R−wrist_R||)/(2||shoulder−hip||))` | 無次元。`preprocess_yolo.py:140–159` |
| 50000103 | `clip(max(||hip_L−ankle_L||,||hip_R−ankle_R||)/||nose−hip||)` | 無次元。名前はankleだが左右足首間距離ではない。`preprocess_yolo.py:161–183` |
| 70000000 | bbox内背景差分画像の画素平均/255 | 無次元。`preprocess_yolo.py:218–220`。現FeatureFrame adapterの直接入力には含まれない |
| 60010000/1 | annotationのID_x,ID_yを行番号対応で代入 | map座標の継承。`preprocess_others.py:64–67` |
| 50001100/1 | 固定ID_00003のannotation x,y、欠損は固定staff_station | map座標の継承。最近傍スタッフ探索ではない。`preprocess_others.py:29–43,69–77` |
| 50001110/11 | annotationのスタッフx,yを転記した直後、座標のfillna前に `.diff()` を計算。その後、座標欠損はstaff_station、差分欠損はstaff_directionで別々に補填 | 座標量/1行でありm/sではない（dt除算なし）。補填後の座標列の差分とは一致する保証がない。`preprocess_others.py:69–77` |
| 50001020/21 | 設定した矩形境界の4候補から最短の壁までの投影点 | map座標の継承。「handrail」入力へ渡るが、この前処理では壁境界。`preprocess_others.py:31–43,79–99` |

confidenceの実装はBLIP出力の最後の生成トークンのsoftmax最大値 (`M:scripts/preprocess/blipTools.py:49–56`) であり、物体位置の測定不確かさではない。YOLO poseでは座標0をNaNに変換し、keypoint confidenceは取得するがこれら式の計算時に閾値選別しない (`preprocess_yolo.py:235–247`)。IDに依存した遮蔽除去規則で姿勢4列と70000000をNaNにする (`264–273`)。

人物IDは `ID_00021` のようなannotation列の接頭辞から得て、CSV保存時は `data_00021_raw.csv` へ変換する。raw CSVにはtimestamp列と数値ノードIDのヘッダがあり、読込後に数字文字列をint列名へ正規化する。pickleの契約は `dict[person_id, pandas.DataFrame]`。`M:scripts_202511/4_修論データのリバイバル/修論データの格納.py:6–20` は `/media/hayashide/MasterThesis/20251211_MasterThesisData/*.csv` を読み、ファイル名第2要素をIDにして `data_dicts.pickle` を保存する実装。ただしglobがrawだけを限定しないことに注意。

現在の `M:risk_core/features/dataframe_adapter.py:18–47,92–120` は、このdictを人物別FeatureFrameSequenceへ変換する。time_sはtimestampをそのままfloat化し、原点を引かない。患者/年齢・confidence・姿勢4量・患者座標・点滴/車椅子/手すり/スタッフ座標・スタッフ速度・任意heightを取り込む。欠損姿勢と患者位置は0、欠損confidenceは1、任意座標/速度は非有限値ならNoneとなる。timestamp列がない場合の連番挿入も実装されているため、入力欠損が単にエラーになる契約ではない。生CSVの欠損とFeatureFrame内の値は区別する。

## NAS内相対パスとの対応

ユーザーの記憶は要求書revision 2へ原文保存されている。コード側でも `H:scripts/tracking/M_20240228_analysisManagement.py:194–206` が同じ `strage=NASK` に対してWindowsでは `//192.168.1.5/common/FY2024/09_MobileSensing`、Linuxでは `/media/hayashide/MobileSensing` を選ぶ。Git object `55f0717aba1d4eaf069618dae1c4ccf03c25069a`（2024-09-08）の同関数にもこの対応が存在する。これはルート対応のコード証拠であり、過去のOSマウント記録の証拠ではない。

| 旧絶対パスの相対部分 | SMB共有 `//192.168.1.5/common` 内の対応候補 | コード根拠 |
|---|---|---|
| `Nagasaki20241205193158/csv/annotation/Nagasaki20241205193158_annotation_ytpc2024j_20241205_193158.csv` | `FY2024/09_MobileSensing/`＋左記 | `H:scripts/tracking/get_position_features.py:25`、`annotator.py:163–164` のUNC/旧パス対照 |
| `Nagasaki20241205193158/jpg/diff/*.jpg` | `FY2024/09_MobileSensing/`＋左記 | `H:scripts/tracking/annotator.py:147–148` の対照 |
| `PullWheelchairObaachan/csv/annotation/PullWheelchairObaachan_annotation_ytnpc2021h_20240827_192540New_fullimagePath.csv` | `FY2024/09_MobileSensing/`＋左記 | `H:scripts/tracking/annotator.py:249–256`（main節入力名は未コミット版）、master_thesisのpreprocess入力 |
| `TeaChair/csv/cluster/TeaChair_cluster_20241102_180430.csv` | `FY2024/09_MobileSensing/`＋左記 | `H:scripts/tracking/P_postProcess02_singlefile.py:38–40` |
| `Nagasaki20240827/csv/raw` | `FY2024/09_MobileSensing/`＋左記 | `M:scripts_202511/5_1日の人流データをざっくり/1_1日の人流データをざっくり把握する.py:12` |

`M:scripts/visualize/visualize_video.py:284–287` にも、同一fullrgb_imagePathに対して旧ルートを前置する実行行と `//NASK/common/FY2024/09_MobileSensing` を前置するコメント行が併記される。共有ルートcommon全体を旧マウント先へ対応付ける根拠はない。NAS実物は本担当では未参照で、上表の存在・内容・同一性は未確認。

bag入力は別途ローカル/HDDを指す例があり、NASが必ずbag保管元だとはコードだけでは確定できない。`H:scripts/tracking/P_20241030_postProcess01_local.py:17–18` には、コメントの `ytpc2024j_2024-12-05-19-31-58_1787.bag` と実行設定の `ytnpc2021h_2024-08-27-19-06-53_51_short.bag` がcatkin_ws配下に置かれる。`get_rosbag_paths()` は `/media/hayashide/*HDD*/rosbag/*.bag` を探索し、ファイル名時刻と30分長の仮定で選択する (`analysisManagement.py:162–192`)。これらbagのNAS内相対パスは未特定。`Nagasaki20241205193158` とbag時刻名の一致は候補の絞込みに使えるが、一致だけで来歴を確定しない。

## 実装の存在と実使用の証拠

| 証拠 | 言えること | 言えないこと |
|---|---|---|
| handheld `31512c50b5567f74448851152a57b8e8b39fe94a`、2024-12-20、動画出力・切出しに関するcommit文とannotator等の変更 | 特定日付データの動画出力に取り組んだ履歴 | 対象bagのhash、全工程の完走、使用環境 |
| handheld `1cf7f87`、2024-12-17、イベントカメラ相当処理の確認を述べるcommit文 | 著者による動作確認の記述がある | 独立した実行ログによる裏付け、全データへの適用 |
| handheld `51d0f1542312386275049894045fe72481dc629b`、2025-01-15、reid bbox対応 | bbox付きreid処理の開発履歴 | master_thesisの最終使用CSVの生成commit |
| master_thesis `eca1261a2268758809d661ca0aeb040f89a0db30`、2025-01-16、対象試行名を含むcommit文とBLIP/YOLO/位置前処理の同時変更 | 外部annotationから修論特徴量を作る目的の開発履歴 | 当時実行したスクリプト・順番・引数・完走 |
| 設定内のbag名、CSV名、画像path | 入出力候補と開発者の意図 | ファイル存在・実際の選択・内容の同一性 |

従って「実装経路は復元できる」が「このbagをこのcommitで処理し、このpickleになった」は未確認である。収録・変換ログ、NAS上の中間生成物のヘッダ/時刻/作成情報、当時の起動設定と突き合わせる必要がある。

## 依存環境・コードに残るコマンド

handheldの `docker/Dockerfile:3,94–103,206,245` はUbuntu20.04/CUDA12.1/ZED4.0ベース、ROS Noetic、Python3、catkin、ultralytics、hdbscanを記述する。追跡はnumpy/pandas、OpenCV、tf/tf2、PCL、munkres、KF補助コードにも依存する。master_thesis旧前処理はPyTorch/transformersのBLIP2 (`Salesforce/blip2-flan-t5-xl`、local model優先)、ultralytics `yolo11x-pose.pt`、OpenCV等に依存する。handheldの検出ノードもファイル名yolov8とは異なり `yolo11x-seg.pt` を選ぶ (`yolov8_node.py:21–24`)。重みのhash・ライブラリの実インストール版・GPU/ドライバ・当時のコンテナimage IDは未確認で、Dockerfileを実行環境そのものの証拠とはしない。今回乱数やseedを使用していない。過去の推論seed設定も確定していない。

以下は実装上の起動例を仕様として示すだけで、今回実行していない。

```bash
# handheld側：P_20241030_postProcess01_local.py:37 のコマンド形式
roslaunch ytlab_handheld_sensoring_system_modules csvExporter_left.launch hostname:=/<hostname> trial_name:=<trial> bag_path:=<bag>

# master_thesis側：legacy前処理（main節に固定設定、入出力の上書きあり）
python3 master_thesis_modules/scripts/preprocess/preprocess_blip.py
python3 master_thesis_modules/scripts/preprocess/preprocess_yolo.py
python3 master_thesis_modules/scripts/preprocess/preprocess_others.py

# 現行の最終入力接続（CLI仕様のみ。評価は未実施）
python -m master_thesis_modules.real_data.runner.run_real_data_eval --input <data_dicts.pickleまたはCSVディレクトリ> --output <出力先> --common-dir <評価設定ディレクトリ>
```

これらを現HEADでそのまま再現可能と扱わない。特に以下は未解決である。

- `Manager.get_database_dir():100–103` はmobilesensing_dir_pathへ既にtrial_nameを付けるが、旧preprocess3本はさらに `/PullWheelchairObaachan` を付ける。mainのtrial_nameもBLIPとYOLO/othersで末尾2の有無が異なる。既定設定のパスが整合しない。
- BLIPは `...New_fullimagePath.csv`、YOLO/othersは `...New_fixposition.csv` を読み、後者の手動修正・生成コマンドは未特定。行番号で転記するため、両CSVと人物別特徴CSVの行順・長さ一致が必要で、timestamp joinはしていない。
- annotation補間はID 0〜6だけbfillする等の固定規則、画像bboxは処理により1/2への縮小有無が異なる。最終CSVに適用した規則と解像度は未確認。
- `csvExporter_left.launch` はbag再生と認識処理を含むが、カメラ補正画像・CameraInfo・LiDAR/TFがどこで生成/再生されるかは上流launchとの照合が必要。単にbag_pathだけ渡して全topicが揃うとは断定しない。
- `get_database_dir`、前処理、eventcamera等は書込を伴う。読取調査の確認コマンドとしては実行しない。

## 確認方法と残る接続点

1. 既存接続またはユーザー承認後の接続でNASの対象サブディレクトリだけを確認し、上記annotation/raw/reid/JPG候補とbagを列挙する。共有ルート対応を広げない。
2. bag metadataからtopic/型/記録時刻範囲を確認し、対応するTF・画像・点群の有無を調べる。今回bagを開いていないためmetadataは未確認。
3. 中間CSVのヘッダ・timestamp・ID集合・画像pathを読取照合する。bbox座標系、行数/行順、0とNaN、重複時刻、補間箇所を確認し、raw→reid→annotation→特徴CSV→pickleを結ぶ。
4. 実行ログやコンテナ情報とGit commitを対応付ける。コード中の固定入力名やcommit文は補助証拠にとどめる。
5. これらを経て再実行が必要なら、別途範囲を合意する。現時点でパス修正・補間変更・速度のdt除算・保存形式変更を実装しない。

## 調査コマンドと検証

使用した読取コマンドは `git status --short --branch`、`git rev-parse HEAD`、`git branch -avv`、`git for-each-ref`、`git ls-tree`、`git log --all`（対象path、`-G`検索を含む）、`git show`、`git diff`、`rg --files`、`rg -n`、`nl -ba`、`sed`、`cat`。外部reflogや全生成データの網羅調査はしていない。動作テスト・モデル起動・CSV変換・評価計算は未実施。成果は本Markdownのみで、危険度・順位アルゴリズム、コード、原データの変更はない。
