# REQ-002 revision 2：rosbag側の読取調査

調査日：2026-10-15。担当：researcher（bag再生・抽出側）。対象はR-04〜06 / AC-05〜07のうちrosbag側である。[要求書](../requests/REQ-002-simplifyDataFormat.md)と[実施可否報告](2026-10-15-REQ-002-investigation-feasibility.md)を読み、ユーザーの直接指示を実行根拠とした。以前の保存形式案は実装未承認のままである。

## 結論と証拠の区別

調査対象の`ytlab_rosbag_modules`は、bagの選択、ROS topicの再生・記録、TFフレーム名の診断を担当する。画像・追跡CSV・特徴量をファイルとして切り出す実装は、確認した現行ファイル、ローカルrefのスクリプト一覧、関連履歴からは確認できなかった。下流の抽出処理とは区別する必要がある。

この結論はコードの実装仕様であり、特定bagにこの処理を実行した証拠ではない。固定bag名、commitメッセージ、ブランチの存在は手掛かりだが、実行記録・生成物・元bagを照合できていない。外部コード、NAS、原データを変更せず、再生・変換・build・Docker起動は実行していない。

## 対象と版

Gitの実際のルートは`/home/hayashide/ytlab_ros_ws/ytlab_rosbag`。主対象はその`ytlab_rosbag_modules`配下であり、接続環境を確認するため同じリポジトリの`docker/`、README、`.gitmodules`も読んだ。`~/.codex/AGENTS.md`を確認し、主対象までの親ディレクトリに別の`AGENTS.md`は見つからなかった。

| ローカルref | 調査時のcommit | 関連性・相違 |
|---|---|---|
| `hayashide`, `origin/hayashide` | `a98fceb135f77bce9a031ae91f37dbbbd2bfd6a3` | 現在のHEAD。TF診断あり。launchに未コミット変更あり |
| `master`, `origin/master`, `origin/HEAD` | `1deec671fe9382ab1b0aea64e71c86773f425887` | 基本再生・record recommender |
| `origin/devel` | `2e778f6a144f7c426649f98b4d9215bec94a945f` | 再生時`use_sim_time=false`, `--clock --pause -l`。compressed再配信あり |
| `origin/feature/nagasaki_people_sensing` | `16846e51a817fdd752efe01557b6a6929b6ddd9e` | Ouster等の記録、Rviz画面録画。再生は`--clock --pause -r 1.0` |
| `origin/feature/rosbag_record_minimum` | `2b6f81ffc5dd4f88a12800c26dc985801a0b9f64` | 記録topic除外の候補 |
| `origin/feature/rosbag_record_recommender` | `a2d4578909386a28786a8b9ddd58dc15041990df` | topic件数・メモリ量の診断 |
| `origin/ytjs2022a` | `5dd892682497a6eaf83ab6d6c4021b8f1fb9c7b8` | 再生default bagが2023-06-12 |
| `origin/ytnpc2021g` | `8057e7875b30747657d5d22717d8136cfbd5f8ef` | 再生倍率`-r 5` |

remote-tracking refはローカルに保存された参照であり、リモートの最新状態は未確認。fetch・checkoutは実行していない。上記refの`scripts/`、`shell_scripts/`、再生launchのblob一覧も比較した。

調査前後の`git status --porcelain=v1`は、既存の`M ytlab_rosbag_modules/launch/rosbag_play.launch`のみ。この差分はHEADの再生引数`--clock`に`--pause -l`を加え、`use_sim_time=false`を加えている。調査時のワークツリーを実際の過去実行版と扱っていない。

## bag選択・再生の仕様

以下の相対パスは、特記がなければ上記Gitルートを基準とする。行番号は調査時ワークツリーを示し、未コミット差分のないファイルはHEADと一致する。

| 処理 | 入力・出力 | 時刻・座標系・同期 | 根拠 |
|---|---|---|---|
| GUI選択 | `*.bag`を選択し絶対パス文字列をstdoutへ出す。初期ディレクトリ`/rosbag/` | bag内容は読まず、同期処理なし | `ytlab_rosbag_modules/scripts/get_rosbag.py:4–13` |
| shell接続 | GUIのstdoutを`rosbag_filename`に入れ、launchの`filename:=`へ渡す | リサンプリング・変換なし | `ytlab_rosbag_modules/shell_scripts/play_rosbag.sh:3–5` |
| HEADの再生 | `rosbag play <filename> --clock`を起動。topic選択・remap指定なし | `/clock`を送出する指定はあるが、HEADでは`use_sim_time`設定なし。座標変換なし | `a98fceb:ytlab_rosbag_modules/launch/rosbag_play.launch` |
| ワークツリーの再生 | HEADに`--pause -l`、`use_sim_time=false`追加 | 一時停止開始・反復の設定。`--clock`と購読側のsim time利用は別設定 | 同launch `:2–5`と`git diff` |
| TF診断 | 固定bagの`/tf`を読み、`child_frame_id`に`hmn`を含むtransformの親・子フレーム名をprint | `bag.read_messages`の`t`は使わず、`transform.header.stamp`も処理しない。座標値は出力しない | `ytlab_rosbag_modules/scripts/checkrosbagtf.py:5–25` |

TF診断は`tf2_msgs.msg.TFMessage`をimportするが、実bagの接続型・定義を検証するコードではない。型名は実装上の想定であり、実データの型を確認した事実ではない。`/tf_static`はこの診断の読取対象ではない。座標系の具体名、軸方向、単位、world/map/odomとの接続は実bag未確認のため不明。`hmn`という文字列だけから人物ID・座標定義は確定しない。

## 記録topicと生成物

`ytlab_rosbag_modules/launch/rosbag_record_for_hsrd.launch:2–4`は次の正規表現でbagへの記録対象を選ぶ。これはbagにこれらが実在した証拠ではない。

| 指定topic | 型の根拠 | 時刻・座標系・生成物 |
|---|---|---|
| `/tf.*` | launchには型指定なし。TF診断は`TFMessage`を想定 | 具体的なframeとstampは未確認 |
| `/hsrb/odom` | launchには型指定なし | frame、単位、odomの出自は未確認 |
| `/hsrb/joint_states` | 同上 | stamp・関節配列は未確認 |
| `/zed/zed_node/depth/depth_registered` | 同上 | 画像encoding、深度単位、frameは未確認 |
| `/zed/zed_node/rgb/camera_info` | 同上 | 校正値・frameは未確認 |
| `/zed/zed_node/rgb/image_rect_color` | 同上 | 画像encoding、stamp、frameは未確認 |
| `/teleop_scenario/merged_scan` | 同上 | 座標と型は未確認 |

`compressed`と`theora`を除外し、出力prefixは`/catkin_ws/src/rosbag/`、`--repeat-latched`指定。`cc7ca5b21fed06a14bf9c52d35b8fc19baf0a2d5`（2023-12-09、`include tf_static`）で`/tf`から`/tf.*`へ変更されている。メッセージ名と型の慣例だけで型を断定しない。実際のbagを読める場合、接続情報とメッセージheaderを調べて確定する。

同じlaunchの`:7–8`にはOctober experimentというコメント付きの旧topic候補（`/hsrb/wheel_odom`、`/hsrb/imu_odom`、`/hsrb/laser_odom`、`/stereo/depth`、`/hsrb/zed2_stereo/left.*`等）がある。コメントは当時の使用証明ではなく、現行の有効な指定でもない。

他のrecord launchは`record -a`と除外正規表現の組合せで、hsrd/hsrb/zed/stereo/realsense/whill等の候補を持つ。`rosbag_record_all.launch:3`は`record -a -o /rosbag/ --repeat-latched`である。どのlaunchでどのbagを記録したかは不明。

## 別refと履歴の補足

- `origin/devel:ytlab_rosbag_modules/launch/rosbag_play_compressed.launch`は`rosbag play --clock`に`image_transport republish compressed raw`を組み合わせ、`in=/usb_cam/image_raw`、`out=/image_raw`へremapする。ファイルへの画像抽出処理ではない。`play_rosbag_kodac.sh`からこのlaunchを呼ぶ。
- `origin/devel:ytlab_rosbag_modules/scripts/rosbag_record_recommender.py`は選択bagの開始〜終了を1秒ずつ走査し、topicの件数と`pympler.asizeof`によるメモリ推定量をprintする。出力ファイルや特徴量は作成しない。メモリ推定値をbag内の圧縮後サイズと扱えず、区間境界の重複集計等の可能性もあるため、実測容量の証拠としない。
- `origin/feature/nagasaki_people_sensing:ytlab_rosbag_modules/shell_scripts/record_nagasaki_rosbags.sh`はRviz画面をffmpegのX11キャプチャでmp4へ保存する。固定入力は`/home/ytpc2022e/decwest_workspace/ytlab_rosbag/rosbag/1113`。特徴量抽出との接続は確認できない。
- 同refの`ytlab_rosbag_modules/scripts/ouster_info_publisher.py`は`/ouster/get_metadata`の`ouster_ros.srv.GetMetadata`を呼び、`/ouster/metadata`へ`std_msgs/String`を0.5秒Timerでpublishする。今回の下流入力との対応は未確認。
- `c8cdba46f0df422270040f4f95142133444ba01a`のcommit名は`extract_PCD_from_record`だが、差分はZED記録launchの除外topic追加だけである。PCD抽出実装がある証拠にはならない。
- `.gitmodules`に`third_party/rosbag_editor`の記述が残るが、HEADのtreeでは当該gitlinkはない。`a98fceb`で削除されており、submoduleの存在・過去の実行は確定しない。取得や起動は行っていない。

## 依存環境とコード上の実行コマンド

`docker/Dockerfile:3,98–109`はUbuntu 20.04、ROS Noetic desktop-full、Python 3系を構成する。`ytlab_rosbag_modules/package.xml`の明示依存はcatkin中心で、実際に使うrosbag/tf2_msgs/tkinter等を網羅していない。GUI選択にはTkとX表示、TF診断には`rosbag`と`tf2_msgs`、別refの診断には`pympler`が必要。環境が現ホストに実在するか・当時のインストール版は未確認。

`docker/ros_entrypoint.sh:5`はROSのsetupをsourceし、`docker/global_ros_setting.sh:4,55–57,69`は`/catkin_ws/devel/setup.bash`、ローカルROS master設定、`catkin_make`のaliasを定義する。ROS masterの実際の稼働状況・過去の環境変数は不明。

コードが示すコマンド例（今回未実行）：

```bash
# ROS環境とcatkin packageが利用できる環境でGUI選択から再生
bash /catkin_ws/src/ytlab_rosbag_modules/shell_scripts/play_rosbag.sh

# GUIを経由しない、launchの入力インタフェース
roslaunch ytlab_rosbag_modules rosbag_play.launch filename:=<bagの絶対パス>

# 固定bagを開く診断。任意bagを指定するCLIではない
python3 /catkin_ws/src/ytlab_rosbag_modules/scripts/checkrosbagtf.py
```

これらは実行成功・過去使用を保証する再現手順ではない。Docker起動やbuildは環境・書込を伴うため今回実施していない。READMEは`./build`、`./run`と記載する一方、実ファイルは`build.sh`、`run.sh`であり、recordのpackage/launch名にも現行ファイルとの差がある。README単独で再現可能とは判定しない。

## NAS・絶対パス・bag名の対応

追加ユーザー情報は「過去のマウント先に関する記憶」として扱う。

```text
\\192.168.1.5\common\FY2024\09_MobileSensing
これが/media/hayashide/MobileSensing にかつてマウントされていたはず。
```

候補規則は旧`/media/hayashide/MobileSensing/<rel>` → SMB共有`//192.168.1.5/common`内の`FY2024/09_MobileSensing/<rel>`である。common全体を旧マウント先へ対応付けない。このリポジトリのワークツリーと到達可能な履歴の検索では、`MobileSensing`、`FY2024`、`09_MobileSensing`、`192.168.1.5/common`の具体的一致を確認できなかった。そのため、この担当範囲だけでは`<rel>`を確定できない。

| コード上のパス・名前 | 証拠 | NAS対応の判定 |
|---|---|---|
| `/rosbag/_2022-05-05-17-42-27.bag` | HEAD再生launchのdefault | サンプルdefault。NAS上の所在・使用は未確認 |
| `/rosbag/_2023-06-12-15-05-55.bag` | `origin/ytjs2022a`再生launch | 別refのdefault。NAS上の所在・使用は未確認 |
| `/catkin_ws/src/rosbag/_2023-12-02-21-52-57.bag` | `scripts/checkrosbagtf.py:5`、追加commit`ad9bd9de9193cd65e76f585be230d73105ea0042` | 固定診断対象。実bagの存在・NAS上の相対パス未確認 |
| `/media/ytpc2020d/Data02/FromUbuntu/rosbag/` | `origin/devel:rosbag/rosbag`、`origin/feature/nagasaki_people_sensing:rosbag/rosbag`のsymlink内容 | 別ユーザー・別mediaパス。MobileSensingへ同一視しない |

`docker/run.sh:35–42`はhostのrepo内`rosbag`をcontainerの`/catkin_ws/src/rosbag`へ、hostの`/media`をcontainerの`/media`へbindする。このため、host旧NASパスが存在すれば同じ`/media/...`でcontainerから見える構成ではあるが、これはNAS接続や実使用の証拠ではない。`/rosbag`のbindは現行run.shにないので、GUI初期ディレクトリ・launch default・多くのrecord出力先と不一致である。`8c4f4c7`の差分に`/rosbag`から`/catkin_ws/src/rosbag`へのbind変更が残る。

repo内`rosbag/`の非再帰的な一覧では`.gitignore`のみで、bagファイルは確認できなかった。NASをこの担当では接続・列挙していない。実物照合の確認状況はPM側の接続確認を参照する。

## 未確認事項と確認方法

| 未確認事項 | 追加確認方法（今回未実行） |
|---|---|
| 実bagのNAS相対パス | 許可された既存接続で`FY2024/09_MobileSensing`配下のbag名を照合。新規接続が必要ならPMから接続方法を提示し確認 |
| bagのtopic・型・収録期間・message数 | 読取可能なbagの`rosbag info`等と接続情報を確認。未知のbag名をdefaultと同一視しない |
| 時刻と座標系 | bagの記録時刻`t`、`header.stamp`、`frame_id`、TF親子とstampを下流が読む時刻・frameと対照 |
| 実際の再生・抽出版 | 実行ログ、保存設定、生成物の来歴、bag識別子とcommitの記録を照合。現在のbranchだけで判定しない |
| 追跡・特徴量への実接続 | handheld側のbag直読／topic購読実装と生成ファイル契約を照合。実bag由来の生成物で使用実績を確認 |
| Python/Tk/ROS/Dockerの当時版と使用環境 | 当時のimage digest・環境記録・依存lockがあるか確認。Dockerfileは構成意図として扱う |

## 調査方法・検証・限界

実施した読取は`git status --short --branch`、`git rev-parse`、`git branch -av`、`git for-each-ref`、`git ls-tree`、`git log --all`、`git show`、`git diff`、`rg --files`、`rg -n`、`nl -ba`、非再帰的`ls`である。履歴について`git log --all -p -G '\.bag'`、`git log --all -S '<NAS識別文字列>'`、処理名の履歴検索を実施し、ローカルrefに存在するスクリプト・shell・再生launchのblobを比較した。seedは使用していない。

NAS識別文字列の履歴検索は`MobileSensing`、`192.168.1.5/common`、`FY2024`、`09_MobileSensing`。このリポジトリから参照されない別リポジトリ・削除済み未到達object・shell履歴・ユーザー認証設定は探索していない。コードに記録のない手動操作や外部ツールの存在を否定する調査ではない。

外部リポジトリへの書込、ブランチ切替、fetch、データ変換、環境変更、commit、pushは未実施。報告書のみをmaster_thesisのreports配下へ保存した。調査は実装仕様の静的確認であり、実データによる動作検証は行っていない。
