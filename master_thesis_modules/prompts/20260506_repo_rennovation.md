あなたは既存Python研究コードのリファクタリングと新規シミュレーション基盤設計を支援する開発エージェントです。

このリポジトリは、修士論文で作成した「病棟共有空間における空間的文脈を考慮したリスク優先順位モニタリングシステム」の実装です。

現状の正本は以下です。

- master_thesis_modules/scripts/master_v5.py
- master_thesis_modules/scripts/risk/schema.py
- master_thesis_modules/scripts/network/graph_manager_v3.py
- master_thesis_modules/scripts/fuzzy/fuzzy_reasoning_v5.py
- master_thesis_modules/scripts/AHP/get_comparison_mtx_v3.py

master_v5.py は、人物ごとの時系列特徴量を入力し、以下のような階層構造で総合危険度を計算します。

- 第5・6層: 入力特徴量・幾何情報
  - 例:
    - 50000000: 患者判別
    - 50000010: 年齢カテゴリ
    - 50000100-50000103: 姿勢特徴量
      - 立位度
      - 体幹傾き
      - 手首特徴
      - 足首特徴
    - 50001000系: 周辺物体座標
    - 50001100系: スタッフ座標・移動方向
    - 60010000, 60010001: 本人座標
    - 60010002: 対象者高さ最大値

- 第4層: 解釈可能な要因別リスク
  - 例:
    - 40000010: 立ち上がり動作リスク
    - 40000100: 点滴の近くにいるリスク
    - 40000101: 車椅子に乗っている/近いリスク
    - 40000102: 手すりから離れているリスク
    - 40000110: スタッフが近くにいないリスク
    - 40000111: スタッフが見ていないリスク

- 第3・2・1層: 上位リスク
  - 30000000: 内的・静的リスク
  - 30000001: 内的・動的リスク
  - 30000010: 外的・静的リスク
  - 30000011: 外的・動的リスク
  - 20000000: 内的リスク
  - 20000001: 外的リスク
  - 10000000: 総合危険度

このリポジトリでは、AHP、Fuzzy推論、単純重み和を組み合わせて、人物ごとの総合危険度と順位を算出します。

今回やりたいことは、既存コードをいきなり全面改修することではありません。
既存の master_v5.py は修論再現用の参照実装としてできるだけ壊さず、新しく読みやすい構造のリスク評価コアと、YAML/JSONシナリオから模擬データを生成して評価できるシミュレーション基盤を追加したいです。

目的は以下です。

1. 可読性を上げる
2. 責務ごとに細かくコードを分ける
3. シナリオをYAML/JSONで定義できるようにする
4. 画像・センサ由来の第5層以下特徴量と、シミュレーション由来の擬似特徴量を同じFeatureFrameに変換できるようにする
5. 最終的に、空間的文脈を考慮した場合としない場合で、危険度順位・通知対象・説明理由がどう変わるかを比較できるようにする

重要な設計方針:

- 既存の master_v5.py を直接大きく書き換えないこと
- まずは新規ディレクトリを追加して、既存コードとは分離すること
- 8桁ノード番号をコード中にベタ書きしないこと
- ノード番号は node_ids.py に集約すること
- 外部のシナリオ定義では、40000110 や 50000102 のようなノード番号ではなく、staff_distance_risk や wrist_distance_from_hip のような意味のある名前を使うこと
- 第5層以下は「FeatureFrameを作るための入力・前処理層」として扱うこと
- RiskEngineはFeatureFrameを受け取り、第4層以上のリスクを計算する責務にすること

まず、以下のような新規構成を提案・作成してください。

master_thesis_modules/
  risk_core/
    schema/
      node_ids.py
      node_labels.py
      evaluation_order.py

    features/
      position.py
      pose_features.py
      semantic_features.py
      spatial_features.py
      feature_frame.py
      legacy_node_converter.py

    factors/
      attribution_risk.py
      action_risk.py
      object_risk.py
      staff_risk.py

    aggregators/
      weighted_sum.py
      ahp.py
      fuzzy.py

    engine/
      risk_config.py
      risk_engine.py
      risk_result.py

    explanation/
      factor_extractor.py
      explanation_generator.py
      templates.py

    notification/
      notification_policy.py
      notification_result.py

  scenario_sim/
    scenarios/
      reach_object_context_demo.yaml
      staff_nearby_suppression_demo.yaml
      priority_reversal_by_context_demo.yaml

    domain/
      patient.py
      staff.py
      object_entity.py
      action.py
      world_state.py

    events/
      scenario_event.py
      event_engine.py

    encoder/
      scenario_loader.py
      pose_preset_encoder.py
      feature_encoder.py

    runner/
      run_scenario.py
      compare_models.py

    visualization/
      plot_risk_timeseries.py
      plot_ranking.py

  tests/
    risk_core/
    scenario_sim/

最初の実装範囲は、完全な既存コード移植ではなく、MVPで構いません。

MVPの範囲:

1. node_ids.py を作る
   - 主要ノード番号を定数化する
   - 例:
     - TOTAL_RISK = 10000000
     - INTERNAL_RISK = 20000000
     - EXTERNAL_RISK = 20000001
     - STANDING_RISK = 40000010
     - IV_POLE_RISK = 40000100
     - WHEELCHAIR_RISK = 40000101
     - HANDRAIL_DISTANCE_RISK = 40000102
     - STAFF_DISTANCE_RISK = 40000110
     - STAFF_NOT_WATCHING_RISK = 40000111

2. FeatureFrameを定義する
   - 人物1人・1時刻分の特徴量を保持するdataclassにする
   - 例:
     - person_id
     - time_s
     - is_patient_label
     - is_patient_confidence
     - age_group_label
     - age_confidence
     - pose_features
     - patient_position
     - nearest_iv_position
     - nearest_wheelchair_position
     - nearest_handrail_position
     - nearest_staff_position
     - nearest_staff_velocity

3. PoseFeaturesを定義する
   - standing_degree
   - trunk_tilt
   - wrist_distance_from_hip
   - ankle_spread

4. action_risk.py: 40000010番台の動作リスク計算を、修士論文の定式化に沿って実装してください。

対象ノード:

- 40000010: 立ち上がり動作リスク
- 40000011: 車椅子ブレーキ解除リスク
- 40000012: 車椅子を動かすリスク
- 40000013: バランスを崩すリスク
- 40000014: 手を挙げる/動かすリスク
- 40000015: せき込むリスク
- 40000016: 顔を触るリスク

入力特徴量:

- 50000100: 立位・座位の度合い
- 50000101: 体幹の傾き
- 50000102: 手首特徴
- 50000103: 足首特徴
- 60010002: 対象者高さ最大値。存在する場合は、40000010の立ち上がり動作リスク計算に優先的に使用する。

修士論文上の考え方:

- YOLO11-Pose等で得た姿勢・bbox情報から、人物 i の4次元姿勢特徴量 v_i(k) を作る。
- v_i(k) は以下の4成分を持つ。
  - standing_degree
  - trunk_tilt
  - wrist_distance_from_hip
  - ankle_spread
- 事前定義された危険動作ごとに、4次元の参照姿勢特徴量 v_ref_j を持つ。
- 各危険動作 j について、観測姿勢特徴量 v_i(k) と参照姿勢特徴量 v_ref_j の距離を求め、距離が近いほど高リスクとなる類似度に変換する。
- 既存 master_v5.py の risky_motion_dict と pose_similarity() を確認し、可能な限り既存挙動と整合させる。
- READMEでは、既存実装は4次元ベクトルに対する危険動作テンプレートとの差分平均から類似度を計算し、similarity ** 4 で強調していると整理されている。まずはこの挙動を再現する。
- ただし、40000010 については、60010002 が入力として存在する場合、既存 master_v5.py と同様に、対象者高さ最大値からシグモイド等で立ち上がり度を算出して上書きする。60010002 が存在しない場合は、通常の姿勢特徴量ベースの類似度、または 50000100 を使う。

実装方針:

- risk_core/factors/action_risk.py を作成する。
- 1ファイルに巨大な処理を書かず、以下のように小さく分ける。

推奨クラス・関数:

1. PoseFeatures
   - 既に risk_core/features/pose_features.py に定義する想定。
   - 以下のfloat値を持つ。
     - standing_degree
     - trunk_tilt
     - wrist_distance_from_hip
     - ankle_spread

2. RiskyMotionTemplate
   - action_name: str
   - node_id: int
   - reference_pose: PoseFeatures

3. ActionRiskCalculator
   - calculate(pose_features: PoseFeatures, height_max: float | None = None) -> dict[int, float]
   - 戻り値は、40000010〜40000016 をキー、0.0〜1.0 のリスク値を値に持つdict。

4. pose_distance()
   - 観測姿勢特徴量と参照姿勢特徴量の距離を計算する。
   - 既存 master_v5.py の実装を確認し、差分平均またはユークリッド距離のどちらが使われているかに合わせる。
   - README上は「テンプレートとの差分平均から類似度を計算」とあるため、まずは差分平均を優先する。

5. pose_similarity()
   - distanceを0.0〜1.0の類似度に変換する。
   - 例:
     similarity = 1.0 - clip(mean_abs_diff, 0.0, 1.0)
     risk = similarity ** 4
   - ただし、既存 master_v5.py に明示的な式がある場合はそちらを優先する。

6. standing_risk_from_height()
   - height_max: float から 40000010 の立ち上がり動作リスクを算出する。
   - 既存 master_v5.py にシグモイドの係数・閾値がある場合はそれを使う。
   - 既存実装から値が読み取れない場合は、暫定実装として設定可能なパラメータを risk_config.py に置く。
   - マジックナンバーを action_risk.py に直書きしない。

テンプレート定義:

- 事前定義危険動作の参照姿勢特徴量は、修士論文 Table 3.7 および既存 master_v5.py の risky_motion_dict を確認して設定する。
- 既存コードに risky_motion_dict がある場合は、まずそれを正とする。
- Table 3.7 と risky_motion_dict が一致しない場合は、既存 master_v5.py の値を優先し、差分をコメントまたはREADMEに記録する。
- テンプレート定義は action_risk.py に直書きしてもよいが、将来的に差し替えやすいように risk_core/factors/action_templates.py に分けることを推奨する。

推奨ファイル構成:

risk_core/
  factors/
    action_risk.py
    action_templates.py

action_templates.py には以下を定義する。

- STANDING
- RELEASING_BRAKES
- MOVING_WHEELCHAIR
- LOSING_BALANCE
- RAISING_HANDS
- COUGHING_UP
- TOUCHING_FACE

それぞれ、node_id と reference_pose を持つ。

注意点:

- 40000010〜40000016 はすべて「値が大きいほど危険」とする。
- 出力値は必ず 0.0〜1.0 にclipする。
- 40000010 のみ、60010002 が存在する場合の上書きロジックを持つ。
- シナリオYAMLでは 40000010 などのノード番号を直接書かない。
- シナリオYAMLでは、まず action.label や pose_features を書き、feature_encoder.py で PoseFeatures に変換する。
- action_risk.py は、シナリオ由来か実データ由来かを意識しない。PoseFeaturesを受け取って動作リスクを返す純粋な計算モジュールにする。
- 将来、YOLO-Poseの関節座標から PoseFeatures を作る処理は、action_risk.py ではなく pose_feature_calculator.py に置く。

最低限のテスト:

1. 観測姿勢が STANDING の reference_pose と完全一致する場合、40000010 が高くなること。
2. 観測姿勢が RELEASING_BRAKES の reference_pose と完全一致する場合、40000011 が高くなること。
3. 観測姿勢がどのテンプレートからも遠い場合、40000010〜40000016 が低くなること。
4. similarity ** 4 により、中程度の類似度が抑制されること。
5. height_max が与えられた場合、40000010 は通常のpose similarityではなく height_max ベースの値で上書きされること。
6. すべての出力が 0.0〜1.0 の範囲に収まること。
7. 既存 master_v5.py の pose_similarity() に対して、代表的な入力で同程度の出力になる回帰テストを可能なら追加すること。

この実装の目的は、40000010番台の動作リスクを単なるダミー重みではなく、修士論文で定義した「4次元姿勢特徴量と事前定義危険動作テンプレートの類似度」に基づく形で再実装することです。

5. RiskEngineを作る
   - FeatureFrameを受け取り、第4層リスクを計算する
   - 最初は簡易的な重み和または簡易Fuzzyで total_risk を出してよい
   - ただし、将来的に既存のAHP/Fuzzyに差し替えられるように、aggregatorを差し替え可能にする

6. Scenario YAMLを1つ作る
   - reach_object_context_demo.yaml
   - 内容:
     - A: 車椅子患者、床の物を拾おうとしている、スタッフ遠い
     - B: 軽症患者、机上のコップに手を伸ばしている、スタッフ近い
     - C: 車椅子患者、点滴あり、点滴付近で手を伸ばす、スタッフ視野外
   - 期待として C > A > B になりやすい設計にする

7. scenario_loader.py と feature_encoder.py を作る
   - YAMLを読み込む
   - WorldStateまたは直接FeatureFrameへ変換する
   - シナリオではノード番号を書かない
   - 内部でFeatureFrameに変換する

8. run_scenario.py を作る
   - コマンド例:
     python -m master_thesis_modules.scenario_sim.runner.run_scenario \
       --scenario master_thesis_modules/scenario_sim/scenarios/reach_object_context_demo.yaml
   - 出力:
     - 各患者の total_risk
     - リスク順位
     - 第4層リスク内訳
     - 簡単な説明文

9. explanation_generator.py を作る
   - total_riskが高い理由を、第4層リスクの上位要因からテンプレートで生成する
   - LLMは使わない
   - 例:
     「Cさんは、点滴付近で手を伸ばしており、スタッフによる見守りが弱いため危険度が高く評価されました。」

10. 最低限のpytestを追加する
   - object_risk:
     - 点滴・車椅子は近いほどリスクが高い
     - 手すりは遠いほどリスクが高い
   - staff_risk:
     - スタッフが遠いほどリスクが高い
     - スタッフが患者に向かうと見守り喪失リスクが低い
     - スタッフが患者から外れる方向に動くと見守り喪失リスクが高い
   - scenario:
     - reach_object_context_demo.yaml で C > A > B となることを確認する

実装上の注意:

- 既存のmaster_v5.pyを壊さないこと
- 既存の外部ストレージパス `/media/hayashide/MasterThesis` に依存しないこと
- 新規MVPはローカルのYAMLだけで動くようにすること
- 既存のAHP/Fuzzy実装を無理に最初から使わなくてよい
- ただし、後から差し替えられるように interface / class 構造を作ること
- dataclassと型ヒントを積極的に使うこと
- 1ファイルに巨大な処理を書かず、責務ごとに分けること
- コメントは、研究コードの読者が理解できる程度に入れること
- 既存ノード番号との対応は legacy_node_converter.py に集約すること
- ノード番号をシナリオYAMLに露出させないこと

最初にやってほしいこと:

1. リポジトリを読み、既存の構成と master_v5.py / schema.py の使われ方を確認する
2. 上記方針に基づき、新規ディレクトリとMVPファイル群を追加する
3. 既存コードは最小限しか触らない
4. 追加・変更したファイル一覧をまとめる
5. 実行方法とテスト方法をREADMEまたはコメントで示す
6. MVPの実行結果として、reach_object_context_demo.yaml の各患者リスク、順位、説明文が表示されるようにする

この作業の目的は、修論コードをすぐに完全に置き換えることではなく、今後「空間的文脈を考慮した危険度優先順位付け」を複数シナリオで検証・可視化するための、読みやすく拡張しやすい土台を作ることです。