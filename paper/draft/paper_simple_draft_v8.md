# タイトル

病棟共有空間における空間的文脈とスタッフの判断知識を統合した危険度評価・見守り優先順位付け

英語タイトル案：  
**Context-Integrated Risk Evaluation and Monitoring Prioritization Incorporating Staff-specific Judgment Knowledge in Shared Hospital Spaces**

# 投稿先

第一候補：IEEE Access  
上振れ候補：IEEE Transactions on Systems, Man, and Cybernetics

# 論文の位置づけ

本研究は、病棟共有空間における短時間の見守り優先度を対象として、患者本人の属性・行動からなる患者文脈と、周囲物体・施設構造・スタッフとの相対位置からなる空間的文脈を統合し、患者ごとの時系列危険度と患者間順位を算出するフレームワークを提案する。

さらに、スタッフへのアンケートから得た判断傾向をAHP重みとFuzzyルールとしてモデルへ反映し、同一の入力時系列に対する危険度波形・患者順位の違いを分析する。

通知生成は、危険度時系列と内部要因を実際の見守り支援へ接続する補助機能として扱う。

# Research Questions

* **RQ1：** 空間的文脈を考慮することで、患者の時系列危険度および患者間の見守り優先順位はどのように変化するか。
* **RQ2：** スタッフごとの判断知識を危険度評価モデルに反映することで、同一入力に対する時系列危険度および患者間の見守り優先順位はどのように変化するか。

# 1. Introduction

病棟共有空間では、スタッフが複数患者の状態や行動を並行して把握し、状況に応じて対応対象を選択する必要がある。患者見守り研究では、転倒、離床、姿勢、活動等の検知が広く研究されてきた \cite{Mubashir2013SurveyFalls,Gutierrez2021VisionFallReview,Balaguera2017MedicalIoT}。

一方、見守り上の優先度は患者本人の状態や行動だけでは決まらない。同じ立ち上がり動作であっても、近くにスタッフや手すりがある場合と、支援者が近くにいない場合では、優先して確認すべき程度が異なり得る。Figure 1に、この問題設定の例を示す。

![Example motivating the need for spatial context in monitoring prioritization.](./paper_v8_figures/fig1_problem_setting.png)

**Figure 1.** 同じ立ち上がり動作でも、スタッフ配置や支援物等の周囲状況によって見守り優先度が変化し得る例。  
*Draft note: source slide p.2. Final figureでは “notification priority” を “monitoring priority” に変更し、必要に応じて模式図として描き直す。*

空間的位置や環境情報を危険度評価へ用いる研究は既に存在する。病院内ではベッド周辺の患者位置に基づく転倒リスク推定が行われており \cite{Isomoto2018FallRisk}、Ambient Assisted Livingでは本人情報と環境情報を統合したcontext-aware risk assessmentが検討されている \cite{Koshmak2014ContextFallRisk}。ただし、これらは主に単一患者や局所的な転倒リスクを対象としている。一方、病院全体では電子カルテやバイタル等から患者状態を連続的に評価し、高リスク患者を優先する研究が行われている \cite{Rothman2013ContinuousCondition,Escobar2020Deterioration}。

本研究が対象とするのは、これらの中間に位置する**病棟共有空間の短時間見守り優先度**である。すなわち、同じ空間にいる複数患者について、患者本人の状態・行動だけでなく、周囲物体、施設構造、スタッフとの相対関係を含めて危険度を更新し、その時点での見守り順位を算出する。

さらに、このような危険度評価では、各因子の重要度や複数条件の組み合わせ方を設定する必要がある。医療意思決定支援では、専門家が各因子へ与える重要度や条件間の関係をモデルへ取り込む研究が行われている \cite{Boulet2020ExpertWeights,TenBroeke2021BAIT}。本研究では、一つの固定した判断基準を置くのではなく、スタッフごとの回答を明示的な重み・ルールとして保持し、その違いが危険度時系列と患者順位へどう反映されるかを分析する。

以上を踏まえ、本研究では、患者文脈と空間的文脈を階層的に統合し、スタッフごとの判断知識をモデルパラメータとして反映する危険度評価フレームワークを構築する。評価では、空間的文脈の有無とスタッフ別パラメータの違いが、危険度時系列と患者間順位へ与える影響を検証する。

本研究の主な貢献は以下の2点である。

1. **Context-integrated risk evaluation:** 患者属性・行動、周辺物体・施設構造、スタッフとの相対関係を階層的に統合し、時系列危険度と患者間の見守り優先順位へ変換するフレームワークを構築した。
2. **Staff-specific parameterization and evaluation:** スタッフの一対比較および状況判断をAHP重みとFuzzyルールへ変換し、同一入力に対するスタッフ別モデルの出力差を危険度波形と順位から分析する。

# 2. Related Work

## 2.1 Patient Monitoring and Context-aware Risk Assessment

病院・高齢者ケア環境では、ウェアラブルセンサ、環境センサ、RGB・赤外・深度カメラ等を用いた転倒検知、活動認識、離床検知が広く検討されてきた \cite{Mubashir2013SurveyFalls,Igual2013FallDetectionChallenges,Gutierrez2021VisionFallReview}。病院内でも、ベッド離脱や転倒につながり得る動作を検知してスタッフへ通知するシステムが実装されている \cite{Balaguera2017MedicalIoT,Jones2020VideoMonitoring}。

周囲環境を扱う研究も存在する。ベッド・床等の施設構造を患者監視に利用する方法 \cite{Inoue2018BedDetection,Komagata2019BedMonitoring} や、ベッド周辺の患者位置を転倒リスクへ対応づける方法 \cite{Isomoto2018FallRisk} が提案されている。また、Ambient Assisted Livingでは、本人情報と環境センサを統合して文脈依存の危険を推定する研究が行われている \cite{SaldanaJimenez2009ContextRisk,Koshmak2014ContextFallRisk}。

これらに対し、本研究は病棟共有空間を対象とし、患者属性・行動、周辺物体・施設構造、スタッフとの相対位置を同一モデルで扱い、複数患者の時系列危険度と見守り順位を算出する。

## 2.2 Risk Prioritization and Expert-informed Decision Models

病院では、電子カルテ、バイタル、検査値等から患者状態を連続的にスコア化し、高リスク患者の確認や介入を支援する研究が行われている \cite{Rothman2013ContinuousCondition,OBrien2020OptimizedEWS,Escobar2020Deterioration}。本研究が対象とするのは生理的悪化ではなく、共有空間内の患者行動と周囲状況から短時間の見守り優先度を算出する問題である。

専門家知識を意思決定モデルへ反映する方法として、医師が各臨床因子へ与える重要度を数値化する方法 \cite{Boulet2020ExpertWeights} や、仮想症例への選択から因子重みと条件間相互作用を推定する方法 \cite{TenBroeke2021BAIT} が提案されている。AHPとFuzzy推論も、多基準意思決定や医療リスク評価に用いられている \cite{Saaty1990AHP,Zadeh1965FuzzySets,Mamdani1975LinguisticControl,Uzoka2011FuzzyAHPClinical,Chang2022FuzzyAHPRisk}。

本研究では、スタッフごとの回答をAHP重みとFuzzyルールへ変換し、同一のセンシング時系列に適用したときの危険度波形と患者順位の差を分析する。

## 2.3 Position of This Study

既存研究では、空間的位置を用いた転倒リスク、context-aware risk assessment、患者の連続リスクと優先化、専門家知識のパラメータ化がそれぞれ検討されている。一方、本調査で確認した範囲では、病棟共有空間において、患者・物体・施設・スタッフの文脈を統合し、スタッフごとの判断知識を反映した時系列危険度と患者間順位を分析する研究は限定的である。

本研究は、この統合モデルと、その出力挙動の評価に焦点を当てる。

# 3. Proposed Method

## 3.1 Overview

本研究では、患者文脈と空間的文脈から患者ごとの危険度を算出し、複数患者間の見守り優先順位を決定する。

スタッフの判断知識は、以下の2種類のパラメータとしてモデルへ反映する。

* 同一カテゴリ内の特徴量の重要度：AHP重み。
* 複数のリスク因子の組み合わせと危険度の関係：Fuzzyルール。

Figure 2は、同一の状況であっても、患者文脈と空間的文脈のどちらを重視するかによって、階層モデル内での危険度の寄与が変化する概念例を示す。

![Conceptual examples of context- and staff-dependent risk propagation.](./paper_v8_figures/fig2_context_staff_concept_2x2.png)

**Figure 2.** 空間状況とスタッフの判断傾向に応じた危険度評価の概念例。  
*Draft note: source slides pp.3-6. Final figureでは4枚のスライドをそのまま並べず、2×2の模式図として再作図する。*

Figure 3に提案システムの全体構成を示す。患者属性・行動、周辺物体、周辺スタッフに関する特徴量を取得し、階層的に危険度へ統合する。スタッフごとの重み・ルールは危険度評価部へ入力される。

![Overview of the proposed risk evaluation system.](./paper_v8_figures/fig3_system_overview.png)

**Figure 3.** 提案システムの全体構成。  
*Draft note: source slide p.7. Final figureでは左下のスタッフ入力を “Questionnaire → AHP weights / Fuzzy rules” と明示し、Notification generatorは補助出力として小さくする。*

## 3.2 Feature Selection and Classification

### 3.2.1 Feature selection

特徴量候補は、対象病院の病棟デイルームにおいて熟練看護師2名へ実施した約30分の聞き取りを基に整理した。見守り時に着目する情報、危険と判断する状況、複数患者の優先順位を中心に質問し、探索的な特徴量選定に用いた。

### 3.2.2 Feature classification

危険度評価に用いる特徴量を、内外・動静の2軸で4群に分類する。

| 分類 | 内容 | 例 |
|---|---|---|
| 内的・静的要因 | 患者本人の属性 | 年齢層、重症度など |
| 内的・動的要因 | 患者本人の行動 | 立ち上がり、姿勢崩れ、顔を触る等 |
| 外的・静的要因 | 周囲物体・施設構造 | 車椅子、点滴棒、手すりとの距離等 |
| 外的・動的要因 | スタッフ配置・見守り状況 | スタッフとの距離、相対方向、視野等 |

患者文脈は主に内的要因、空間的文脈は主に外的要因としてモデルへ入力する。

### 3.2.3 Feature extraction

* 内的・静的要因：人物画像や事前情報から患者属性を取得する。
* 内的・動的要因：姿勢推定結果から危険動作との類似度を算出する。
* 外的・静的要因：周辺物体の有無や施設構造との距離を算出する。
* 外的・動的要因：スタッフとの相対距離・方向等を算出する。

実装では、RGB画像、点群、位置情報、姿勢推定、VQA、施設地図等を組み合わせて特徴量を取得する。

## 3.3 Questionnaire-based Parameterization

### 3.3.1 Questionnaire design

スタッフごとの判断知識を反映するため、以下の回答を取得する。

* 同一カテゴリ内の特徴量間の重要度比較。
* 患者文脈と空間的文脈の重視度。
* 複数状況に対する危険度または優先順位判断。

一対比較回答からAHP重みを算出し、状況判断からFuzzyルールおよびメンバーシップ関数を設定する。

### 3.3.2 AHP-based weighting

同一カテゴリ内の特徴量に対する一対比較行列を

$$
A=[a_{ij}], \quad a_{ji}=\frac{1}{a_{ij}}, \quad a_{ii}=1
$$

とする。最大固有値を$\lambda_{\max}$、対応する固有ベクトルを$\mathbf{w}$とすると、

$$
A\mathbf{w}=\lambda_{\max}\mathbf{w}
$$

を満たす。正規化した$\mathbf{w}$を特徴量重みとして用いる \cite{Saaty1990AHP}。

回答の整合性はConsistency Index（CI）またはConsistency Ratio（CR）で確認する。

$$
CI=\frac{\lambda_{\max}-n}{n-1}
$$

## 3.4 Hierarchical Risk Evaluation

患者$i$の時刻$k$における総合危険度を$R_i(k)$とする。

まず、内的・静的、内的・動的、外的・静的、外的・動的の各リスクを算出する。

$$
R_i^{IS}(k)=f_{IS}(\mathbf{x}_i^{IS}(k))
$$

$$
R_i^{ID}(k)=\sum_j w_j^{ID}x_{ij}^{ID}(k)
$$

$$
R_i^{ES}(k)=\sum_j w_j^{ES}x_{ij}^{ES}(k)
$$

$$
R_i^{ED}(k)=F_{ED}(\mathbf{x}_i^{ED}(k))
$$

ここで、$IS$、$ID$、$ES$、$ED$はそれぞれ内的・静的、内的・動的、外的・静的、外的・動的要因を表す。

内的リスクと外的リスクを以下のように統合する。

$$
R_i^{I}(k)=\alpha R_i^{IS}(k)+(1-\alpha)R_i^{ID}(k)
$$

$$
R_i^{E}(k)=F_E(R_i^{ES}(k),R_i^{ED}(k))
$$

最終的な危険度は、

$$
R_i(k)=F_T(R_i^{I}(k),R_i^{E}(k))
$$

とする。

$F_{ED}$、$F_E$、$F_T$にはFuzzy推論を用い、異なるリスク因子間の条件依存関係を表現する \cite{Zadeh1965FuzzySets,Mamdani1975LinguisticControl}。

スタッフごとにAHP重み、$\alpha$、Fuzzyルール等を設定することで、同一入力に対するスタッフ別の危険度時系列を生成する。

## 3.5 Prioritization and Traceability

各時刻で$R_i(k)$を患者間で比較し、見守り優先順位を決定する。

また、階層構造内の特徴量と中間危険度を保持することで、危険度変化に寄与した要因を参照できる。必要に応じて、順位変化や危険度上昇をトリガとして通知を生成し、主要因を通知理由として付加する。

![Illustrative example of extracting factors contributing to a risk change and generating a notification.](./paper_v8_figures/fig9_notification_example.png)

**Figure 4.** 危険度変化と内部要因から通知理由を生成する補助例。  
*Draft note: source slide p.21. Main contributionではなく、Method末尾またはSupplementaryの例として扱う。*

# 4. Experiments

## 4.1 Evaluation Design

評価はRQ1とRQ2に対応する2つの実験から構成する。

* **Evaluation 1 / RQ1:** 空間的文脈の有無による危険度時系列と患者順位の変化を比較する。
* **Evaluation 2 / RQ2:** スタッフごとの判断知識による危険度時系列と患者順位の変化を比較する。

RQ2では、対照的な判断傾向を持つ2名のケーススタディと、全回答者（n=10）の傾向分析を行う。

## 4.2 Evaluation 1: Effect of Spatial Context

患者本人の行動条件を揃え、スタッフ配置、支援物、危険物等の空間的文脈を変化させたシナリオを用いる。Figure 5に立ち上がり動作を用いた代表シナリオを示す。

![Simulation scenario for evaluating the effect of spatial context.](./paper_v8_figures/fig4_rq1_standup_scenario.png)

**Figure 5.** RQ1で用いる立ち上がりシナリオ。  
*Draft note: source slide p.8. Final figureでは代表4時刻程度に削減し、患者・スタッフ・支援物の位置変化を読みやすくする。*

比較条件は以下とする。

* **Patient context only:** 患者文脈のみ。
* **Spatial context included:** 患者文脈＋空間的文脈。

Figure 6に、両条件における危険度時系列の比較例を示す。

![Risk time-series comparison with and without spatial context.](./paper_v8_figures/fig5_rq1_standup_result.png)

**Figure 6.** 空間的文脈の有無による危険度時系列の比較。  
*Draft note: source slide p.9. Final figureでは同一profileの2条件をoverlayし、RQ1の比較が直接読める形式へ再設計する。*

主な評価項目は以下とする。

* 空間的文脈変化前後の危険度差。
* 最大危険度・危険度変動幅。
* 2条件で順位が異なる時間割合。
* 患者間の順位逆転。

## 4.3 Evaluation 2: Reflection of Staff-specific Knowledge

### 4.3.1 Case study of two contrasting staff members

病棟共有空間で取得した同一の実データに、異なるスタッフのアンケート回答に基づくモデルを適用する。判断傾向が対照的な2名を選定し、AHP重み・Fuzzyルールと危険度出力の対応を確認する。

Figure 7に、実データへ複数スタッフモデルを適用した危険度波形の例を示す。

![Risk time series obtained from staff-specific models for real-world data.](./paper_v8_figures/fig6_rq2_case_study.png)

**Figure 7.** 実データに対するスタッフ別危険度波形の例。  
*Draft note: source slide p.16. Final figureでは対照的な2名を強調し、他回答者は薄色または非表示とする。イベント・空間条件変化の時刻も併記する。*

比較項目は以下とする。

* AHP重みおよびFuzzyルールの特徴。
* 危険度波形のピーク・変動幅・上昇タイミング。
* 患者順位および順位逆転。
* 危険度上昇に寄与した中間リスク因子。

### 4.3.2 Group-level analysis

全回答者（n=10）について、アンケート回答から得た判断傾向と危険度出力の関係を分析する。

現時点の可視化候補をFigures 8および9に示す。

![Relationship between spatial-context orientation and model output.](./paper_v8_figures/fig7_rq2_group_context.png)

**Figure 8.** 空間的文脈の重視傾向と危険度出力の関係を示す現行可視化。  
*Draft note: source slide p.17. Final figureでは、x軸をスタッフごとの空間文脈重視指標、y軸を空間文脈変化に対する$\Delta$Risk等とした散布図へ変更する。*

![Relationship between the object-vs-monitoring preference index and model output.](./paper_v8_figures/fig8_rq2_group_di.png)

**Figure 9.** 物体要因と見守り要因の重視傾向と危険度出力の関係を示す現行可視化。  
*Draft note: source slide p.18. Figure 8と同様、スタッフごとの指標と対応する出力差を直接比較する散布図へ変更する。*

分析候補は以下とする。

* 空間的文脈の重視度と、空間条件変化時の危険度変化量。
* 患者行動の重視度と、危険行動発生時の危険度ピーク。
* 特徴量重みと、その特徴量が変化した区間の危険度変化量。

n=10であるため、散布図、順位相関、回答と出力の対応傾向を中心に確認する。AHP回答についてはCIまたはCRを算出し、回答の整合性を確認する。

# 5. Discussion

## 5.1 Effect of Spatial Context

RQ1では、患者行動が類似していても、スタッフ配置や周辺物体等の空間条件によって危険度と患者順位が変化するかを確認する。

特に患者間順位の変化は、空間的文脈を含めることが「誰を優先して見るか」という見守り支援の出力に与える影響として解釈する。

## 5.2 Staff-specific Risk Evaluation

RQ2では、スタッフごとのアンケート回答が、AHP重み・Fuzzyルールを介して危険度波形と患者順位へどのように反映されるかを分析する。

ケーススタディでは個々の出力差の形成過程を示し、n=10の分析では回答傾向と出力特徴の対応を確認する。

## 5.3 Limitations

* 危険度の臨床的な正解ラベルに対する精度検証は行っていない。
* n=10の分析は探索的であり、スタッフ一般への統計的な一般化には限界がある。
* 実データにはセンサ・認識ノイズが含まれる。
* 特徴量は限られた現場観察と熟練看護師2名への聞き取りを基に選定している。
* スタッフ別モデルの望ましさ、通知理由の理解度、業務負荷、Alarm fatigue、事故防止効果は評価していない。

## 5.4 Future Work

* スタッフ自身による危険度・優先順位評価と提案手法の出力を比較する。
* 空間的文脈なし、平均モデル、スタッフ別モデル等の比較により各構成要素の寄与を評価する。
* より多くのスタッフ・患者状況を用いて、判断傾向と危険度出力の関係を検証する。
* 通知理由の理解度・有用性を独立したユーザー評価として検証する。

# 6. Conclusion

本研究では、病棟共有空間における見守り支援に向けて、患者文脈、空間的文脈、スタッフごとの判断知識を統合する危険度評価フレームワークを提案する。

患者属性・行動、周辺物体・施設構造、スタッフ配置を階層的に統合し、患者ごとの時系列危険度と患者間順位を算出する。また、スタッフへのアンケートからAHP重みとFuzzyルールを設定し、同一入力に対するスタッフ別モデルの出力差を評価する。

RQ1では空間的文脈による危険度・順位の変化を、RQ2ではスタッフごとの判断知識による出力差を分析する。これにより、病棟共有空間における見守り優先度を、状況とスタッフ判断の双方に応じて表現するための基礎的枠組みを示す。

# Appendix: Draft-only Figure Management Notes

以下は本文に直接入れず、最終図の再設計またはSupplementary候補として保持する。

## A. Additional RQ1 scenario

顔を触る動作を用いたシナリオと結果（source slides pp.10-11）は、立ち上がり以外の動作でも同様の傾向が得られることを示す補助実験として利用できる。

![Additional RQ1 scenario using face-touching behavior.](./paper_v8_figures/figS1_touchface_scenario.png)

![Additional RQ1 result using face-touching behavior.](./paper_v8_figures/figS2_touchface_result.png)

## B. Four-patient simulation candidates

Source slides pp.12-15は、スタッフ判断パラメータと患者順位の関係を統制条件で示す素材として利用可能。ただし、現状の多数の水平線グラフは読みにくいため、Main textへ入れる場合はheatmapまたは順位表へ再設計する。

## C. Implementation detail

Source slide p.20は、VQA、Pose similarity evaluator、relative position calculator等の実装詳細を示す。今回の論文ではFigure 3をMain textのシステム図とし、p.20相当の詳細図はSupplementaryまたはImplementation節の補助図とする。

![Implementation-level system diagram.](./paper_v8_figures/figS3_implementation_detail.png)
