# タイトル

病棟共有空間における空間的文脈とスタッフの判断知識を統合した危険度評価・見守り優先順位付け

英語タイトル案：  
**Context-Integrated Risk Evaluation and Monitoring Prioritization Incorporating Staff-specific Judgment Knowledge in Shared Hospital Spaces**

# 投稿先

第一候補：IEEE Access  
上振れ候補：IEEE Transactions on Systems, Man, and Cybernetics

# 論文の位置づけ

本研究は、病棟共有空間における複数患者の見守りを対象とする。患者本人の属性・行動からなる**患者文脈**に加え、周囲物体・施設構造・スタッフの位置や見守り状況からなる**空間的文脈**を統合し、患者ごとの時系列危険度と見守り優先順位を算出する。

また、危険度評価で何をどの程度重視するか、条件の組み合わせをどう評価するかという**スタッフごとの判断知識**を、AHP重みとFuzzyルールとしてモデルへ反映する。

評価では、空間的文脈の有無による出力差（RQ1）と、スタッフごとの判断知識による出力差（RQ2）を分析する。通知生成は、危険度評価結果の利用例として扱う。

# Research Questions

* **RQ1：** 空間的文脈を考慮することで、患者の時系列危険度と見守り優先順位はどのように変化するか。
* **RQ2：** スタッフごとの判断知識を反映することで、同一入力に対する時系列危険度と見守り優先順位はどのように変化するか。

# 1. Introduction

患者見守り支援は、転倒・離床・姿勢・活動の検知を中心に広く研究されてきた。さらに、ベッド周辺の位置情報を用いたリスク評価や、電子カルテ・バイタル等に基づく継続的な患者リスク評価も提案されている \cite{Mubashir2013SurveyFalls,Gutierrez2021VisionFallReview,Isomoto2018FallRisk,Rothman2013ContinuousCondition,Escobar2020Deterioration}。詳細はSection 2で述べる。

本研究では、こうした見守り支援を**病棟共有空間における複数患者の見守り**へ適用する。この環境には、危険度と見守り優先順位を評価するうえで重要な2つの特徴がある。

第一に、見守り優先順位は患者本人の状態・行動だけでなく、患者を取り巻く状況に依存する。Figure 1に示すように、同じ立ち上がり動作でも、スタッフが患者を直接介助している場合と、近くのスタッフが別業務に従事している場合では、見守り優先順位が異なり得る。本研究では、周囲物体、施設構造、スタッフとの距離・方向、見守り状況など、患者本人以外の空間的状況を**spatial context（空間的文脈）**と定義する。

![Example illustrating spatial-context- and staff-dependent monitoring priority.](./paper_v9_figures/fig1_problem_setting.png)

**Figure 1.** 病棟共有空間における見守り優先順位の例。同じ立ち上がり動作でも、周囲物体・施設構造・スタッフの見守り状況によって優先順位が変化し得る。また、スタッフが直接介助している比較的安全な状況でも、患者の立ち上がり自体を重視するスタッフと、介助中であることを重視するスタッフでは評価が異なり得る。  
*Draft note: source slide p.2. Final figureでは “notification priority” を “monitoring priority” に変更し、左側の状況に「患者文脈を重視するスタッフ：立ち上がりを重視して優先度を維持」「空間的文脈を重視するスタッフ：スタッフが直接介助していれば優先度を下げる」という対比を追加する。*

第二に、病棟共有空間では複数のスタッフが見守りに関与し、同じ状況でも危険度の判断基準が異なり得る。Figure 1左側の状況でも、患者の立ち上がり自体を重視して一定の優先度を保つスタッフと、スタッフが直接介助しているため優先度を下げるスタッフが考えられる。本研究では、このような特徴量の重み付けや条件の組み合わせに関する判断を**スタッフの判断知識**と呼ぶ。

空間的文脈を用いたリスク評価と、専門家の判断をモデルへ反映する研究はそれぞれ存在する \cite{Koshmak2014ContextFallRisk,Boulet2020ExpertWeights,TenBroeke2021BAIT}。一方、病棟共有空間において、**空間的文脈とスタッフごとの判断知識を同時に扱い、複数患者の時系列危険度と見守り優先順位を評価する研究は、確認した範囲で限定的である**。

そこで本研究では、患者文脈と空間的文脈を階層的に統合し、スタッフごとの判断知識をモデルパラメータとして反映する危険度評価フレームワークを構築する。評価では、空間的文脈の有無とスタッフごとの判断知識が、時系列危険度と見守り優先順位へ与える影響を検証する。

本研究の主な貢献は以下の2点である。

1. **Context-integrated risk evaluation:** 患者属性・行動、周囲物体・施設構造、スタッフとの相対関係を階層的に統合し、時系列危険度と見守り優先順位を算出するフレームワークを構築した。
2. **Staff-specific parameterization and evaluation:** スタッフの一対比較と状況判断をAHP重みとFuzzyルールへ変換し、同一入力に対するスタッフ別モデルの出力差を時系列危険度と見守り優先順位から分析する。

# 2. Related Work

## 2.1 Patient Monitoring and Context-aware Risk Assessment

病院・高齢者ケア環境では、ウェアラブルセンサ、環境センサ、RGB・赤外・深度カメラ等を用いた転倒検知、活動認識、離床検知が広く検討されてきた \cite{Mubashir2013SurveyFalls,Igual2013FallDetectionChallenges,Gutierrez2021VisionFallReview}。病院内でも、ベッド離脱や転倒につながり得る動作を検知してスタッフへ通知するシステムが実装されている \cite{Balaguera2017MedicalIoT,Jones2020VideoMonitoring}。

患者周囲の情報を扱う研究もある。ベッド・床等の施設構造を患者監視に利用する方法 \cite{Inoue2018BedDetection,Komagata2019BedMonitoring} や、ベッド周辺の患者位置を転倒リスクへ対応づける方法 \cite{Isomoto2018FallRisk} が提案されている。また、Ambient Assisted Livingでは、本人情報と環境センサを統合して文脈依存の危険を推定する研究が行われている \cite{SaldanaJimenez2009ContextRisk,Koshmak2014ContextFallRisk}。

これらは空間的文脈を扱ううえで重要な先行研究である。一方、本研究は病棟共有空間の複数患者を対象とし、患者文脈と空間的文脈を統合して時系列危険度と見守り優先順位を算出する。

## 2.2 Risk Prioritization and Expert-informed Decision Models

病院では、電子カルテ、バイタル、検査値等から患者状態を連続的にスコア化し、高リスク患者の確認や介入を支援する研究が行われている \cite{Rothman2013ContinuousCondition,OBrien2020OptimizedEWS,Escobar2020Deterioration}。本研究では、生理的悪化ではなく、共有空間内の患者行動と空間的文脈に基づく短時間の見守り優先順位を扱う。

専門家の判断を意思決定モデルへ反映する方法として、医師が各臨床因子へ与える重要度を数値化する方法 \cite{Boulet2020ExpertWeights} や、仮想症例への選択から因子重みと条件間相互作用を推定する方法 \cite{TenBroeke2021BAIT} が提案されている。AHPとFuzzy推論も、多基準意思決定や医療リスク評価に用いられている \cite{Saaty1990AHP,Zadeh1965FuzzySets,Mamdani1975LinguisticControl,Uzoka2011FuzzyAHPClinical,Chang2022FuzzyAHPRisk}。

本研究では、スタッフごとの回答をAHP重みとFuzzyルールへ変換し、同一入力に適用したときの時系列危険度と見守り優先順位の差を分析する。

## 2.3 Position of This Study

既存研究では、空間的位置を用いた転倒リスク、context-aware risk assessment、患者の連続リスクと優先化、専門家判断のパラメータ化がそれぞれ検討されている。一方、病棟共有空間において、患者文脈・空間的文脈・スタッフごとの判断知識を同一モデルで扱い、複数患者の時系列危険度と見守り優先順位を評価する研究は、確認した範囲で限定的である。

本研究は、この統合と出力挙動の評価に焦点を当てる。

# 3. Proposed Method

## 3.1 Overview

提案手法は、患者文脈と空間的文脈から患者ごとの危険度を算出し、複数患者間の見守り優先順位を決定する。

スタッフの判断知識は、以下の2種類のパラメータとして反映する。

* 同一カテゴリ内の特徴量の重要度：AHP重み。
* 複数のリスク因子の組み合わせと危険度の関係：Fuzzyルール。

Figure 2は、同一状況でも、患者文脈と空間的文脈のどちらを重視するかによって危険度評価が変わる概念例を示す。

![Conceptual examples of context- and staff-dependent risk propagation.](./paper_v9_figures/fig2_context_staff_concept_2x2.png)

**Figure 2.** 空間的文脈とスタッフの判断知識に応じた危険度評価の概念例。  
*Draft note: source slides pp.3-6. Final figureでは4枚をそのまま並べず、空間的文脈の2条件×スタッフの判断知識の2条件からなる2×2の模式図として再作図する。*

Figure 3に提案システムの全体構成を示す。患者属性・行動、周囲物体・施設構造、スタッフの位置・見守り状況から特徴量を取得し、階層的に危険度へ統合する。スタッフごとのAHP重みとFuzzyルールは危険度評価部へ入力する。

![Overview of the proposed risk evaluation system.](./paper_v9_figures/fig3_system_overview.png)

**Figure 3.** 提案システムの全体構成。  
*Draft note: source slide p.7. Final figureではスタッフ入力を “Questionnaire → AHP weights / Fuzzy rules” と明示し、Notification generatorは補助出力として小さくする。*

## 3.2 Feature Selection and Classification

### 3.2.1 Feature selection

特徴量候補は、対象病院の病棟デイルームで熟練看護師2名に実施した約30分の聞き取りを基に整理した。見守り時に着目する情報、危険と判断する状況、複数患者の優先順位を中心に質問し、特徴量選定に用いた。

### 3.2.2 Feature classification

危険度評価に用いる特徴量を、内外・動静の2軸で4群に分類する。

| 分類 | 内容 | 例 |
|---|---|---|
| 内的・静的要因 | 患者本人の属性 | 年齢層、重症度など |
| 内的・動的要因 | 患者本人の行動 | 立ち上がり、姿勢崩れ、顔を触る等 |
| 外的・静的要因 | 周囲物体・施設構造 | 車椅子、点滴棒、手すりとの距離等 |
| 外的・動的要因 | スタッフの位置・見守り状況 | スタッフとの距離、相対方向、視野等 |

患者文脈は内的要因、空間的文脈は外的要因としてモデルへ入力する。

### 3.2.3 Feature extraction

* 内的・静的要因：人物画像や事前情報から患者属性を取得する。
* 内的・動的要因：姿勢推定結果から危険動作との類似度を算出する。
* 外的・静的要因：周囲物体の有無や施設構造との距離を算出する。
* 外的・動的要因：スタッフとの相対距離・方向等を算出する。

実装では、RGB画像、点群、位置情報、姿勢推定、VQA、施設地図等を組み合わせて特徴量を取得する。

## 3.3 Questionnaire-based Parameterization

### 3.3.1 Questionnaire design

スタッフごとの判断知識を反映するため、以下の回答を取得する。

* 同一カテゴリ内の特徴量間の重要度比較。
* 患者文脈と空間的文脈の重視度。
* 複数状況に対する危険度または見守り優先順位の判断。

一対比較回答からAHP重みを算出し、状況判断からFuzzyルールとメンバーシップ関数を設定する。

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

$IS$、$ID$、$ES$、$ED$は、それぞれ内的・静的、内的・動的、外的・静的、外的・動的要因を表す。

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

スタッフごとにAHP重み、$\alpha$、Fuzzyルール等を設定し、同一入力に対するスタッフ別の時系列危険度を生成する。

## 3.5 Prioritization and Traceability

各時刻で$R_i(k)$を患者間で比較し、見守り優先順位を決定する。

また、階層構造内の特徴量と中間危険度を保持し、危険度変化に寄与した要因を参照できるようにする。必要に応じて、順位変化や危険度上昇をトリガとして通知を生成し、主要因を通知理由として付加する。

![Illustrative example of extracting factors contributing to a risk change and generating a notification.](./paper_v9_figures/fig9_notification_example.png)

**Figure 4.** 危険度変化と内部要因から通知理由を生成する補助例。  
*Draft note: source slide p.21. Main contributionではなく、Method末尾またはSupplementaryの例として扱う。*

# 4. Experiments

## 4.1 Evaluation Design

評価はRQ1とRQ2に対応する2つの実験から構成する。

* **Evaluation 1 / RQ1:** 空間的文脈の有無による時系列危険度と見守り優先順位の変化を比較する。
* **Evaluation 2 / RQ2:** スタッフごとの判断知識による時系列危険度と見守り優先順位の変化を比較する。

RQ2では、判断知識が対照的な2名のケーススタディと、全回答者（n=10）の傾向分析を行う。

## 4.2 Evaluation 1: Effect of Spatial Context

患者本人の行動条件を揃え、スタッフの位置・見守り状況、周囲物体、施設構造等の空間的文脈を変化させたシナリオを用いる。Figure 5に立ち上がり動作を用いた代表シナリオを示す。

![Simulation scenario for evaluating the effect of spatial context.](./paper_v9_figures/fig4_rq1_standup_scenario.png)

**Figure 5.** RQ1で用いる立ち上がりシナリオ。  
*Draft note: source slide p.8. Final figureでは代表4時刻程度に削減し、患者・スタッフ・周囲物体・施設構造の変化を読みやすくする。*

比較条件は以下とする。

* **Patient context only:** 患者文脈のみ。
* **Spatial context included:** 患者文脈＋空間的文脈。

Figure 6に、両条件における時系列危険度の比較例を示す。

![Risk time-series comparison with and without spatial context.](./paper_v9_figures/fig5_rq1_standup_result.png)

**Figure 6.** 空間的文脈の有無による時系列危険度の比較。  
*Draft note: source slide p.9. Final figureでは同一profileの2条件をoverlayし、RQ1の比較が直接読める形式へ再設計する。*

主な評価項目は以下とする。

* 空間的文脈変化前後の危険度差。
* 最大危険度・危険度変動幅。
* 2条件で見守り優先順位が異なる時間割合。
* 患者間の順位逆転。

## 4.3 Evaluation 2: Reflection of Staff-specific Judgment Knowledge

### 4.3.1 Case study of two contrasting staff members

病棟共有空間で取得した同一の実データに、異なるスタッフのアンケート回答に基づくモデルを適用する。判断知識が対照的な2名を選定し、AHP重み・Fuzzyルールと出力の対応を確認する。

Figure 7に、実データへ複数のスタッフ別モデルを適用した時系列危険度の例を示す。

![Risk time series obtained from staff-specific models for real-world data.](./paper_v9_figures/fig6_rq2_case_study.png)

**Figure 7.** 実データに対するスタッフ別の時系列危険度の例。  
*Draft note: source slide p.16. Final figureでは対照的な2名を強調し、他回答者は薄色または非表示とする。イベントと空間的文脈の変化時刻も併記する。*

比較項目は以下とする。

* AHP重みとFuzzyルールの特徴。
* 時系列危険度のピーク・変動幅・上昇タイミング。
* 見守り優先順位と順位逆転。
* 危険度上昇に寄与した中間リスク因子。

### 4.3.2 Group-level analysis

全回答者（n=10）について、アンケートから得た判断指標と危険度出力の関係を分析する。

現時点の可視化候補をFigures 8および9に示す。

![Relationship between spatial-context orientation and model output.](./paper_v9_figures/fig7_rq2_group_context.png)

**Figure 8.** 空間的文脈の重視度と危険度出力の関係を示す現行可視化。  
*Draft note: source slide p.17. Final figureでは、x軸をスタッフごとの空間的文脈重視指標、y軸を空間的文脈変化に対する$\Delta$Risk等とした散布図へ変更する。*

![Relationship between the object-vs-monitoring preference index and model output.](./paper_v9_figures/fig8_rq2_group_di.png)

**Figure 9.** 周囲物体とスタッフの見守り状況の重視度と危険度出力の関係を示す現行可視化。  
*Draft note: source slide p.18. Figure 8と同様、スタッフごとの判断指標と対応する出力差を直接比較する散布図へ変更する。*

分析候補は以下とする。

* 空間的文脈の重視度と、空間的文脈変化時の危険度変化量。
* 患者行動の重視度と、危険行動発生時の危険度ピーク。
* 特徴量重みと、その特徴量が変化した区間の危険度変化量。

n=10であるため、散布図、順位相関、判断指標と出力の対応を中心に確認する。AHP回答についてはCIまたはCRを算出し、回答の整合性を確認する。

# 5. Discussion

## 5.1 Effect of Spatial Context

RQ1では、患者行動が同じでも、周囲物体・施設構造・スタッフの見守り状況によって時系列危険度と見守り優先順位が変化するかを確認する。

見守り優先順位の変化は、空間的文脈が「誰を優先して見るか」という判断へ与える影響として解釈する。

## 5.2 Staff-specific Risk Evaluation

RQ2では、スタッフごとの判断知識がAHP重み・Fuzzyルールを介して時系列危険度と見守り優先順位へどう反映されるかを分析する。

ケーススタディでは出力差の形成過程を示し、n=10の分析では判断指標と出力の対応を確認する。

## 5.3 Limitations

* 危険度の臨床的な正解ラベルに対する精度検証は行っていない。
* n=10の分析は探索的であり、スタッフ一般への統計的な一般化には限界がある。
* 実データにはセンサ・認識ノイズが含まれる。
* 特徴量は限られた現場観察と熟練看護師2名への聞き取りを基に選定している。
* スタッフ別モデルの妥当性、通知理由の理解度、業務負荷、Alarm fatigue、事故防止効果は評価していない。

## 5.4 Future Work

* スタッフ自身による危険度・見守り優先順位の評価と提案手法の出力を比較する。
* 空間的文脈なし、平均モデル、スタッフ別モデルを比較し、各構成要素の寄与を評価する。
* より多くのスタッフ・患者状況を用いて、判断知識と危険度出力の関係を検証する。
* 通知理由の理解度・有用性を独立したユーザー評価として検証する。

# 6. Conclusion

本研究では、病棟共有空間における複数患者の見守りに向けて、患者文脈、空間的文脈、スタッフごとの判断知識を統合する危険度評価フレームワークを提案する。

患者属性・行動、周囲物体・施設構造、スタッフの位置・見守り状況を階層的に統合し、患者ごとの時系列危険度と見守り優先順位を算出する。また、スタッフへのアンケートからAHP重みとFuzzyルールを設定し、同一入力に対するスタッフ別モデルの出力差を評価する。

RQ1では空間的文脈による出力変化を、RQ2ではスタッフごとの判断知識による出力差を分析する。これにより、病棟共有空間の状況とスタッフの判断知識を反映した見守り優先順位付けの基礎的枠組みを示す。

# Appendix: Draft-only Figure Management Notes

以下は本文に直接入れず、最終図の再設計またはSupplementary候補として保持する。

## A. Additional RQ1 scenario

顔を触る動作を用いたシナリオと結果（source slides pp.10-11）は、立ち上がり以外の動作でも同様の傾向が得られることを示す補助実験として利用できる。

![Additional RQ1 scenario using face-touching behavior.](./paper_v9_figures/figS1_touchface_scenario.png)

![Additional RQ1 result using face-touching behavior.](./paper_v9_figures/figS2_touchface_result.png)

## B. Four-patient simulation candidates

Source slides pp.12-15は、スタッフの判断知識と見守り優先順位の関係を統制条件で示す素材として利用できる。Main textへ入れる場合は、現行の折れ線グラフではなくheatmapまたは順位表へ再設計する。

## C. Implementation detail

Source slide p.20は、VQA、Pose similarity evaluator、relative position calculator等の実装詳細を示す。今回の論文ではFigure 3をMain textのシステム図とし、p.20相当の詳細図はSupplementaryまたはImplementation節の補助図とする。

![Implementation-level system diagram.](./paper_v9_figures/figS3_implementation_detail.png)
