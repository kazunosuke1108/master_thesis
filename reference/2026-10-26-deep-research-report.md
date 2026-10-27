# 病棟見守りにおける階層型危険度・優先度評価モデルの研究ギャップと新規性

## エグゼクティブサマリー

2026年10月25日時点の文献を、病棟転倒予測、連続患者モニタリング、介護施設のcontext-aware alerting、看護師の臨床判断、多基準意思決定、AHP、Fuzzy推論、相互作用を扱うMCDA、ランキング感度解析まで広げて調査した。

結論は、**提示された研究方針は「条件付きで」独立した研究課題として成立するが、現状のままでは手法そのものの新規性は強くない**、である。

| 判断点 | 調査結果 |
|---|---|
| 「判断知識の表現・作用の解明」は研究課題になるか | **なる。ただし「スタッフの判断知識」ではなく「質問紙で表明された評価基準・選好パラメータ」と表現するのが安全**。AHPによる重み取得自体は医療で既に多数実施されている。citeturn16view3turn17search0 |
| 病棟見守り側のギャップ | 患者状態だけでなく、家具、行動、スタッフの存在・近接性まで扱う研究はすでに存在するため、**「空間文脈を初めて導入した」は成立しない**。特に2025年には、看護師―高齢者距離まで用いるcontext-aware alertingが報告されている。citeturn20view2turn20view1turn21search4turn21search28 |
| AHP＋Fuzzyの新規性 | **ほぼない**。FAHP＋Fuzzy reasoning、Pythagorean fuzzy AHP＋FIS、医療診断でのfuzzy AHP＋Sugeno FISなどが既に存在する。citeturn18search16turn18search1turn18search2 |
| 「階層＋要因間相互作用」の新規性 | **一般論としてはない**。Choquet積＋Multiple Criteria Hierarchy Process、interactionを含むELECTRE等が既に階層と非加法的相互作用を同時に扱う。citeturn21search2turn15search18 |
| 「パラメータ変更による順位逆転」の新規性 | **順位感度解析自体は新しくない**。固定された候補集合で、順位を変える最小重み変化を求める研究は少なくとも1997年に存在する。これはAHPで候補追加時に生じる古典的rank reversalとは別問題である。citeturn21search1turn16view3 |
| 最も有望な独自性 | **複数患者を同時に見守るという資源配分問題に対し、患者・環境・スタッフによる現在文脈と、質問紙由来の明示的判断設定を分離してモデル化し、「どの文脈×判断設定で患者間の注意優先順位が変わるか」をモデル内部の構造まで遡って解析すること**。これは今回調べた病棟・介護モニタリング研究では十分体系化されていなかった。 |
| 最大の査読リスク | 「AHP＋FISを既知技術として病棟に適用し、普通の感度分析をしただけ」「アンケート重みが実際の判断を再現する証拠がない」「事故確率でも臨床的優先度でもないスコアを“危険度”と呼んでいる」の三点。 |
| TSMC: Systems向けに必要な強化 | 医療応用の紹介より、**一般化可能なdecision-system formulation、階層・条件依存構造の必然性、ランキング境界・頑健性・要因帰属の体系的解析、加法モデル等との比較**を中心にする必要がある。IEEE SMCの公式scopeはsystems modeling, analysis, decision making, issue formulation/interpretation等を明示している。citeturn8search4 |

したがって、提示された方針、すなわち便宜上「案C」と呼ぶ

> 患者・空間的文脈と現場スタッフの危険度評価に関する判断知識を階層的・明示的に表現し、判断設定の違いがどの状況で、なぜ危険度順位を維持・変化させるのかを明らかにする

は、**方向性としては妥当だが、そのままではやや弱い**。研究課題を

> **患者・空間・見守り状態を含む多患者見守りにおいて、質問紙から明示化された評価基準が、文脈と相互作用して注意優先順位を形成する過程、およびその順位の安定性・変化境界を明らかにする**

と再定義する方が、既存研究との差が明瞭になる。

さらに重要なのは、現在の出力にスタッフ位置・見守り状態を含めるのであれば、それを単純な「転倒危険度」「事故発生リスク」と呼ぶことは避けるべきである。**「見守り必要度」「attention priority」「monitoring priority」などの方が構成概念として整合的**である。事故確率を主張するには事故ラベルに対する予測・校正が、臨床的対応優先度を主張するには専門家判断または実臨床アウトカムによる別の妥当性検証が必要になる。既存の転倒予測研究はその点で、明確な転倒アウトカムを予測対象に置いている。citeturn9view3turn10search0turn12view4


## 既存研究が扱えていることと残る技術課題

### 患者中心の転倒リスク予測から文脈認識型モニタリングへ

病院内転倒の古典的な評価は、既往歴、認知状態、移動能力、薬剤、排泄など患者側の要因を中心にする。近年はEHRを用いた機械学習・Bayesian networkへ移行しており、Choらは時変EHR・看護記録を用いるBayesian networkを構築し、LindbergらもEHR・管理データから機械学習で転倒リスクを予測した。これらは患者状態の時間変化を取り込める一方、**その瞬間に患者の周囲に何があり、誰が見ているか**を主たる入力にはしていない。citeturn9view3turn10search0

2025年の系統レビューでは、入院高齢患者の転倒予測モデル13件のうち外部検証されたものは4件にとどまり、AUCは0.630–0.851、全モデルが高いbias riskと判定されている。また、従来の臨床評価は静的で、患者状態の動的変化を捉えにくいことも問題として整理されている。したがって、**「より動的な文脈を取り込む」こと自体には依然十分な動機があるが、「予測精度を上げれば見守り優先順位問題も解ける」とは限らない**。citeturn12view4

一方、2025年までには空間的文脈を扱う研究がかなり進んでいる。Gabrielらの病院内ビデオモニタリングでは、患者・スタッフ等の人物role、家具位置、動き、boundary crossing、「patient alone」などをリアルタイム推論し、11病院・300人超の高転倒リスク患者、1,000日超のデータで評価している。したがって、**家具・スタッフ・患者行動を統合すること自体を新規性にはできない**。citeturn20view2

さらにDanialらの急性期脳卒中病棟のAI Patient Sitterは、複数病床を同時監視し、ベッド離床、ベッド端滞在、転倒等を検出して色付き警報を提示し、スタッフの到着も認識する。これは「多患者」「リアルタイム」「スタッフ存在」という点で今回の研究に非常に近い。ただし、その論文が主に検証しているのはセンシング・警報システムであり、**複数患者をどの判断原理で比較順位付けするかを、専門家の明示的選好モデルとして分析することではない**。citeturn20view1

より強い競合は、Nahid、Ahad、Inoueによる2025年の介護施設研究である。28名の高齢者、4名の看護師、3室、7日間の実験で、video-derived skeletal motion、care level、動的なnurse–elderly proximityを用い、rule-based logicとLLM reasoningを組み合わせて警報を抑制・遅延・確認するcontext-aware alertingを提案している。つまり、**「患者の状態＋スタッフとの距離から、その状況における警報の必要性を決める」という発想まで既に存在する**。今回の差は、スタッフ近接性そのものではなく、複数の判断基準を明示的に階層化し、その設定差が患者間ランキングにどう作用するか、というdecision-model側に求める必要がある。citeturn21search4turn21search28

なお、このNahidらの論文は今回の調査では出版社本文への直接アクセスが429エラーとなったため、PubMedの査読済み書誌・抄録と、公開されている論文要約・査読情報で方法を照合した。したがって細部については全文確認済み文献より慎重に扱っている。citeturn21search4turn21search8

### 「危険度」「確率」「見守り必要度」「対応優先度」は分けるべき

今回のモデルの正当化に最も重要なのは、手法より先に**出力の意味を限定すること**である。

| 出力 | 意味 | 妥当性を主張するために必要な評価 |
|---|---|---|
| 事故発生確率 | 「今後T時間以内に転倒する確率」等 | 実事故ラベル、校正、discrimination、外部検証 |
| 危険度スコア | 複数危険要因を合成した相対指標 | 構成概念妥当性、既知状況に対する方向性、基準との関連 |
| 見守り必要度 | 現在の患者状態・環境・既存の監視状況からみた追加観察の必要性 | シナリオ妥当性、スタッフ判断等との比較が望ましい |
| 対応優先度 | 同時に複数患者が存在するとき、限られた人員をどこへ向けるべきか | 資源制約・緊急性を含む優先順位妥当性 |

現在の提案に**スタッフ位置・方向、周囲物体、他患者との比較**まで入るなら、事故の「確率」よりも見守り・注意の「優先度」に近い。Iwamotoらの2026年の看護タスク優先順位研究でも、実際の優先付けは医学的緊急性だけでなく、患者個別性や組織・運用上の制約によって動的に決まることが示されている。citeturn21search3turn21search7

これは単なる言葉遣いではない。たとえば同じ転倒しやすい患者でも、スタッフが直近で直接観察している状況と、遮蔽物の向こうで単独歩行を始めた状況では、**患者固有の転倒素因は同じでも「追加注意を割り当てる必要性」は異なる**。この区別を明示すると、空間文脈を含める理由が自然になる。

### 危険要因の統合方法と今回の位置づけ

既存研究は大まかに、加重スコア、統計・確率モデル、機械学習、ルール・Fuzzyモデル、MCDAに分けられる。

加重和は透明である反面、

\[
R=\sum_i w_i x_i
\]

では、原則として一つの要因の効果は他要因に依存しない。「患者が離床している**かつ**スタッフが遠い場合だけ急激に優先度を上げる」といった関係は直接表現しにくい。

Fuzzy IF–THENルールなら、

\[
\text{IF mobility risk = high AND staff coverage = low THEN priority = very high}
\]

のような**評価上の条件付き相互作用**を明示できる。ただし、これは確率論的な「条件依存」を推定しているわけではない。論文では「conditional dependency」と書くより、**context-dependent interaction / conditional decision logic / non-additive interaction**と定義した方が誤解が少ない。

しかしFuzzyだけが相互作用を扱えるわけではない。Choquet積は非加法的criteria interactionを、ANPは基準間相互依存を、interaction付きELECTREも相乗・冗長等の効果を扱える。特にAngilellaらはChoquet preference modelとMultiple Criteria Hierarchy Processを組み合わせ、**criteria hierarchyとcriteria interactionを同時に扱い、decision makerによるpairwise preference informationまで利用している**。citeturn21search2

したがってRQ1について、今回の提案が狙うべき技術課題は、

> **患者・環境・スタッフによる動的見守り文脈を単に検出することではなく、それらが複数患者の注意優先順位へ変換される判断構造を明示し、その構造・判断設定・文脈の相互作用を追跡可能にすること**

である。

この課題設定なら、患者状態のみを予測するMLとも、イベントを検知して警報するcontext-aware monitoringとも差別化できる。


## 現場判断知識を取得する意義と限界

### スタッフ間で判断は実際に異なる

「同一状況に対する専門家判断が一意ではない」という前提には実証的根拠がある。

Lavoieらは62名の看護師による444件の交代時handoffを調査し、少なくとも一方が患者悪化リスクありと判断した125件のうち、双方が一致したのは32件、**93件、74.4%で判断が不一致**だった。情報cueの数・重症度、直近のICU移送、原因不明疼痛、delirium、患者についての既知情報等が判断と関係した。これは転倒そのものではないが、「同じ患者情報を受けても臨床リスク判断は人・文脈により変わる」ことを直接示す。citeturn17search2turn17search6

転倒判断についてはTakaseらが2024年にQ methodologyを用い、看護師が各転倒リスク要因をどのように相対評価するかを研究している。さらに2025年の多施設横断研究は、看護師の転倒リスク判断のばらつきをcognitive biasesとcontextual factorsの双方から検討している。これは、**転倒リスク要因の重視の仕方に一様でない判断パターンが存在する**という、今回のAHPパラメータ化に直接近い根拠である。citeturn17search5turn17search20turn15search4

また2026年のIwamotoらは看護師のtask prioritizationを質的に分析し、医学的緊急性だけではなく、患者個別性・価値観、組織的・運用的制約が優先順位に関与すると報告している。これは「事故リスクが最大の患者＝常に最優先」とは限らず、**見守り・対応優先度が文脈依存のdecision problemである**ことを支持する。citeturn21search3

### ただし「個人ごとの判断を使うべき」とは導けない

ここは非常に重要である。

スタッフ間差が観測されたからといって、

> 個人の評価基準をそのままシステムに反映した方が安全である

とは言えない。Takaseらの2025年研究自体が「cognitive biases」を検討対象としており、個人差には専門性だけでなくbiasも含み得る。citeturn15search4

WHOの患者安全の考え方も、個々の医療従事者を責めるより、プロセス・システムを設計して安全を確保するsystem approachを重視している。したがって実運用では、個人の好みを無制約に反映するより、施設共通・専門家合意・臨床基準を基礎にする方が患者安全上は説明しやすい。citeturn19search18

一方、**研究目的として個人ごとの設定を持たせる価値はある**。それは「個人化した方が正しい」からではなく、

> 同じ形式の意思決定モデルに異なる表明判断基準を入力したとき、どの判断差が出力に実質的な差を生じさせ、どの判断差は最終順位にはほとんど影響しないのか

を調べられるからである。

つまり個別設定を**実運用機能ではなく、判断異質性を解析する実験変数**として位置づける。この方が研究としては強い。

### AHPで取得できるのは「表明された評価基準」

Pecchiaらは既に2011年に、医療専門職の知識・経験に基づき、高齢者の転倒リスク要因の相対的重要度をAHPで取得している。したがって、**「転倒リスク要因の専門家重要度をAHPで質問紙取得する」ことは新規ではない**。citeturn17search0turn17search30

さらにSchmidtらの2015年の医療AHP系統レビューには121論文が含まれ、AHPは医療の多様な判断・選好取得にすでに利用されていた。同レビューは同時に、誰に質問すべきか、不整合回答をどう扱うか、結果の安定性をどう示すか、といった方法上の課題を指摘している。citeturn16view3

AHPの利点は、直接「重みを0.37にしてください」と聞くよりも一対比較で相対的重要度を得られ、階層構造と整合性評価を持てる点にある。一方で、要因数が増えると比較数が急増し、回答者の疲労・不整合も問題になる。さらに、AHPのconsistencyが高いことは、**回答内の論理的一貫性**を示すのであって、その重みが実際の行動判断や臨床的正しさを示すものではない。医療AHP研究の方法論的ばらつきもSchmidtらが指摘している。citeturn16view3

したがって論文中の表現は、

**避けたい**
> We acquired each staff member's clinical judgment knowledge.

より、

**推奨**
> We parameterized explicitly stated evaluation criteria elicited from staff questionnaires.

が適切である。

そして、

\[
\text{質問紙回答}
\rightarrow
\text{表明された評価基準}
\rightarrow
\text{モデルパラメータ}
\]

までは今回検証できるが、

\[
\text{モデル順位}
\approx
\text{本人が実際の現場で行う判断}
\]

は別のbehavioral validationを行わなければ主張できない。

### それでも知識取得を行う学術的意義

本人判断の完全再現を目的としなくても、意義は三つ残る。

第一に、**モデルパラメータに意味のある根拠を与えられる**。恣意的な重み・ルールではなく、「回答された基準」をモデル化できる。

第二に、**判断異質性を構造化した実験変数にできる**。単に「看護師AとBでスコアが違った」で終わらず、「Aは患者行動カテゴリを強く重視するが、Bはstaff coverageとの組合せを重視するため、この文脈でのみ順位が変わる」と分析できる。

第三に、**将来の標準化に必要な情報を得られる**。どのパラメータ差が実際のランキングには影響せず、どの差だけが意思決定を不安定にするかが分かれば、後続研究で「何を合意形成すべきか」を絞れる。これは個人化の推奨とは逆に、標準化すべき判断項目の発見にも使える。

この三点までなら、スタッフ本人との順位一致を今回検証しなくても成立する。ただし「staff judgmentを再現した」「clinical expertiseを忠実にモデル化した」という主張はできない。


## 提案モデルの新規性と最も近い競合

### AHPとFuzzyを組み合わせること自体には新規性がない

この点はかなり明確である。

Ilbaharらは2018年、労働安全リスク評価にFine–Kinney、Pythagorean fuzzy AHP、Fuzzy Inference Systemを統合した。citeturn18search16

RibasとPérez-Díazは2019年、ダム安全リスク評価においてFAHPで指標重要度を定め、その後二段階のfuzzy approximate reasoningでリスクを統合している。**「AHP系重み＋階層的なFuzzy推論＋専門家知識」という構造は今回のモデルにかなり近い**。citeturn18search1

MariadossとAugustinは2023年に、妊娠中の心血管疾患診断でfuzzy AHPとSugeno FISを統合し、臨床専門家によるrisk factor contributionも導入している。citeturn18search2turn18search18

さらに2024年のFernándezらも、AHPで専門家からcriteria importanceを取得しFISと統合する一般的枠組みを提示している。したがって**「AHPで重要度、Fuzzyで非線形統合」という構成そのものをmethodological noveltyと書くことは危険**である。citeturn18search3

### 「カテゴリ内重要度」と「カテゴリ間相互作用」の分離も一般概念としては既知

今回の考え方には設計上の美点がある。

\[
\text{要因}
\overset{\mathrm{AHP}}{\longrightarrow}
\text{カテゴリスコア}
\overset{\mathrm{FIS}}{\longrightarrow}
\text{総合優先度}
\]

とすることで、

- 同一カテゴリ内では「どちらの要因をより重視するか」
- カテゴリ間では「どの組合せなら評価を強く・弱くするか」

を別々に表現できる。

しかし、この**分離原理そのものは新しい数学的概念ではない**。MCDAには、hierarchyとnon-additive interactionを同時に扱う長い蓄積がある。Angilellaらの2016年研究はMultiple Criteria Hierarchy ProcessとChoquet integralを組み合わせ、decision makerによるpairwise preference informationから、階層・相互作用・不確実性を扱っている。citeturn21search2

Correnteらの2017年ELECTRE拡張も、階層化されたcriteria、criteria interaction、imprecise weights、stochastic analysisを同時に扱う。citeturn15search18turn15search22

したがって、

> AHPとFuzzyを使わないとこの関係は表現できない

とは言えない。

言える可能性があるのは、

> **今回の問題では、カテゴリ内重要度をpairwise questionnaireで取得し、カテゴリ間の状況依存ロジックを言語的Fuzzy ruleとして別々に取得・表示することで、医療スタッフが解釈できる形のparameterizationを構成した**

という**設計・知識表現上の利点**である。

### 順位変化の解析にも強い既存研究がある

ここも新規性を過大評価しない方がよい。

固定された候補集合について、

> 「どのcriterion weightをどれだけ変えれば現在の最良候補が変わるか」

を求める研究は古くから存在する。TriantaphyllouとSánchezの1997年の感度分析は、MCDMのcriterion weightsやalternative performanceの**ランキングを変えるために必要な最小変化**を扱っている。citeturn21search1

近年もMCDA sensitivity analysisは大きな研究領域であり、Więckowskiらのレビューはweight perturbation、ranking robustness等を体系化している。citeturn21search13

したがって、

> 「パラメータを変えて順位が逆転するかを調べた」

だけでは弱い。

また、ここで扱う順位変化を**AHPのclassic rank reversal**と呼ばない方がよい。Classic AHP rank reversalは、候補の追加・削除や正規化等に伴い既存候補順位が変化する問題である。今回の問題は、

> **固定された患者集合に対するparameter sensitivity / preference-induced ranking transition**

である。医療AHPレビューも古典的rank reversal研究とsensitivity analysis研究を別々に引用している。citeturn16view3

### 今回のモデルで独自性になり得る部分

独自性を三段階に分けると明確である。

| 要素 | 評価 |
|---|---|
| AHPで専門家の相対重要度を取得 | **既存** |
| Fuzzy ruleで条件付き関係を表現 | **既存** |
| AHP＋Fuzzy | **既存** |
| hierarchy＋criteria interaction | **既存** |
| 個人別parameter set | **一般的には既存** |
| パラメータ感度分析 | **既存** |
| 順位を変える最小parameter change | **概念として既存** |
| 患者＋家具・空間＋スタッフ位置をモニタリング | **近年既存** |
| 複数患者をリアルタイム監視 | **既存** |
| **上記文脈を「注意資源の患者間ランキング」というdecision problemとして、質問紙由来の明示的判断設定と統合** | **有望な応用・systems-level gap** |
| **判断設定×患者文脈から、どのrule/categoryがpairwise順位を形成・逆転させたかを一貫した方法で追跡** | **有望。ただし解析方法まで体系化する必要あり** |
| **ランキングが変化するparameter/context領域と頑健領域をモデル内部構造から明示** | **最も強い独自性候補。ただし一般MCDA sensitivityとの差を示す必要あり** |

特に最後の二点を強めるなら、単なるparameter sweepではなく、患者 \(p,q\) の差

\[
\Delta_{pq}(\boldsymbol{\theta},\mathbf{x})
=
R_p(\boldsymbol{\theta},\mathbf{x}_p)
-
R_q(\boldsymbol{\theta},\mathbf{x}_q)
\]

を中心に据えることを推奨する。

\[
\Delta_{pq}>0 \Rightarrow p\succ q,\qquad
\Delta_{pq}<0 \Rightarrow q\succ p
\]

であり、

\[
\Delta_{pq}=0
\]

が**順位遷移境界**になる。

さらに、

\[
\delta^{*}
=
\min_{\delta}
\|\delta\|
\quad
\mathrm{s.t.}\quad
\Delta_{pq}(\boldsymbol{\theta}+\delta)\le0
\]

を、AHP重みの非負・総和1、Fuzzy parameterの有効範囲等を制約として求めれば、「この順位はどの程度の判断設定差まで維持されるか」を定量化できる。

この最適化問題の考え方そのものは既存感度解析に類似するため新規性を主張すべきではないが、**階層AHP–Fuzzyモデルの内部要因・activated rulesと対応させて患者間優先順位の説明に使うこと**は、今回の研究の具体的な貢献にできる。citeturn21search1turn21search2

### 最も近い競合研究

特に重要な五件をまとめると次のようになる。

**Nahid, Ahad, Inoue, 2025** は応用面で最も危険な競合である。高齢者のmotion、care level、看護師との距離を使い、context-awareにalertを変更するからである。ただし、専門家の明示的な重要度をAHPで取得し、複数患者のranking sensitivityを解析する研究ではない。DOI: [10.3390/s25216560](https://doi.org/10.3390/s25216560)。citeturn21search4turn21search28

**Danial et al., 2025** は病院・多患者という意味で近い。複数病床を同時監視し、色付きpriority alertとstaff-presence recognitionまで含む。しかしpriority生成規則をdecision maker preferenceとしてモデル化・解析することが主目的ではない。DOI: [10.1186/s12984-025-01706-9](https://doi.org/10.1186/s12984-025-01706-9)。citeturn20view1

**Ribas & Pérez-Díaz, 2019** は数理構造面で非常に近い。FAHPによるindicator weightingと二段階Fuzzy reasoningを統合する。したがって「hierarchical AHP/Fuzzy risk assessment」の新規性を強く制限する。一方、動的な多主体監視やranking transition分析は対象外である。DOI: [10.1007/s12665-019-8526-3](https://doi.org/10.1007/s12665-019-8526-3)。citeturn18search1

**Ilbahar, Karaşan, Cebi, Kahraman, 2018** はAHP系＋FISをリスク評価に直接統合しており、「AHP＋Fuzzy risk model」という主張に対する代表的なprior artである。DOI: [10.1016/j.ssci.2017.10.025](https://doi.org/10.1016/j.ssci.2017.10.025)。citeturn18search16

**Angilella, Corrente, Greco, Słowiński, 2016** は理論面で最も強い競合である。hierarchical criteria、criteria interaction、pairwise preference information、不確実性、robustnessを一つのMCDA framework内で扱う。したがって「hierarchy＋interaction＋expert pairwise information」は本研究固有ではない。DOI: [10.1016/j.omega.2015.10.010](https://doi.org/10.1016/j.omega.2015.10.010)。citeturn21search2


## 主要先行研究の比較

下表の「本文」は今回、出版社等の全文ページまたはOA本文を確認できたもの、「抄録」は書誌・抄録・公開要約の範囲で確認したものを示す。「抄録」は、特に本文未確認であることを明示するための区分である。

| 研究・文献 | 対象課題 | 入力文脈 | 知識取得 | 評価モデル | 条件依存・相互作用 | 個別設定 | 説明・解析 | 評価 / 本研究との差 |
|---|---|---|---|---|---|---|---|---|
| **De Backere et al., 2015**, *Towards a social and context-aware multi-sensor fall detection and risk assessment platform*, Computers in Biology and Medicine. DOI [10.1016/j.compbiomed.2014.12.002](https://doi.org/10.1016/j.compbiomed.2014.12.002)〔抄録〕 citeturn12view0 | 高齢者の転倒検知・risk assessment | センサ、社会・環境context | ontology設計 | multisensor＋ontology | contextをsemanticに統合 | 環境適応可 | ontology由来の追跡性 | home中心。context統合は先行。本研究は多患者priorityと判断設定解析 |
| **Cho et al., 2019**, *Novel Approach to Inpatient Fall Risk Prediction and Its Cross-Site Validation Using Time-Variant Data*, JMIR. DOI [10.2196/11505](https://doi.org/10.2196/11505)〔抄録〕 citeturn9view3 | 院内転倒予測 | EHR、看護過程、薬剤等 | データ学習 | Bayesian network | 統計的依存 | 患者別入力 | network構造 | cross-site検証。事故確率寄りで、空間・staff coverageや判断選好は対象外 |
| **Lindberg et al., 2020**, *Identification of important factors in an inpatient fall risk prediction model…*, Int J Med Inform. DOI [10.1016/j.ijmedinf.2020.104272](https://doi.org/10.1016/j.ijmedinf.2020.104272)〔抄録〕 citeturn10search0 | 院内転倒予測 | EHR・管理データ | data-driven | ML | 学習された非線形関係 | 患者入力 | feature importance | 予測性能重視。本研究の明示的判断構造とは異なる |
| **Kim et al., 2025**, *Predicting In-Hospital Fall Risk Using Machine Learning With Real-Time Location System and Electronic Medical Records*, J Cachexia Sarcopenia Muscle. DOI [10.1002/jcsm.13713](https://doi.org/10.1002/jcsm.13713)〔抄録〕 citeturn10search5 | 院内転倒予測 | EHR＋RTLS活動 | data-driven | ML | implicit | 患者別 | feature-level説明 | 動的位置情報まで含むがexpert decision modelではない |
| **Mao et al., 2025**, *Risk prediction models for falls in hospitalized older patients: a systematic review and meta-analysis*, BMC Geriatrics. DOI [10.1186/s12877-025-05688-0](https://doi.org/10.1186/s12877-025-05688-0)〔本文〕 citeturn12view4 | 転倒予測研究レビュー | 主に患者・臨床情報 | ― | 多様 | モデル依存 | ― | ― | 現状の予測モデルのvalidation/bias問題を整理。本研究が予測精度研究ではないことを明確化する根拠 |
| **Gabriel et al., 2025**, *Continuous patient monitoring with AI: real-time analysis of video in hospital care settings*, Frontiers in Imaging. DOI [10.3389/fimag.2025.1547166](https://doi.org/10.3389/fimag.2025.1547166)〔本文〕 citeturn20view2 | 病院内連続monitoring | 患者、staff role、家具、motion、boundary | data-driven | computer vision | implicit spatial relation | 病院環境適応 | perception結果可視化 | 空間・staff contextは強い競合。ただしpriority decision structureを解析しない |
| **Danial et al., 2025**, *AI-based patient monitoring for fall prevention in stroke patients*, J NeuroEngineering Rehabil. DOI [10.1186/s12984-025-01706-9](https://doi.org/10.1186/s12984-025-01706-9)〔本文〕 citeturn20view1 | 急性期病棟fall monitoring | 多床、姿勢、動作、staff presence | system rules / AI | CV＋alert | 行動文脈 | 患者ごとon/off等 | alert reason | 複数bed・priority alertまで存在。本研究との差は判断criteriaの明示化とranking analysis |
| **Gervasi et al., 2025**, *Prevention of falls in hospitalized patients—evaluation of … Verso Vision*, Frontiers in Digital Health. DOI [10.3389/fdgth.2025.1548209](https://doi.org/10.3389/fdgth.2025.1548209)〔本文相当OA〕 citeturn20view3 | AI監視による転倒予防 | 患者行動、室内環境 | learned/system rules | CV alert | 状況別alert | risk患者に適用 | alert種別 | 実転倒削減を後向き評価。本研究は臨床効果を直接主張できない |
| **Nahid, Ahad, Inoue, 2025**, *Context-Aware Alerting in Elderly Care Facilities…*, Sensors. DOI [10.3390/s25216560](https://doi.org/10.3390/s25216560)〔抄録・公開要約〕 citeturn21search4turn21search28 | 介護施設のcontext-aware alert | skeletal motion、care level、nurse proximity | rules＋LLM | hybrid reasoning | **明示的context依存** | care level等 | rule/semantic reasoning | 応用上最も近い。AHP判断設定・patient ranking stabilityは扱わない |
| **Lavoie et al., 2020**, *Nursing handoffs and clinical judgments regarding patient risk of deterioration*, J Clin Nurs. DOI [10.1111/jocn.15409](https://doi.org/10.1111/jocn.15409)〔抄録〕 citeturn17search2 | 看護師の患者悪化リスク判断 | handoff情報、患者context | 直接rating | 人間判断 | 文脈依存 | 個人差あり | disagreement分析 | 74.4% disagreementを報告。判断異質性の根拠 |
| **Takase et al., 2024**, *Exploring nurses' clinical judgment concerning the relative importance of fall risk factors…*, Int J Nurs Stud. DOI [10.1016/j.ijnurstu.2024.104720](https://doi.org/10.1016/j.ijnurstu.2024.104720)〔抄録〕 citeturn17search5turn17search20 | 転倒要因の相対重要度 | fall risk factors | Q methodology | viewpoint抽出 | 人間判断内に反映 | 判断pattern | factor/viewpoint説明 | AHPとは異なるが、要因重要度の異質性を直接研究 |
| **Takase et al., 2025**, *Cognitive biases and contextual factors explaining variability in nurses' fall risk judgements*, Int J Nurs Stud Adv. DOI [10.1016/j.ijnsa.2025.100356](https://doi.org/10.1016/j.ijnsa.2025.100356)〔抄録・書誌〕 citeturn15search4 | 看護師の転倒判断ばらつき | 患者・context | 看護師回答 | human judgment analysis | contextual factors | 個人差 | bias分析 | 「個人差＝望ましい個別化」ではないことを示す重要競合 |
| **Iwamoto et al., 2026**, *Exploring Factors Influencing Nursing Task Prioritization…*, JMIR Human Factors. DOI [10.2196/89940](https://doi.org/10.2196/89940)〔公開本文情報〕 citeturn21search3turn21search7 | 看護タスク優先順位 | 緊急性、個別性、運用制約 | interview/質的分析 | thematic model | 強い文脈依存 | ― | qualitative explanation | 「risk」と「operational priority」の違いを支える |
| **Pecchia et al., 2011**, *Analytic Hierarchy Process for Examining Healthcare Professionals' Assessments of Risk Factors…*, Methods Inf Med. DOI [10.3414/ME10-01-0028](https://doi.org/10.3414/ME10-01-0028)〔抄録〕 citeturn17search0turn17search30 | 高齢者転倒リスク要因の重み付け | fall risk factors | **専門家AHP** | AHP | 原則加法的階層 | expert別可 | weightを説明可能 | 今回のAHP知識取得に直接のprior art |
| **Schmidt et al., 2015**, *Applying the Analytic Hierarchy Process in healthcare research…*, BMC Med Inform Decis Mak. DOI [10.1186/s12911-015-0234-7](https://doi.org/10.1186/s12911-015-0234-7)〔本文〕 citeturn16view3 | 医療AHPレビュー | ― | pairwise comparison | AHP | 主に階層 | group/individual | sensitivity等 | 121研究。AHP自体を新規性にできないこと、elicitation品質問題を示す |
| **Ilbahar et al., 2018**, *A novel approach to risk assessment … Pythagorean fuzzy AHP & fuzzy inference system*, Safety Science. DOI [10.1016/j.ssci.2017.10.025](https://doi.org/10.1016/j.ssci.2017.10.025)〔抄録・書誌〕 citeturn18search16 | 労働安全risk | 複数risk criteria | expert | **fuzzy AHP＋FIS** | FIS | expert設定 | rule/weight | AHP＋FIS risk assessmentの直接prior art |
| **Ribas & Pérez-Díaz, 2019**, *A multicriteria fuzzy approximate reasoning approach for risk assessment of dam safety*, Environmental Earth Sciences. DOI [10.1007/s12665-019-8526-3](https://doi.org/10.1007/s12665-019-8526-3)〔抄録〕 citeturn18search1 | ダムrisk | 多階層risk indicators | expert | **FAHP＋二段Fuzzy reasoning** | rule-based nonlinearity | expert設定 | hierarchical | 数理構造で特に近い。病棟・多患者・ranking dynamicsなし |
| **Mariadoss & Augustin, 2023**, *Enhanced Sugeno fuzzy inference system with fuzzy AHP…*, J King Saud Univ Computer & Information Sciences. DOI [10.1016/j.jksuci.2023.101659](https://doi.org/10.1016/j.jksuci.2023.101659)〔抄録〕 citeturn18search2turn18search18 | 妊娠中CVD診断 | 12 risk factors | clinicians | fuzzy AHP＋Sugeno FIS | nonlinear rules | expert setting | factor/rule | 医療でもAHP＋FISあり。目的は診断精度 |
| **Angilella et al., 2016**, *Robust Ordinal Regression and SMAA in multiple criteria hierarchy process for the Choquet integral preference model*, Omega. DOI [10.1016/j.omega.2015.10.010](https://doi.org/10.1016/j.omega.2015.10.010)〔抄録・書誌〕 citeturn21search2 | hierarchical MCDA | hierarchy＋criteria | pairwise preference | MCHP＋Choquet＋ROR/SMAA | **明示的interaction** | DM preference | robustness | 本研究の「hierarchy＋interaction＋preference」の理論的競合 |
| **Corrente et al., 2017**, *A robust ranking method extending ELECTRE III to hierarchy of interacting criteria…*, Omega. DOI [10.1016/j.omega.2016.11.008](https://doi.org/10.1016/j.omega.2016.11.008)〔抄録・書誌〕 citeturn15search18turn15search22 | robust ranking | hierarchical interacting criteria | DM information | ELECTRE III＋MCHP＋SMAA | **interaction** | uncertain weights | rank robustness | Fuzzy以外でもhierarchy・interaction・robust rankingが可能 |
| **Triantaphyllou & Sánchez, 1997**, *A Sensitivity Analysis Approach for Some Deterministic Multi-Criteria Decision-Making Methods*, Decision Sciences. DOI [10.1111/j.1540-5915.1997.tb01306.x](https://doi.org/10.1111/j.1540-5915.1997.tb01306.x)〔抄録・書誌〕 citeturn21search1 | MCDM ranking sensitivity | weights/performance | ― | deterministic MCDM | ― | weight変動 | **最小順位変化解析** | 「何をどれだけ変えれば順位が変わるか」は既存。本研究ではhybrid modelへの構造的適用が必要 |

この比較から、「既存研究では空間文脈を扱っていない」「AHPとFuzzyを組み合わせたことが新しい」「順位逆転を解析したことが新しい」という三つの主張は避けるべきである。

一方で、今回調べた病棟・介護モニタリング研究の中では、

> **専門家が表明したcriteria importanceとconditional rulesを明示的に保持しながら、複数患者の同時見守りpriorityを算出し、そのpriority rankingがどの判断設定・文脈で安定または変化するかを、decision model内部まで遡って体系的に解析する**

ことを主目的に据えた研究は確認できなかった。

ただしこれは**「存在しないことの証明」ではない**。とくにMCDA一般領域には非常に近い理論が多数あるため、新規性は「新しい数理原理」ではなく、「病棟見守りdecision systemとしての問題定式化＋既存要素の構造的統合＋ranking dynamicsの解析」にある、と限定するのが妥当である。


## 研究課題候補と評価設計

### 最有力案：文脈と判断設定による多患者見守り順位の形成・安定性

**研究課題**

> 複数患者を同時に見守る状況で、患者文脈・空間的文脈と、質問紙から明示化された判断設定は、どのように相互作用して見守り優先順位を形成し、その順位はどの条件で頑健または不安定になるのか。

これが最も推奨できる。

現場課題は、スタッフの注意が有限であり、危険イベントを検出するだけでは「複数患者のうち誰を先に見るか」というdecision problemが残ることである。現在のmonitoring研究はmulti-bed alertやnurse proximityまで扱い始めているが、priority形成規則の異質性を明示的なdecision modelとして解析することは中心課題ではない。citeturn20view1turn21search4

提案モデル固有の価値は、AHPやFuzzyそのものではなく、

\[
\text{stated importance}
\rightarrow
\text{category score}
\rightarrow
\text{context-dependent rule activation}
\rightarrow
\text{patient score}
\rightarrow
\text{cross-patient ranking}
\]

という因果ではなく**計算上のdecision trace**を保持できることにある。

検証する仮説としては、次の三つが適切である。

**H1:** 患者情報のみの加法モデルでは同順位または同順序となる状況でも、staff coverageやenvironmental contextを含む条件付き統合によって合理的に区別されるケースが存在する。

**H2:** スタッフ由来parameter sets間の差が常に順位差を生むのではなく、特定のcontext領域・rule activation領域でのみrank transitionが生じる。

**H3:** 多くの患者対では一定範囲のparameter variationに順位が頑健である一方、少数の境界ケースでは小さな判断差が順位を変え、その原因を少数のcategory/ruleへ帰属できる。

H2・H3は本人判断との一致を必要とせず、**モデル挙動についての検証可能な主張**になっている。

### 次点案：判断異質性の「影響する差」と「影響しない差」の識別

研究課題を、

> スタッフ間で表明される評価基準の違いのうち、どの違いが実際に患者優先順位へ影響し、どの違いは出力上吸収されるか

とする案である。

これは既存アンケート回答数が十分なら実現しやすい。staff parameter setをクラスタリングしてもよいが、クラスタリング自体を主貢献にする必要はない。

価値は、単なる「個人化」ではなく、**decision modelの実効的自由度を明らかにすること**である。例えばAHP重みがかなり違ってもFuzzy ruleで最終順位が同一なら、その差は実運用上あまり重要でない。逆に、ごく小さな一対比較差が特定条件でtop-1を変えるなら、そのcriterionは標準化や合意形成を優先すべき候補になる。

ただし査読上は「それは単なるsensitivity analysisではないか」と指摘されやすいため、単独案より最有力案の一部に組み込む方がよい。

### 第三案：事故リスクではなくcontext-aware monitoring priorityの定式化

これは概念面ではかなり強い。

従来のfall risk predictionを

\[
P(\text{fall}\mid\text{patient history})
\]

とするのに対し、本研究は

\[
\text{Priority}
=
f(
\text{patient state},
\text{behavior},
\text{environment},
\text{current staff coverage},
\text{stated decision criteria}
)
\]

を扱う、と明確に定義する。

この場合、**「転倒しやすさの予測」から「限られた見守り資源を現在どこへ向けるか」への問題転換**が研究の核になる。近年のcontext-aware monitoringや看護task prioritization研究とも論理的に接続できる。citeturn20view2turn21search3

ただし、「このpriorityが臨床的に正しい」と主張するには専門家rankingや将来の介入結果との比較が必要になる。今回それを行わないのであれば、

> **priorityを算出できることと、そのdecision behaviorを検証した**

までに留めなければならない。

総合的には、**最有力案と第三案を統合する構成が最も強い**。すなわち「context-aware multi-patient monitoring priority」をproblem formulationとし、その**ranking formation / transition / robustness**をresearch questionとする。

### 課題から結論までの追跡表

| 課題 | 原因 | アプローチ | 手法 | 評価 | 裏付けられる主張 | 結論できないこと |
|---|---|---|---|---|---|---|
| 患者属性だけでは現在の見守り必要性を区別できない | staff coverage、周囲物体、行動状況が変化する | patient＋spatial context統合 | 現行階層モデル | **patient-only vs full-context ablation** | 空間文脈が出力・順位に追加情報を与える | その順位が臨床的に正しい |
| 単純加重和では条件付き作用を表しにくい | criterion effectが他contextで変わる | within-category weightとcross-category ruleを分離 | AHP＋FIS | **weighted-sum / AHP-only baselineとの比較** | 条件付き相互作用が実際に異なるdecision surfaceを生む | FISが全代替法より優れている |
| questionnaire回答差がどこまで重要か不明 | parameter差≠output差 | 複数parameter setを同一患者群へ適用 | existing staff settings | Kendall/Spearman、top-k一致、pairwise flips | 表明評価基準の異質性がどの程度rankingへ伝播するか | 本人の実際の判断を再現すること |
| 順位変化の理由が不明 | 最終scoreのみでは原因が見えない | pairwise差をcategory/ruleまでtrace | \(\Delta_{pq}\)、activated-rule解析 | 代表的rank-transition case analysis | 「何が、なぜ順位を変えたか」をモデル内部から説明 | 説明が人間にとって最適・有用であること |
| 順位が偶然の微小差で決まる可能性 | elicitation uncertainty | parameter perturbation | constrained sensitivity analysis | minimal perturbation、rank-stability region | 順位の頑健・脆弱性 | 真のparameter distribution |
| Fuzzy ruleに意図しない挙動が入り得る | overlap・rule combination | structural audit | monotonicity / coverage / conflict checking | exhaustive or grid test | model内部の論理的一貫性 | 臨床的妥当性 |
| AHP回答が不整合となる | pairwise burden | consistency management | CR等 | respondent-wise consistency analysis | questionnaire parameterizationの内部品質 | external/behavioral validity |

この対応表の利点は、**各評価が特定の主張だけを支える**ことにある。

特に「モデルがそれらしい順位を出した」という総括評価だけでは不十分である。TSMC: Systemsを考えるなら、model property、context contribution、parameter sensitivity、ranking robustnessを別々に評価する方がsystems researchとして説得力が高い。IEEE SMCの公式scopeもsystems analysis/modelingとdecision makingを中心領域に含む。citeturn8search4


## 推奨する論文構成と最小限の追加評価

### 投稿前に必須と考える評価

現行モデルを大きく変更しない前提なら、最低限必要なのは以下である。

**構成概念の明確化が最優先である。** 「危険度」と呼ぶなら、その意味を「事故確率」から切り離す必要がある。スタッフ位置や観察可能性を含むなら、主出力を「context-aware monitoring priority」あるいは「attention priority index」と定義することを強く推奨する。事故予測モデルと同じvalidationをしていない以上、事故発生確率のような記述は避ける。citeturn12view4turn20view1

**patient-only / additive baselineとの比較は必須に近い。** 少なくとも、

\[
\text{patient factors only}
\]

\[
\text{patient + spatial factors, weighted additive}
\]

\[
\text{proposed hierarchical AHP + FIS}
\]

の三者を同一scenarioで比較し、「複雑なmodelにした結果、具体的にどの関係が新たに表現されるのか」を示すべきである。AHP＋FISという組合せ自体には先行例が多いため、この比較なしでは「なぜFuzzyが必要か」が弱い。citeturn18search16turn18search1

**固定患者集合に対するsystematic ranking sensitivity analysisも必須である。** 各staff parameter setについて、順位そのものだけでなくKendall's \(\tau\)、Spearman、top-1/top-k一致率、pairwise reversal率を報告し、さらに重要な患者対について\(\Delta_{pq}=0\)の境界または最小parameter perturbationを求める。これは既存MCDA sensitivityとの差を意識し、「単に順位逆転を発見した」ではなく、**hybrid modelのどの構造がrank transitionを生んだか**まで解析する必要がある。citeturn21search1turn21search13

**Fuzzy modelのstructural sanity checkも必要である。** 例えば他条件一定で「危険行動が強くなったのにpriorityが下がる」「スタッフcoverageが悪化したのにpriorityが下がる」ような挙動がないかを、全組合せまたは高密度gridで検査する。意図的な例外があればruleとして説明する。これは臨床妥当性の証明ではなく、**設計したdecision logicが設計意図と整合しているか**のverificationである。

**AHP・質問紙側の品質も示すべきである。** 回答者ごとのpairwise consistency、除外・補正基準、group aggregationをする場合の方法、欠損回答の扱いを報告する。医療AHP文献ではこの報告品質が問題になってきた。citeturn16view3

### 採録可能性を高める評価

最も効果が高い追加評価は、**一つだけinteraction-capableな競合モデルを置くこと**である。

例えば2-additive Choquet integral等を同一criteriaに適用し、

> Fuzzyにしかinteractionを表せない

ではなく、

> 両者ともinteractionを表せるが、今回のquestionnaire＋linguistic rule設計ではFuzzyがこのようなknowledge traceを保持する

と比較できれば、AngilellaらやCorrenteらを知る査読者からの反論に先回りできる。citeturn21search2turn15search18

ただし、新モデル開発まで行う必要はない。簡易baselineとして実装し、**表現可能性と説明単位の違い**を示せれば十分である。

次に有効なのは、質問紙parameter uncertaintyを入れたrank robustnessである。例えば各AHP重みを回答誤差を想定して摂動し、

\[
P(p\text{ is ranked first})
\]

やpairwise ranking probabilityをMonte Carloで求める。SMAAやrobust MCDAには既に類似思想があるので新規性にはしないが、「この患者はどのstaff settingでもほぼ最上位」「この二人は設定に非常に敏感」と示せる。citeturn21search2

さらに、代表的な数scenarioについて、

> 患者AがBを上回る  
> → staff distance ruleが発火  
> → category Xの差が増幅  
> → AHP weight差によって境界を越えた

という**pairwise explanation**を作るとよい。これは単なるfeature importanceより、今回の明示的model architectureを活用できる。

既存アンケート参加者へ追加負担をかけずに可能なら、「質問紙から得たparameter set」と「全員共通の中央値・group consensus set」を比較することも有益である。これにより、個人化を推奨するのではなく、**個人差を無視した共通設定がどのscenarioで結果を変えるか**を議論できる。

### 今回は省略してよい評価

事故件数減少の前向き臨床試験は、今回の主張には必要ない。Gervasiらのような実転倒率低下を評価する研究とは研究段階が異なるため、「事故予防効果を示した」と言わなければよい。citeturn20view3

大規模な新規staff surveyも必須ではない。既存回答が複数人あり、parameter variationを研究できるなら、今回はそのデータを用いたsystems analysisとして成立し得る。

また、スタッフ本人のactual rankingとの高一致率を必須評価にする必要もない。ただし未実施の場合、

> model represents staff judgment

ではなく、

> model is parameterized by stated evaluation criteria

と限定する。

スタッフの内省促進、trust、explanation usefulness等のhuman-subject evaluationも、今回の中心RQでなければ次段階に回してよい。

### 推奨する論文のストーリー

最も通しやすい構成は次の流れである。

冒頭では、転倒予測モデルの不足を単純に攻撃しない。むしろ、

> prediction/detection technologies increasingly capture dynamic patient and spatial context, including staff presence and proximity, but monitoring multiple patients still raises a decision-level question: how should heterogeneous contextual factors be transformed into an interpretable relative attention priority?

と置く。実際に2025年には病院CVやnurse-proximityを含む研究が存在するため、この書き方なら先行研究に正面から整合する。citeturn20view2turn21search4

次に、clinical predictionではなく**multi-patient decision support**の問題として定式化する。

その上でMethodでは、

\[
\text{patient/environment/staff context}
\rightarrow
\text{hierarchical factor representation}
\rightarrow
\text{within-category stated importance}
\rightarrow
\text{cross-category conditional logic}
\rightarrow
\text{priority ranking}
\]

を提示する。

AHPとFISは「新規アルゴリズム」としてではなく、**異なる種類の判断知識を分けて保持する実装手段**と位置付ける。AHP＋FIS既存研究を正面から引用し、「本研究のcontributionはその組合せ自体ではない」と明記した方が、むしろ査読者の信頼を得やすい。citeturn18search16turn18search1

Resultsの中心をaccuracyではなく、

\[
\text{context contribution}
\rightarrow
\text{interaction effect}
\rightarrow
\text{ranking difference}
\rightarrow
\text{rank-transition boundary}
\rightarrow
\text{robustness}
\rightarrow
\text{structural explanation}
\]

に置く。

Discussionでは、個人差の存在を「個別最適化の正当化」とせず、

> 個人・スタッフ間の表明criteria差を、decision systemのrobustnessを検査するための複数parameterizationとして利用した

と説明する。これは患者安全上も、Takaseらが示すbiasの可能性とも整合する。citeturn15search4

最後に限界として、少なくとも

**質問紙回答は実判断ではない、priorityはclinical ground truthではない、事故予防効果は検証していない、Fuzzy/AHP構成自体は既存、実運用には標準化・臨床validationが必要**

と明示する。

### 最終的な新規性判定

今回の問いに対する最終回答は次のようになる。

> **病棟共有空間における患者・空間的文脈とアンケート由来の判断基準を階層的に統合すること自体は、新しい方法論ではない。AHPによる専門家重要度取得、Fuzzyによる条件付きリスク統合、criteria hierarchy、criteria interaction、context-aware patient monitoring、ranking sensitivity analysisのいずれにも十分な先行研究がある。** citeturn17search0turn18search16turn21search2turn20view2turn21search1
>
> **一方、複数患者を同時に見守るというsystems decision problemにおいて、患者・環境・現在のstaff coverageを「見守り優先度」へ変換する判断構造を明示し、質問紙で表明されたcriteria/rulesの差が、どのcontextで患者間rankingを維持・変化させるのかを、rank-transition boundary、robustness、activated rulesまで遡って説明する、という統合的な研究課題には差別化余地がある。**
>
> **したがって新規性の中心は「AHP＋Fuzzy」ではなく、「context-aware multi-patient prioritizationという問題定式化」と「判断設定×文脈→ranking dynamicsを説明・検証する分析枠組み」に置くべきである。**

逆に、現行論文が

> アンケートでAHP重みを作った → Fuzzy ruleを設定した → 患者スコアを計算した → パラメータを数種類変えたら順位が逆転した

までで終わるなら、**IEEE Transactions on Systems, Man, and Cybernetics: Systemsに対しては、既存手法の特定応用への組合せとして評価される危険がかなり高い**。同誌がsystems modeling、analysis、decision makingを対象にしていることを考えると、少なくとも「なぜ順位が変わるか」「どこまで変わらないか」「単純モデルでは何を表せないか」を一般化可能な形で示す必要がある。citeturn8search4

したがって、**案Cは捨てる必要はないが、「判断知識を表現できた」から一段進めて、「多患者見守りdecision systemにおけるpriority形成とrank robustnessを解明する研究」に修正することを推奨する**。この修正なら、大規模な新規臨床調査やモデル全面改修を要求せず、現在の階層モデル・既存アンケート・既存患者scenarioをかなりそのまま活用できる。同時に、「第三の目」という最終目標に対しても、**センシングされた状況を、限られた人間の注意へどう変換するかというdecision layerを今回確立する**という明確な論理的接続を与えられる。