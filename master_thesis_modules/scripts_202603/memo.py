import copy
import math
from itertools import product

import pandas as pd
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


# ============================================================
# 0. 初期状態
# ============================================================

# ------------------------------------------------------------
# 部屋定義
# 5部屋を2次元座標に割り当てる
# ------------------------------------------------------------
ROOMS = {
    "room_00": (0, 0),
    "room_01": (1, 0),
    "room_10": (0, 1),
    "room_11": (1, 1),
    "room_20": (2, 0),
}
POS_TO_ROOM = {v: k for k, v in ROOMS.items()}

# ------------------------------------------------------------
# 隣接関係
# 「移動または見通し可能」を1とする
# ------------------------------------------------------------
room_names = ["room_00", "room_01", "room_10", "room_11", "room_20"]
room_to_idx = {room: i for i, room in enumerate(room_names)}

adj_mtx = [
    [1, 1, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 1],
]

# ------------------------------------------------------------
# エージェント属性
# ------------------------------------------------------------
agents_info = {
    0: {"attr": "patient", "attr_details": "dementia"},
    1: {"attr": "patient", "attr_details": "ecmo"},
    2: {"attr": "staff", "attr_details": "nurse"},
    3: {"attr": "robot", "attr_details": "monitoring"},
}

# ------------------------------------------------------------
# 観測
# ------------------------------------------------------------
observations = {
    0: {"location": "room_11", "velocity": "slow"},
    1: {"location": "room_01", "velocity": "stop"},
    2: {"location": "room_00", "velocity": "stop"},
    3: {"location": "room_20", "velocity": "stop"},
}

# ------------------------------------------------------------
# タスク推定結果
# 行: タスク, 列: エージェント
# ------------------------------------------------------------
task_estimation_result = [
    [0.2, 0.8, 0.1, 0.0],  # sleep
    [0.8, 0.2, 0.0, 0.0],  # wander
    [0.0, 0.0, 0.0, 0.1],  # wait
    [0.0, 0.0, 0.3, 0.0],  # monitor
    [0.0, 0.0, 0.6, 0.0],  # pc
]
task_df = pd.DataFrame(
    task_estimation_result,
    columns=[f"agent_{i}" for i in range(len(agents_info))],
    index=["sleep", "wander", "wait", "monitor", "pc"]
)


# ============================================================
# 補助関数
# ============================================================

def clamp(x, low=0.0, high=1.0):
    """値を指定範囲に収める。"""
    return max(low, min(high, x))


def is_adjacent(room_a, room_b):
    """2部屋が隣接・見通し可能かを返す。"""
    ia = room_to_idx[room_a]
    ib = room_to_idx[room_b]
    return adj_mtx[ia][ib] == 1


def build_initial_belief(agents_info, observations, task_df):
    """
    観測とタスク推定から belief を構築する。
    belief は「真の状態の確率分布」の簡略表現として使う。
    今回は患者ごとに、平均的なリスク推定値を持つ。
    """
    belief = {}

    for aid, info in agents_info.items():
        if info["attr"] != "patient":
            continue

        location = observations[aid]["location"]
        velocity = observations[aid]["velocity"]

        wander_prob = task_df.loc["wander", f"agent_{aid}"]
        sleep_prob = task_df.loc["sleep", f"agent_{aid}"]

        # 属性ごとの基本値
        if info["attr_details"] == "dementia":
            base_risk = 0.45
            need_monitor_prob = 0.75
            needs_continuous_staff_attention = False
        elif info["attr_details"] == "ecmo":
            base_risk = 0.55
            need_monitor_prob = 0.85
            needs_continuous_staff_attention = True
        else:
            base_risk = 0.30
            need_monitor_prob = 0.50
            needs_continuous_staff_attention = False

        velocity_bonus = {
            "stop": 0.00,
            "slow": 0.10,
            "fast": 0.20,
        }.get(velocity, 0.0)

        risk_score = base_risk + 0.35 * wander_prob - 0.15 * sleep_prob + velocity_bonus
        risk_score = clamp(risk_score)

        belief[aid] = {
            "location": location,
            "velocity": velocity,
            "risk_score": risk_score,
            "unattended_steps": 1,
            "need_monitor_prob": need_monitor_prob,
            "needs_continuous_staff_attention": needs_continuous_staff_attention,
        }

    return belief


# ============================================================
# Mesa エージェント定義
# ============================================================

class BaseWardAgent(Agent):
    """病棟内エージェントの共通基底クラス。"""

    def __init__(self, model, agent_id, attr_details):
        super().__init__(model)
        self.agent_id = agent_id
        self.attr_details = attr_details

    @property
    def room_name(self):
        """現在位置を部屋名で返す。"""
        return POS_TO_ROOM[self.pos]


class PatientAgent(BaseWardAgent):
    """
    患者エージェント。
    Mesa上の「状態」を簡易表現する。
    """

    def __init__(self, model, agent_id, attr_details, init_room, velocity, belief_state):
        super().__init__(model, agent_id, attr_details)
        self.velocity = velocity

        # belief由来の初期状態
        self.risk_score = belief_state["risk_score"]
        self.unattended_steps = belief_state["unattended_steps"]
        self.need_monitor_prob = belief_state["need_monitor_prob"]
        self.needs_continuous_staff_attention = belief_state["needs_continuous_staff_attention"]

        # このステップでどんな介入を受けるか
        self.staff_direct_attention = False      # スタッフが直接見に行く
        self.staff_remote_attention = False      # スタッフがロボット中継を見る
        self.robot_attention = False             # ロボットが見に行く
        self.robot_monitor_link = False          # ロボット映像とスタッフ通知が連携している

        self.model.grid.place_agent(self, ROOMS[init_room])

    def apply_natural_dynamics(self):
        """
        介入がない場合の自然な時間経過。
        """
        self.unattended_steps += 1

        # 見守り不足でリスク上昇
        self.risk_score += 0.05 * self.need_monitor_prob

        # 属性ごとの自然な悪化
        if self.attr_details == "dementia":
            self.risk_score += 0.03
        elif self.attr_details == "ecmo":
            self.risk_score += 0.02

        # ECMO患者は、人の継続監視が重要。
        # この患者を誰も直接見ていない場合、やや追加悪化させる。
        if self.needs_continuous_staff_attention and not self.staff_direct_attention:
            self.risk_score += 0.05

        self.risk_score = clamp(self.risk_score)

    def apply_interventions(self):
        """
        各介入の効果を反映する。
        """
        # スタッフが直接見に行くと一番強い
        if self.staff_direct_attention:
            self.risk_score -= 0.28
            self.unattended_steps = max(0, self.unattended_steps - 2)

        # スタッフがロボット中継を見るだけなら少し弱い
        if self.staff_remote_attention:
            self.risk_score -= 0.10
            self.unattended_steps = max(0, self.unattended_steps - 1)

        # ロボットが見に行く効果
        if self.robot_attention:
            self.risk_score -= 0.12
            self.unattended_steps = max(0, self.unattended_steps - 1)

        # スタッフ遠隔確認 + ロボット実観測 の相乗効果
        if self.staff_remote_attention and self.robot_attention and self.robot_monitor_link:
            self.risk_score -= 0.08

        self.risk_score = clamp(self.risk_score)

    def reset_action_flags(self):
        """各ステップ後に介入フラグをリセットする。"""
        self.staff_direct_attention = False
        self.staff_remote_attention = False
        self.robot_attention = False
        self.robot_monitor_link = False

    def step(self):
        """
        患者1人分の1ステップ更新。
        """
        self.apply_natural_dynamics()
        self.apply_interventions()
        self.reset_action_flags()


class StaffAgent(BaseWardAgent):
    """
    スタッフエージェント。
    今回は1人の看護師を想定し、
    「誰を主に見ているか」を持たせる。
    """

    def __init__(self, model, agent_id, attr_details, init_room, velocity, primary_patient_id):
        super().__init__(model, agent_id, attr_details)
        self.velocity = velocity

        # 主担当患者
        self.primary_patient_id = primary_patient_id

        # このステップで直接見に行く対象
        self.direct_target_patient_id = None

        # このステップでロボット中継を見る対象
        self.remote_target_patient_id = None

        # 忙しさの簡易指標
        self.workload = 0.0

        self.model.grid.place_agent(self, ROOMS[init_room])

    def assign_direct_check(self, patient_id):
        """直接見に行く対象を設定。"""
        self.direct_target_patient_id = patient_id

    def assign_remote_monitor(self, patient_id):
        """ロボット中継を見る対象を設定。"""
        self.remote_target_patient_id = patient_id

    def apply_staff_attention(self):
        """
        スタッフ行動を患者側へ反映する。
        ここで重要なのは「しわ寄せ」を起こすこと。
        """
        # スタッフが直接別患者を見に行くと、
        # 主担当患者（ここではECMO患者）の見守りが薄くなる。
        if self.direct_target_patient_id is not None:
            target_id = self.direct_target_patient_id
            self.model.patient_agents[target_id].staff_direct_attention = True

            # 今見ている主担当患者から離れるなら、その患者のリスクを悪化させる
            if self.primary_patient_id is not None and self.primary_patient_id != target_id:
                primary_patient = self.model.patient_agents[self.primary_patient_id]

                # 「見守りが薄くなる」ことによる悪化
                primary_patient.risk_score = clamp(primary_patient.risk_score + 0.18)
                primary_patient.unattended_steps += 1

            # 直接見に行くのは負荷が高い
            self.workload += 1.0

        # ロボット中継を見るだけなら、
        # スタッフは主担当患者の近くに残れるので、しわ寄せは小さい
        if self.remote_target_patient_id is not None:
            target_id = self.remote_target_patient_id
            self.model.patient_agents[target_id].staff_remote_attention = True

            # 遠隔確認も少し負荷はある
            self.workload += 0.3

        # 何もしなければ workload はあまり増えない
        self.workload = min(self.workload, 2.0)

    def reset_assignments(self):
        """このステップの指示をリセットする。"""
        self.direct_target_patient_id = None
        self.remote_target_patient_id = None

    def step(self):
        """
        スタッフの1ステップ更新。
        今回は割当結果を患者状態へ反映するのみ。
        """
        self.apply_staff_attention()
        self.reset_assignments()

        # workload は少し自然減衰
        self.workload = max(0.0, self.workload - 0.2)


class RobotAgent(BaseWardAgent):
    """
    ロボットエージェント。
    どの患者を見に行くかを外部から指示される。
    """

    def __init__(self, model, agent_id, attr_details, init_room, velocity):
        super().__init__(model, agent_id, attr_details)
        self.velocity = velocity
        self.target_patient_id = None
        self.model.grid.place_agent(self, ROOMS[init_room])

    def move_to_patient_room(self, patient_agent):
        """
        単純化のため、1ステップで対象患者の部屋へ移動する。
        """
        self.model.grid.move_agent(self, patient_agent.pos)

    def apply_robot_attention(self):
        """
        ロボットが対象患者を確認する効果を患者側へ反映。
        """
        if self.target_patient_id is not None:
            patient = self.model.patient_agents[self.target_patient_id]
            patient.robot_attention = True

    def reset_assignment(self):
        """このステップの指示をリセットする。"""
        self.target_patient_id = None

    def step(self):
        """
        ロボットの1ステップ更新。
        """
        if self.target_patient_id is not None:
            patient = self.model.patient_agents[self.target_patient_id]
            self.move_to_patient_room(patient)
            self.apply_robot_attention()

        self.reset_assignment()


# ============================================================
# Mesa モデル定義
# ============================================================

class WardModel(Model):
    """
    病棟全体モデル。
    ここに POMDP 的な
      - belief
      - action
      - reward
    を載せる。
    """

    def __init__(self, agents_info, observations, task_df):
        super().__init__()

        # Grid
        self.grid = MultiGrid(width=3, height=2, torus=False)

        # belief 初期化
        self.belief = build_initial_belief(agents_info, observations, task_df)

        # エージェント辞書
        self.patient_agents = {}
        self.staff_agents = {}
        self.robot_agents = {}

        # 患者生成
        for aid, info in agents_info.items():
            if info["attr"] != "patient":
                continue

            init_room = observations[aid]["location"]
            velocity = observations[aid]["velocity"]

            patient = PatientAgent(
                model=self,
                agent_id=aid,
                attr_details=info["attr_details"],
                init_room=init_room,
                velocity=velocity,
                belief_state=self.belief[aid],
            )
            self.patient_agents[aid] = patient

        # スタッフ生成
        # 今回は看護師1人が ECMO患者(agent 1) を主担当として見る設定
        for aid, info in agents_info.items():
            if info["attr"] != "staff":
                continue

            init_room = observations[aid]["location"]
            velocity = observations[aid]["velocity"]

            staff = StaffAgent(
                model=self,
                agent_id=aid,
                attr_details=info["attr_details"],
                init_room=init_room,
                velocity=velocity,
                primary_patient_id=1,  # ECMO患者を主担当とする
            )
            self.staff_agents[aid] = staff

        # ロボット生成
        for aid, info in agents_info.items():
            if info["attr"] != "robot":
                continue

            init_room = observations[aid]["location"]
            velocity = observations[aid]["velocity"]

            robot = RobotAgent(
                model=self,
                agent_id=aid,
                attr_details=info["attr_details"],
                init_room=init_room,
                velocity=velocity,
            )
            self.robot_agents[aid] = robot

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "mean_patient_risk": self.get_mean_patient_risk,
                "max_patient_risk": self.get_max_patient_risk,
                "ecmo_risk": self.get_ecmo_risk,
                "dementia_risk": self.get_dementia_risk,
                "staff_workload": self.get_staff_workload,
            }
        )

    # --------------------------------------------------------
    # 指標取得
    # --------------------------------------------------------
    def get_mean_patient_risk(self):
        risks = [p.risk_score for p in self.patient_agents.values()]
        return sum(risks) / len(risks)

    def get_max_patient_risk(self):
        risks = [p.risk_score for p in self.patient_agents.values()]
        return max(risks)

    def get_ecmo_risk(self):
        return self.patient_agents[1].risk_score

    def get_dementia_risk(self):
        return self.patient_agents[0].risk_score

    def get_staff_workload(self):
        staff = next(iter(self.staff_agents.values()))
        return staff.workload

    # --------------------------------------------------------
    # belief同期
    # --------------------------------------------------------
    def sync_belief_from_agents(self):
        """
        エージェント状態を belief に書き戻す。
        """
        for pid, patient in self.patient_agents.items():
            self.belief[pid]["location"] = patient.room_name
            self.belief[pid]["risk_score"] = patient.risk_score
            self.belief[pid]["unattended_steps"] = patient.unattended_steps

    # --------------------------------------------------------
    # 介入候補生成
    # --------------------------------------------------------
    def generate_action_candidates(self):
        """
        介入候補を生成する。
        スタッフ:
          - do_nothing
          - direct_check_patient_X
          - watch_robot_monitor_for_patient_X
        ロボット:
          - do_nothing
          - check_patient_X
        """
        patient_ids = list(self.patient_agents.keys())

        staff_actions = ["do_nothing_staff"]
        robot_actions = ["do_nothing_robot"]

        for pid in patient_ids:
            staff_actions.append(f"notify_staff_to_check_agent_{pid}")
            staff_actions.append(f"notify_staff_to_watch_robot_monitor_for_agent_{pid}")
            robot_actions.append(f"send_robot_to_check_agent_{pid}")

        joint_actions = list(product(staff_actions, robot_actions))

        filtered = []
        for staff_act, robot_act in joint_actions:
            valid = True

            # ロボット中継を見る通知なのに、
            # ロボットが別患者を見に行くのは不自然なので除外
            if "watch_robot_monitor_for_agent_" in staff_act and robot_act.startswith("send_robot_to_check_agent_"):
                pid_staff = int(staff_act.split("_")[-1])
                pid_robot = int(robot_act.split("_")[-1])
                if pid_staff != pid_robot:
                    valid = False

            if valid:
                filtered.append({
                    "staff_action": staff_act,
                    "robot_action": robot_act,
                })

        return filtered

    # --------------------------------------------------------
    # 行動適用
    # --------------------------------------------------------
    def apply_action(self, action):
        """
        介入候補をモデルに適用する。
        """
        staff_action = action["staff_action"]
        robot_action = action["robot_action"]

        staff = next(iter(self.staff_agents.values()))
        robot = next(iter(self.robot_agents.values()))

        # スタッフ行動の割当
        if staff_action.startswith("notify_staff_to_check_agent_"):
            pid = int(staff_action.split("_")[-1])
            staff.assign_direct_check(pid)

        elif staff_action.startswith("notify_staff_to_watch_robot_monitor_for_agent_"):
            pid = int(staff_action.split("_")[-1])
            staff.assign_remote_monitor(pid)
            self.patient_agents[pid].robot_monitor_link = True

        # ロボット行動の割当
        if robot_action.startswith("send_robot_to_check_agent_"):
            pid = int(robot_action.split("_")[-1])
            robot.target_patient_id = pid

    # --------------------------------------------------------
    # 1ステップ進行
    # --------------------------------------------------------
    def step(self):
        """
        更新順序:
          1. スタッフ割当を患者へ反映
          2. ロボット割当を患者へ反映
          3. 患者状態更新
          4. belief同期
        """
        # スタッフ行動適用
        for staff in self.staff_agents.values():
            staff.step()

        # ロボット行動適用
        for robot in self.robot_agents.values():
            robot.step()

        # 患者状態更新
        for patient in self.patient_agents.values():
            patient.step()

        # belief同期
        self.sync_belief_from_agents()

        # ログ収集
        self.datacollector.collect(self)

    # --------------------------------------------------------
    # 報酬関数
    # --------------------------------------------------------
    def reward_function(self, action):
        """
        病棟全体としての行動評価。
        ECMO患者の見守り悪化を強く嫌うように設計する。
        """
        reward = 0.0

        for pid, state in self.belief.items():
            risk = state["risk_score"]
            unattended = state["unattended_steps"]
            need_monitor = state["need_monitor_prob"]
            needs_continuous = state["needs_continuous_staff_attention"]

            # 基本の安全性評価
            reward -= 8.0 * risk
            reward -= 1.5 * unattended * need_monitor

            # 高リスク閾値
            if risk > 0.80:
                reward -= 15.0

            # 人の継続監視が重要な患者（ECMO）を特に重く扱う
            if needs_continuous:
                reward -= 6.0 * risk
                reward -= 2.0 * unattended

        # スタッフ負荷コスト
        staff = next(iter(self.staff_agents.values()))
        reward -= 3.0 * staff.workload

        # 行動コスト
        if action["staff_action"] != "do_nothing_staff":
            reward -= 1.5

        if action["robot_action"] != "do_nothing_robot":
            reward -= 1.0

        # スタッフをECMOから引き剥がして別患者を直接見に行かせる行為に追加ペナルティ
        if action["staff_action"].startswith("notify_staff_to_check_agent_"):
            target_pid = int(action["staff_action"].split("_")[-1])
            if target_pid != staff.primary_patient_id:
                reward -= 4.0

        # ロボットで認知症患者を見に行きつつ、スタッフはECMO担当を維持する構図に小ボーナス
        if (
            action["staff_action"] == "do_nothing_staff"
            and action["robot_action"] == "send_robot_to_check_agent_0"
        ):
            reward += 3.0

        # スタッフがロボット中継で認知症患者を確認し、
        # ロボットも同患者を見に行くなら、現場的に自然な補助手段として少し加点
        if (
            action["staff_action"] == "notify_staff_to_watch_robot_monitor_for_agent_0"
            and action["robot_action"] == "send_robot_to_check_agent_0"
        ):
            reward += 2.0

        return reward

    # --------------------------------------------------------
    # シミュレーション評価
    # --------------------------------------------------------
    def simulate_action_sequence(self, action_sequence, gamma=0.90):
        """
        行動列を適用したときの割引累積報酬を返す。
        """
        sim_model = copy.deepcopy(self)
        total_return = 0.0

        for t, action in enumerate(action_sequence):
            sim_model.apply_action(action)
            sim_model.step()
            reward = sim_model.reward_function(action)
            total_return += (gamma ** t) * reward

        return total_return, sim_model

    # --------------------------------------------------------
    # 近似POMDPプランニング
    # --------------------------------------------------------
    def plan_with_lookahead(self, horizon=2, gamma=0.90):
        """
        候補行動を horizon ステップ先まで総当たり評価し、
        最初の1手として最善手を返す。
        """
        action_candidates = self.generate_action_candidates()

        best_first_action = None
        best_value = -math.inf
        best_sequence = None

        for seq in product(action_candidates, repeat=horizon):
            total_return, _ = self.simulate_action_sequence(seq, gamma=gamma)
            if total_return > best_value:
                best_value = total_return
                best_first_action = seq[0]
                best_sequence = seq

        return best_first_action, best_value, best_sequence


# ============================================================
# 実行例
# ============================================================

if __name__ == "__main__":
    model = WardModel(
        agents_info=agents_info,
        observations=observations,
        task_df=task_df,
    )

    print("===== 初期 belief =====")
    for pid, state in model.belief.items():
        print(f"patient {pid}: {state}")

    print("\n===== 初期スタッフ状態 =====")
    staff = next(iter(model.staff_agents.values()))
    print(f"primary_patient_id = {staff.primary_patient_id} (ECMO患者を主担当)")
    print(f"workload = {staff.workload}")

    print("\n===== 介入候補 =====")
    action_candidates = model.generate_action_candidates()
    for i, action in enumerate(action_candidates):
        print(i, action)

    best_action, best_value, best_sequence = model.plan_with_lookahead(
        horizon=2,
        gamma=0.90,
    )

    print("\n===== 最適な最初の行動 =====")
    print(best_action)
    print("expected value =", best_value)

    print("\n===== 最良シーケンス =====")
    for t, action in enumerate(best_sequence):
        print(f"step {t}: {action}")

    # 最初の1手を適用
    model.apply_action(best_action)
    model.step()

    print("\n===== 1ステップ後の belief =====")
    for pid, state in model.belief.items():
        print(f"patient {pid}: {state}")

    print("\n===== 1ステップ後のスタッフ状態 =====")
    staff = next(iter(model.staff_agents.values()))
    print(f"workload = {staff.workload}")
    print(f"primary_patient_id = {staff.primary_patient_id}")

    print("\n===== 収集ログ =====")
    print(model.datacollector.get_model_vars_dataframe())