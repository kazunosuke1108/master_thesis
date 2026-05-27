from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import mesa
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid


# =========================
# ユーティリティ
# =========================
Position = tuple[int, int]


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_one_step_toward(src: Position, dst: Position) -> Position:
    """src から dst にマンハッタン距離で 1 マスだけ近づく"""
    x, y = src
    dx, dy = dst

    if x < dx:
        return (x + 1, y)
    if x > dx:
        return (x - 1, y)
    if y < dy:
        return (x, y + 1)
    if y > dy:
        return (x, y - 1)
    return src


# =========================
# エージェント
# =========================
class WardAgent(mesa.Agent):
    def __init__(self, model: "WardModel"):
        super().__init__(model)

    @property
    def pos2d(self) -> Position:
        if self.pos is None:
            raise ValueError("Agent position is None.")
        return self.pos


class PatientAgent(WardAgent):
    def __init__(
        self,
        model: "WardModel",
        name: str,
        start_pos: Position,
        wandering_prob: float = 0.6,
    ):
        super().__init__(model)
        self.name = name
        self.start_pos = start_pos
        self.wandering_prob = wandering_prob
        self.task = "wandering"   # wandering / staying
        self.unobserved_steps = 0
        self.fall_risk = 0.0

    def step(self) -> None:
        # 患者の簡易行動:
        # wandering 中なら確率的に隣接マスへ移動
        if self.task == "wandering" and self.random.random() < self.wandering_prob:
            neighbors = self.model.grid.get_neighborhood(
                self.pos2d,
                moore=False,          # 上下左右のみ
                include_center=False
            )
            new_pos = self.random.choice(neighbors)
            self.model.grid.move_agent(self, new_pos)

        # 見守られているか判定
        observed = self.model.is_patient_observed(self)

        if observed:
            self.unobserved_steps = 0
        else:
            self.unobserved_steps += 1

        # リスク更新
        # 発想:
        # - wandering なら上がりやすい
        # - 見守られていない時間が長いほど上がる
        # - ロボットが近いと少し下がる
        base_increase = 0.12 if self.task == "wandering" else 0.04
        unobserved_bonus = 0.08 * self.unobserved_steps

        robot = self.model.robot
        robot_distance = manhattan(self.pos2d, robot.pos2d)
        robot_relief = 0.18 if robot_distance <= 1 else 0.0

        delta = base_increase + unobserved_bonus - robot_relief
        self.fall_risk = max(0.0, min(1.0, self.fall_risk + delta))

        # ロボットが近くで声かけしていれば wandering を抑制しやすい
        if robot.pos2d == self.pos2d or robot_distance == 1:
            if robot.speaking and self.random.random() < 0.5:
                self.task = "staying"


class NurseAgent(WardAgent):
    def __init__(
        self,
        model: "WardModel",
        name: str,
        start_pos: Position,
        station_pos: Position,
    ):
        super().__init__(model)
        self.name = name
        self.start_pos = start_pos
        self.station_pos = station_pos
        self.task = "recording"   # recording / responding

    def step(self) -> None:
        # 一番リスクが高い患者が閾値超えなら対応しに行く
        target = max(self.model.patients, key=lambda p: p.fall_risk)

        if target.fall_risk >= self.model.nurse_dispatch_threshold:
            self.task = "responding"
            new_pos = move_one_step_toward(self.pos2d, target.pos2d)
            self.model.grid.move_agent(self, new_pos)
        else:
            self.task = "recording"
            if self.pos2d != self.station_pos:
                new_pos = move_one_step_toward(self.pos2d, self.station_pos)
                self.model.grid.move_agent(self, new_pos)


class RobotAgent(WardAgent):
    def __init__(
        self,
        model: "WardModel",
        name: str,
        start_pos: Position,
    ):
        super().__init__(model)
        self.name = name
        self.start_pos = start_pos
        self.target_pos: Optional[Position] = None
        self.speaking: bool = False

    def step(self) -> None:
        # 毎ステップ最初に policy が次行動を決める
        self.model.robot_policy(self.model, self)

        # target_pos があれば 1 マス移動
        if self.target_pos is not None and self.target_pos != self.pos2d:
            new_pos = move_one_step_toward(self.pos2d, self.target_pos)
            self.model.grid.move_agent(self, new_pos)


# =========================
# モデル
# =========================
RobotPolicy = Callable[["WardModel", RobotAgent], None]


class WardModel(mesa.Model):
    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        seed: Optional[int] = 42,
        robot_policy: Optional[RobotPolicy] = None,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)

        self.nurse_dispatch_threshold = 0.65
        self.robot_policy = robot_policy or stay_policy

        # 位置の例
        self.nurse_station_pos = (0, 0)

        # エージェント生成
        self.patient_a = PatientAgent(
            model=self,
            name="PatientA",
            start_pos=(3, 3),
            wandering_prob=0.8,
        )
        self.patient_b = PatientAgent(
            model=self,
            name="PatientB",
            start_pos=(4, 1),
            wandering_prob=0.2,
        )
        self.patient_b.task = "staying"

        self.nurse = NurseAgent(
            model=self,
            name="NurseC",
            start_pos=self.nurse_station_pos,
            station_pos=self.nurse_station_pos,
        )

        self.robot = RobotAgent(
            model=self,
            name="RobotD",
            start_pos=(1, 4),
        )

        self.patients = [self.patient_a, self.patient_b]

        # 配置
        for agent in [self.patient_a, self.patient_b, self.nurse, self.robot]:
            self.grid.place_agent(agent, agent.start_pos)

        # データ収集
        self.datacollector = DataCollector(
            model_reporters={
                "mean_fall_risk": lambda m: sum(p.fall_risk for p in m.patients) / len(m.patients),
                "max_fall_risk": lambda m: max(p.fall_risk for p in m.patients),
                "total_unobserved": lambda m: sum(p.unobserved_steps for p in m.patients),
                "nurse_task": lambda m: m.nurse.task,
                "robot_pos": lambda m: m.robot.pos2d,
            },
            agent_reporters={
                "type": lambda a: type(a).__name__,
                "pos": lambda a: a.pos,
                "task": lambda a: getattr(a, "task", None),
                "fall_risk": lambda a: getattr(a, "fall_risk", None),
                "unobserved_steps": lambda a: getattr(a, "unobserved_steps", None),
            },
        )

        self.datacollector.collect(self)

    def is_patient_observed(self, patient: PatientAgent) -> bool:
        """看護師またはロボットが近ければ '見守られている' とみなす"""
        nurse_near = manhattan(patient.pos2d, self.nurse.pos2d) <= 1
        robot_near = manhattan(patient.pos2d, self.robot.pos2d) <= 1
        return nurse_near or robot_near

    def step(self) -> None:
        # 順番を固定:
        # 1. ロボットと看護師が動く
        # 2. 患者が動く
        # こうすると「介入→患者変化」が見やすい
        self.robot.step()
        self.nurse.step()

        for patient in self.patients:
            patient.step()

        self.datacollector.collect(self)

    def run(self, steps: int = 10) -> None:
        for _ in range(steps):
            self.step()

    def print_state(self, step_label: str = "") -> None:
        header = f"\n--- {step_label} ---" if step_label else "\n--- state ---"
        print(header)
        print(f"Nurse pos={self.nurse.pos2d}, task={self.nurse.task}")
        print(f"Robot pos={self.robot.pos2d}, speaking={self.robot.speaking}, target={self.robot.target_pos}")
        for p in self.patients:
            print(
                f"{p.name}: pos={p.pos2d}, task={p.task}, "
                f"unobserved={p.unobserved_steps}, risk={p.fall_risk:.2f}"
            )


# =========================
# ロボット方策
# =========================
def stay_policy(model: WardModel, robot: RobotAgent) -> None:
    """その場待機"""
    robot.target_pos = robot.pos2d
    robot.speaking = False


def go_to_highest_risk_patient_policy(model: WardModel, robot: RobotAgent) -> None:
    """一番リスクの高い患者へ向かって声かけする"""
    target_patient = max(model.patients, key=lambda p: p.fall_risk)
    robot.target_pos = target_patient.pos2d
    robot.speaking = True


def go_to_patient_a_policy(model: WardModel, robot: RobotAgent) -> None:
    """PatientA に向かう固定方策"""
    robot.target_pos = model.patient_a.pos2d
    robot.speaking = True


# =========================
# 比較実験
# =========================
def simulate(policy: RobotPolicy, steps: int = 10, seed: int = 42) -> WardModel:
    model = WardModel(robot_policy=policy, seed=seed)
    model.print_state("initial")
    model.run(steps=steps)
    model.print_state(f"after {steps} steps")
    return model


if __name__ == "__main__":
    print("=== Case 1: robot stays ===")
    model1 = simulate(stay_policy, steps=12, seed=42)

    print("\n=== Case 2: robot goes to PatientA ===")
    model2 = simulate(go_to_patient_a_policy, steps=12, seed=42)

    df1 = model1.datacollector.get_model_vars_dataframe()
    df2 = model2.datacollector.get_model_vars_dataframe()

    print("\n=== Summary comparison (final step) ===")
    print("Stay policy:")
    print(df1.tail(1).to_string(index=False))

    print("\nGo-to-PatientA policy:")
    print(df2.tail(1).to_string(index=False))