"""スクリプト方策（デモ生成用）。

各フェーズの終わりの手先位置を決め、その間を min-jerk 補間した経路を IK で追う。
行動は CONTROLLED_JOINTS の目標角（右腕7関節は IK の解、右手7関節は指の目標）。
"""

import numpy as np

from g1_manip import config
from g1_manip.ik import RightHandIK


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class ScriptedReachPolicy:
    """リーチング: 準備姿勢の手のひら位置から目標点へ min-jerk で移動し、その後は静止する。"""

    def __init__(self, env):
        self.env = env
        self.ik = RightHandIK(env.model)
        self.n_arm = len(config.RIGHT_ARM_JOINTS)
        self.hand_target = np.array([config.HOME_QPOS.get(j, 0.0) for j in config.RIGHT_HAND_JOINTS])
        self.n_move = int(round(config.REACH_SCRIPT_MOVE_TIME * config.CONTROL_HZ))

    def reset(self) -> None:
        """エピソード開始時に呼ぶ（env.reset の後）。IK の状態を物理側の現在の姿勢に合わせる。"""
        self.ik.reset(self.env.data.qpos)
        self.start = self.env.palm_pos()
        self.goal = self.env.goal.copy()
        self.t = 0

    def act(self) -> np.ndarray:
        """現在のステップの行動（CONTROLLED_JOINTS の目標角）を返し、内部の時刻を1進める。"""
        self.t += 1
        target = self.start + min_jerk(self.t / self.n_move) * (self.goal - self.start)
        arm_q = self.ik.solve(target)
        return np.concatenate([arm_q, self.hand_target]).astype(np.float32)
