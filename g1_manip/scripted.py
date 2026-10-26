"""スクリプト方策（デモ生成用）。

各フェーズの終わりの手先位置を決め、その間を min-jerk 補間した経路を IK で追う。
行動は CONTROLLED_JOINTS の目標角（右腕7関節は IK の解、右手7関節は指の目標）。
"""

import mujoco
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


def side_palm_rotation(yaw_deg: float) -> np.ndarray:
    """横からつかむ手のひらの姿勢（回転行列、列 = サイトの x, y, z 軸の世界座標）。

    yaw 0 で指（x）が世界 +x、法線（z）が世界 +y（体の内側）、y = z×x は真下。yaw は世界 z まわり。
    """
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    x = np.array([c, s, 0.0])
    z = np.array([-s, c, 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).flatten())
    return quat


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """quat（wxyz）の球面線形補間（短い方の回り方）。"""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)


class ScriptedPickPlacePolicy:
    """ピック＆プレース: 横から近づいてつかみ、持ち上げ、目標の上へ運んで置き、離して退く。

    区間（PICK_TIMES）: approach（把持位置の横・上へ、手のひらを立てながら）→ descend（横まで下りる）
    → insert（法線方向に近づく）→ close（指を閉じる）→ settle → lift → move（目標の上へ、手の向きを置く向きに回す）
    → lower → release（指を開く）→ back（法線の逆向きに引く）→ up。各区間の中は手先の位置を min-jerk、姿勢を slerp で
    補間し、50Hz ごとに IK（姿勢 3 軸すべて、腕と胴体の衝突回避あり）で右腕の目標角を出す。
    置く位置は、持ち上げた後と下ろす前に手の中のキューブの位置を測って決める（閉じるときや手首を回すときにキューブが少しずれるため）。
    """

    SEGMENTS = ("approach", "descend", "insert", "close", "settle", "lift", "move", "lower", "release", "back", "up")

    def __init__(self, env):
        self.env = env
        full = (1.0, 1.0, 1.0)
        self.ik = RightHandIK(env.model, orientation_cost=full, avoid_body_collision=True)
        self.check_ik = RightHandIK(env.model, orientation_cost=full, avoid_body_collision=True)  # 到達できるかの確認用
        self.hand_home = np.array([config.HOME_QPOS.get(j, 0.0) for j in config.RIGHT_HAND_JOINTS])
        self.hand_open = np.asarray(config.HAND_OPEN, dtype=float)
        self.hand_close = np.asarray(config.HAND_CLOSE, dtype=float)
        self.n_steps = {k: int(round(v * config.CONTROL_HZ)) for k, v in config.PICK_TIMES.items()}

    # --- 計画 ---
    def _path_error(self, waypoints, quat) -> float:
        """check_ik の現在の状態から waypoints を順に IK で追ったときの、手先誤差の最大（位置 ＋ 0.05m × 姿勢）。"""
        worst = 0.0
        for p in waypoints:
            for _ in range(config.PICK_FEASIBLE_IK_REPEATS):
                self.check_ik.solve(p, quat)
            err = self.check_ik.hand_task.compute_error(self.check_ik.configuration)
            worst = max(worst, float(np.linalg.norm(err[:3]) + config.PICK_FEASIBLE_ORI_WEIGHT * np.linalg.norm(err[3:])))
        return worst

    def _grasp_waypoints(self, cube: np.ndarray, yaw: float):
        rot = side_palm_rotation(yaw)
        grasp = cube - rot @ np.asarray(config.GRASP_CUBE_IN_PALM)
        pre = grasp - rot[:, 2] * config.GRASP_STANDOFF
        return rot, [pre + [0.0, 0.0, config.GRASP_ABOVE], pre, grasp, grasp + [0.0, 0.0, config.LIFT_HEIGHT]]

    def _place_waypoints(self, offset: np.ndarray, yaw: float):
        """offset: 手のひらサイト座標系でのキューブ中心の位置。"""
        rot = side_palm_rotation(yaw)
        center = self.env.target + [0.0, 0.0, config.CUBE_HALF + config.PLACE_CLEARANCE]
        place = center - rot @ offset
        back = place - rot[:, 2] * config.RETREAT_BACK
        return rot, [place + [0.0, 0.0, config.LIFT_HEIGHT], place, back, back + [0.0, 0.0, config.RETREAT_UP]]

    def _choose_yaws(self) -> tuple[float, float, bool]:
        """把持と置くときの手の向き（yaw [deg]）を選ぶ。届く組み合わせが無ければ、調べた中で最も誤差の小さい組み合わせ
        （最初の把持候補と、把持の経路が届いた候補だけを調べる）を使い、False を返す。"""
        e = self.env
        qw, qz = e.cube_quat()[[0, 3]]
        cube_yaw = np.degrees(2 * np.arctan2(qz, qw))
        cands = [cube_yaw + 90 * k + off for k in range(-2, 3) for off in config.GRASP_YAW_OFFSETS_DEG]
        cands = [c for c in cands if config.GRASP_YAW_RANGE_DEG[0] <= c <= config.GRASP_YAW_RANGE_DEG[1]]
        cands.sort(key=lambda c: (min(abs(((c - cube_yaw) % 90) - d) for d in (0, 90)), abs(c - config.GRASP_YAW_PREF_DEG)))
        best = None
        for g in cands:
            self.check_ik.reset(e.data.qpos)
            rot, wps = self._grasp_waypoints(e.cube_pos(), g)
            g_err = self._path_error(wps, mat_to_quat(rot))
            if g_err > config.PICK_FEASIBLE_TOL and best is not None:
                continue
            q_lifted = self.check_ik.configuration.q.copy()
            for p in sorted(config.PLACE_YAW_CANDIDATES_DEG, key=lambda c: abs(c - g)):
                if abs(p - g) > config.PICK_MAX_YAW_CHANGE_DEG:
                    continue
                self.check_ik.configuration.update(q_lifted)
                rot_p, wps_p = self._place_waypoints(np.asarray(config.GRASP_CUBE_IN_PALM), p)
                err = max(g_err, self._path_error(wps_p, mat_to_quat(rot_p)))
                if best is None or err < best[2]:
                    best = (g, p, err)
                if err <= config.PICK_FEASIBLE_TOL:
                    return g, p, True
        if best is None:  # どの候補も手首を回す角度の上限を超える
            g = cands[0]
            return g, min(config.PLACE_YAW_CANDIDATES_DEG, key=lambda c: abs(c - g)), False
        return best[0], best[1], False

    # --- 実行 ---
    def reset(self) -> None:
        """エピソード開始時に呼ぶ（env.reset の後）。手の向きを選び、経路を作る。"""
        e = self.env
        self.ik.reset(e.data.qpos)
        self.grasp_yaw, self.place_yaw, self.feasible = self._choose_yaws()
        self.ik.reset(e.data.qpos)
        self.grasp_rot, self.grasp_wps = self._grasp_waypoints(e.cube_pos(), self.grasp_yaw)
        self.start = (e.palm_pos(), mat_to_quat(e.data.site_xmat[e.palm_site].reshape(3, 3)), self.hand_home.copy())
        self.seg, self.t = 0, 0
        self.prev_arm = self.ik.arm_q
        self.cube_start = e.cube_pos()
        self.pregrasp_clean = True  # 指を閉じる前に手がキューブに触れず、キューブが動いていないか
        self.phase = self.SEGMENTS[0]
        self.goal = self._segment_goal(self.phase)

    def _segment_goal(self, name: str):
        """区間の終わりの (手のひら位置, 姿勢 quat, 指の目標)。"""
        q_g = mat_to_quat(self.grasp_rot)
        above, pre, grasp, lifted = self.grasp_wps
        if name == "approach":
            return above, q_g, self.hand_open
        if name == "descend":
            return pre, q_g, self.hand_open
        if name == "insert":
            return grasp, q_g, self.hand_open
        if name in ("close", "settle"):
            return grasp, q_g, self.hand_close
        if name == "lift":
            return lifted, q_g, self.hand_close
        if name in ("move", "lower"):
            # 手の中のキューブの位置（物理側の手のひら座標系）を測り、キューブが目標の上に来る手の位置を決める。
            # move で手首を回すと手の中でキューブがずれることがあるので、下ろす前（lower の始め）にも測り直す
            e = self.env
            rot_act = e.data.site_xmat[e.palm_site].reshape(3, 3)
            offset = rot_act.T @ (e.cube_pos() - e.palm_pos())
            rot_p, self.place_wps = self._place_waypoints(offset, self.place_yaw)
            self.place_quat = mat_to_quat(rot_p)
            if name == "move":
                return self.place_wps[0], self.place_quat, self.hand_close
        above_p, place, back, up = self.place_wps
        return {"lower": (place, self.place_quat, self.hand_close),
                "release": (place, self.place_quat, self.hand_open),
                "back": (back, self.place_quat, self.hand_open),
                "up": (up, self.place_quat, self.hand_open)}[name]

    def demo_ok(self) -> bool:
        """デモとして保存してよいか（成功判定とは別）。指を閉じる前に手がキューブを押したエピソードは、
        たまたま成功しても把持の経路がリセット時のキューブ位置と合っていないので保存しない。"""
        return self.pregrasp_clean

    def act(self) -> np.ndarray:
        """現在のステップの行動（CONTROLLED_JOINTS の目標角）を返し、内部の時刻を1進める。最後の区間の後は静止する。"""
        if self.phase in ("approach", "descend", "insert") and (
                self.env.hand_cube_contacts()
                or np.linalg.norm(self.env.cube_pos() - self.cube_start) > config.PICK_PREGRASP_MAX_SHIFT):
            self.pregrasp_clean = False
        if self.seg < len(self.SEGMENTS):
            n = self.n_steps[self.phase]
            self.t += 1
            s = min_jerk(self.t / n)
            (p0, q0, h0), (p1, q1, h1) = self.start, self.goal
            pos, quat, hand = p0 + s * (p1 - p0), slerp(q0, q1, s), h0 + s * (h1 - h0)
            if self.t >= n:
                self.start, self.seg, self.t = self.goal, self.seg + 1, 0
                if self.seg < len(self.SEGMENTS):
                    self.phase = self.SEGMENTS[self.seg]
                    self.goal = self._segment_goal(self.phase)
                else:
                    self.phase = "done"
        else:
            pos, quat, hand = self.goal
        arm_q = self.ik.solve(pos, quat)
        # 1 ステップの変化が上限を超えたら全関節を同じ比率で縮め（関節空間で動く向きを保ち、手先が経路から外れにくくする）、
        # IK の状態も縮めた値に合わせる（次のステップはそこから解き直す）
        delta = arm_q - self.prev_arm
        scale = min(1.0, config.PICK_MAX_JOINT_STEP / max(float(np.abs(delta).max()), 1e-12))
        clipped = self.prev_arm + scale * delta
        if scale < 1.0:
            q = self.ik.configuration.q.copy()
            q[self.ik.arm_qpos_ids] = clipped
            self.ik.configuration.update(q)
        self.prev_arm = clipped
        return np.concatenate([clipped, hand]).astype(np.float32)
