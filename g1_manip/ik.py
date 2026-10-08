"""mink による右手先の IK（手先の目標姿勢 → 右腕7関節の目標角）。

IK は物理シミュレーションとは別の運動学モデル（mink.Configuration が持つ MjData）上で解く。
物理側の関節角は位置制御の遅れを含むので、IK の状態は前回の解を引き継ぎ（ウォームスタート）、
物理側の qpos は必要なとき（エピソード開始時など）だけ reset() で取り込む。
"""

import mink
import mujoco
import numpy as np

from g1_manip import config


class RightHandIK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.configuration = mink.Configuration(model)

        self.hand_task = mink.FrameTask(
            frame_name=config.RIGHT_PALM_SITE,
            frame_type="site",
            position_cost=config.IK_POSITION_COST,
            orientation_cost=config.IK_ORIENTATION_COST,
            lm_damping=config.IK_FRAME_LM_DAMPING,
        )
        # 右腕を基準姿勢（reset 時の姿勢）付近に保つ弱い正則化。冗長自由度（肘の位置など）を安定させる。
        self.posture_task = mink.PostureTask(model, cost=config.IK_POSTURE_COST)
        self.tasks = [self.hand_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model)]

        # 右腕7関節以外（左腕・両手の指・キューブ）は IK では動かさない（等式制約で厳密に固定）。
        arm_dofs = {model.jnt_dofadr[model.joint(j).id] for j in config.RIGHT_ARM_JOINTS}
        frozen = [i for i in range(model.nv) if i not in arm_dofs]
        self.constraints = [mink.DofFreezingTask(model, dof_indices=frozen)]

        self.arm_qpos_ids = np.array([model.joint(j).qposadr[0] for j in config.RIGHT_ARM_JOINTS])

    def reset(self, qpos: np.ndarray) -> None:
        """IK の状態を与えた qpos（通常は物理側の現在値）に合わせ、姿勢正則化の基準もそこにする。"""
        self.configuration.update(qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    @property
    def arm_q(self) -> np.ndarray:
        """IK の現在の解（右腕7関節、RIGHT_ARM_JOINTS の順）。"""
        return self.configuration.q[self.arm_qpos_ids].copy()

    def palm_pose(self) -> mink.SE3:
        """IK の現在の解での手のひらサイトの姿勢（世界座標）。"""
        return self.configuration.get_transform_frame_to_world(config.RIGHT_PALM_SITE, "site")

    def errors(self) -> tuple[float, float]:
        """手先タスクの (位置誤差 [m], 姿勢誤差 [rad])。姿勢誤差はコストを付けた軸だけで測る。"""
        err = self.hand_task.compute_error(self.configuration)
        ori_mask = np.asarray(config.IK_ORIENTATION_COST) > 0
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:][ori_mask]))

    def solve(self, target_pos, target_quat=config.PALM_DOWN_QUAT,
              max_iters: int = config.IK_MAX_ITERS) -> np.ndarray:
        """手のひらサイトの目標 (位置, 姿勢 quat wxyz) に対する右腕7関節の目標角を返す。

        注意: IK_ORIENTATION_COST の z 成分が 0 のため、target_quat のうち手のひらの法線まわりの回転
        （yaw）は無視され、傾き（法線の向き）だけが合わせられる。

        前回の解から反復する（ウォームスタート）。誤差が許容値に入るか、反復の更新が十分小さくなったら
        打ち切る。収束しなくても最後の解を返すので、誤差は errors() で確認すること。
        """
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(target_quat, dtype=float)), np.asarray(target_pos, dtype=float))
        self.hand_task.set_target(target)
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks, config.IK_DT, config.IK_SOLVER,
                                damping=config.IK_DAMPING, limits=self.limits, constraints=self.constraints)
            self.configuration.integrate_inplace(vel, config.IK_DT)
            pos_err, ori_err = self.errors()
            if pos_err <= config.IK_POS_TOL and ori_err <= config.IK_ORI_TOL:
                break
            if np.abs(vel).max() * config.IK_DT < config.IK_STEP_TOL:
                break
        return self.arm_q
