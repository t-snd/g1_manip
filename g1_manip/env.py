"""シーンの読み込み・リセット・観測取得・行動適用・成功判定。

下の関数群（load_model / reset_to_* / joint_* / apply_joint_targets / step_control）が基本部品で、
タスク単位の環境は ReachEnv（フェーズ5のリーチング）にまとめている。
"""

import mujoco
import numpy as np

from g1_manip import config


def _load_g1_spec() -> mujoco.MjSpec:
    """menagerie の G1（ハンド付き）を読み込み、下半身を固定した spec を返す。"""
    g1 = mujoco.MjSpec.from_file(str(config.G1_XML))

    # 別ディレクトリのシーンに attach すると相対 meshdir が解決できないため、絶対パスにする。
    meshdir = config.G1_XML.parent / g1.meshdir
    for mesh in g1.meshes:
        mesh.file = str(meshdir / mesh.file)
    g1.meshdir = ""

    # menagerie のキーフレームは freejoint・脚・腰を含む qpos なので、関節削除後は使えない。
    for key in list(g1.keys):
        g1.delete(key)

    # 脚・腰の関節とそのアクチュエータを削除し、骨盤の freejoint も削除してワールドに固定する。
    removed = set(config.REMOVED_JOINTS)
    for act in list(g1.actuators):
        if act.target in removed:
            g1.delete(act)
    for name in config.REMOVED_JOINTS:
        g1.delete(g1.joint(name))
    g1.delete(g1.joint(config.FREEJOINT_NAME))

    # 指のアクチュエータを低ゲインにする（明示的な kv は biasprm[2] に負値で入れる）
    for act in g1.actuators:
        if "_hand_" in act.target:
            act.gainprm[0] = config.FINGER_KP
            act.biasprm[1] = -config.FINGER_KP
            act.biasprm[2] = -config.FINGER_KV

    # 重力補償: 全ボディに gravcomp を付け、補償力を各関節のアクチュエータ経由で加える
    if config.GRAVITY_COMPENSATION:
        pelvis = g1.body("pelvis")
        for body in [pelvis] + list(pelvis.find_all(mujoco.mjtObj.mjOBJ_BODY)):
            body.gravcomp = 1.0
        for joint in pelvis.find_all(mujoco.mjtObj.mjOBJ_JOINT):
            joint.actgravcomp = True

    # 右手先サイト（手のひら表面の中心。x = 指の向き、z = 手のひらの法線）
    g1.body(config.RIGHT_PALM_BODY).add_site(
        name=config.RIGHT_PALM_SITE, pos=config.RIGHT_PALM_POS, quat=config.RIGHT_PALM_QUAT,
        size=[0.01, 0.01, 0.01], rgba=[0.1, 0.4, 1.0, 1.0], group=4,
    )

    # 頭部カメラ: 頭部メッシュは torso_link に付いているので、torso_link にカメラを追加する。
    # MuJoCo のカメラは -z 方向を向く（x: 画像の右、y: 画像の上）。
    p = np.deg2rad(config.HEAD_CAMERA_PITCH_DEG)
    yaw = np.deg2rad(config.HEAD_CAMERA_YAW_DEG)
    forward = np.array([np.cos(p) * np.cos(yaw), np.cos(p) * np.sin(yaw), -np.sin(p)])
    x_axis = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # 画像の右（yaw=0 で -y = ロボットの右）
    y_axis = np.cross(-forward, x_axis)  # 画像の上
    g1.body(config.HEAD_CAMERA_BODY).add_camera(
        name=config.HEAD_CAMERA_NAME,
        pos=config.HEAD_CAMERA_POS,
        xyaxes=np.concatenate([x_axis, y_axis]),
        fovy=config.HEAD_CAMERA_FOVY,
    )
    return g1


def build_spec() -> mujoco.MjSpec:
    """机のシーン（scene_table.xml）に下半身固定の G1 を attach した spec を返す。"""
    scene = mujoco.MjSpec.from_file(str(config.SCENE_XML))
    g1 = _load_g1_spec()
    # 骨盤ボディ自身の pos（z=0.793、足裏が床に接する高さ）をそのまま使うのでフレームは原点。
    frame = scene.worldbody.add_frame()
    frame.attach_body(g1.body("pelvis"), "", "")
    return scene


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """全状態をリセットし、腕・手をホーム姿勢、位置アクチュエータの目標もホーム姿勢にする。

    キューブと目標マーカーは scene_table.xml の初期位置になる。
    """
    mujoco.mj_resetData(model, data)
    for name, q in config.HOME_QPOS.items():
        data.qpos[model.joint(name).qposadr[0]] = q
    # 位置アクチュエータの目標＝現在の関節角（ホームで静止させる）
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)


def reset_to_ready(model: mujoco.MjModel, data: mujoco.MjData, ik) -> None:
    """ホーム姿勢にリセットしたうえで、右腕を準備姿勢（READY_PALM_POS、手のひら下向き）に置く。

    右腕の関節角は IK（g1_manip.ik.RightHandIK）で求め、qpos と位置アクチュエータの目標の両方に入れる。
    IK の状態もこの姿勢に合わせる（姿勢正則化の基準も準備姿勢になる）。
    """
    reset_to_home(model, data)
    ik.reset(data.qpos)
    arm_q = ik.solve(config.READY_PALM_POS)
    arm_qpos = joint_qpos_ids(model, config.RIGHT_ARM_JOINTS)
    data.qpos[arm_qpos] = arm_q
    data.ctrl[joint_actuator_ids(model, config.RIGHT_ARM_JOINTS)] = arm_q
    mujoco.mj_forward(model, data)
    ik.reset(data.qpos)


def joint_qpos_ids(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """関節名の並びに対応する qpos のインデックス（1自由度の関節のみ）。"""
    return np.array([model.joint(n).qposadr[0] for n in joint_names])


def joint_actuator_ids(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """関節名の並びに対応する（その関節を駆動する）アクチュエータのインデックス。"""
    by_joint: dict[int, int] = {}
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = model.actuator_trnid[i, 0]
        if jid in by_joint:
            raise ValueError(f"関節 {model.joint(jid).name} を駆動するアクチュエータが複数ある")
        by_joint[jid] = i
    return np.array([by_joint[model.joint(n).id] for n in joint_names])


def apply_joint_targets(model: mujoco.MjModel, data: mujoco.MjData,
                        actuator_ids: np.ndarray, q_target: np.ndarray) -> None:
    """関節の目標角度（絶対値）を位置アクチュエータの ctrl に設定する（ctrlrange があればクリップ）。"""
    lo, hi = model.actuator_ctrlrange[actuator_ids].T
    limited = model.actuator_ctrllimited[actuator_ids].astype(bool)
    data.ctrl[actuator_ids] = np.where(limited, np.clip(q_target, lo, hi), q_target)


def step_control(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """制御1周期（1/CONTROL_HZ 秒）分だけ物理シミュレーションを進める。"""
    mujoco.mj_step(model, data, nstep=config.N_SUBSTEPS)


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """シーンをコンパイルし、ホーム姿勢を "home" キーフレームとして追加した model/data を返す。"""
    spec = build_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    # ビューアの「Reset」やキーフレーム選択でホーム姿勢に戻れるようにキーフレームを登録する。
    spec.add_key(name="home", qpos=data.qpos.copy(), ctrl=data.ctrl.copy(),
                 mpos=data.mocap_pos.flatten().copy(), mquat=data.mocap_quat.flatten().copy())
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    assert np.isclose(model.opt.timestep, config.SIM_TIMESTEP), "scene_table.xml の timestep と config が不一致"
    return model, data


class ReachEnv:
    """リーチングタスク: 準備姿勢から、机の上のランダムな点へ右手のひらを持っていく。

    観測:
      observation.state             CONTROLLED_JOINTS の関節角（14）
      observation.environment_state 目標点の位置（3）
    行動: CONTROLLED_JOINTS の目標角（絶対値、14）
    """

    task_name = config.REACH_TASK_NAME

    def __init__(self):
        from g1_manip.ik import RightHandIK  # リセット時の準備姿勢を求めるときだけ使う

        self.model, self.data = load_model()
        m = self.model
        self.state_qpos = joint_qpos_ids(m, config.CONTROLLED_JOINTS)
        self.action_act = joint_actuator_ids(m, config.CONTROLLED_JOINTS)
        self.palm_site = m.site(config.RIGHT_PALM_SITE).id
        self.target_mocap = m.body(config.TARGET_BODY).mocapid[0]
        self.cube_qpos = m.joint(config.CUBE_JOINT).qposadr[0]
        top = m.geom(config.TABLE_TOP_GEOM)
        self.table_top_z = float(self.data.geom_xpos[top.id][2] + top.size[2])

        # 準備姿勢の qpos を一度だけ IK で求めておく。リセット時はここから、ずらした手のひら位置へ IK で合わせる。
        self.ik = RightHandIK(m)
        reset_to_ready(m, self.data, self.ik)
        self.ready_qpos = self.data.qpos.copy()
        self.ready_ctrl = self.data.ctrl.copy()
        self.arm_qpos = joint_qpos_ids(m, config.RIGHT_ARM_JOINTS)
        self.arm_act = joint_actuator_ids(m, config.RIGHT_ARM_JOINTS)
        self.goal = np.zeros(3)

    def reset(self, seed: int) -> dict:
        """準備姿勢（手のひら位置をランダムにずらす）に戻し、目標点をランダムに決める。
        キューブは手の届かない机の奥の隅に置く。"""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.ready_qpos
        self.data.ctrl[:] = self.ready_ctrl
        start = np.asarray(config.READY_PALM_POS) + rng.uniform(-1, 1, 3) * np.asarray(config.READY_PALM_RANDOM)
        self.ik.reset(self.ready_qpos)
        arm_q = self.ik.solve(start)
        self.data.qpos[self.arm_qpos] = arm_q
        self.data.ctrl[self.arm_act] = arm_q
        self.data.qpos[self.cube_qpos:self.cube_qpos + 2] = config.REACH_CUBE_PARK_XY
        self.goal = np.array([rng.uniform(*config.REACH_GOAL_X), rng.uniform(*config.REACH_GOAL_Y),
                              self.table_top_z + rng.uniform(*config.REACH_GOAL_Z)])
        self.data.mocap_pos[self.target_mocap] = self.goal  # 目標マーカーを目標点に表示する
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self) -> dict:
        return {
            "observation.state": self.data.qpos[self.state_qpos].astype(np.float32),
            "observation.environment_state": self.goal.astype(np.float32),
        }

    def step(self, action: np.ndarray) -> dict:
        """行動（関節目標角）を適用し、制御1周期だけ進める。"""
        apply_joint_targets(self.model, self.data, self.action_act, np.asarray(action, dtype=float))
        step_control(self.model, self.data)
        return self.observation()

    def palm_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.palm_site].copy()

    def palm_speed(self) -> float:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.palm_site, vel, 0)
        return float(np.linalg.norm(vel[3:]))

    def is_success(self) -> bool:
        """手のひらが目標点から REACH_SUCCESS_TOL 以内で、ほぼ静止している。"""
        return (np.linalg.norm(self.palm_pos() - self.goal) <= config.REACH_SUCCESS_TOL
                and self.palm_speed() <= config.SUCCESS_SPEED_TOL)
