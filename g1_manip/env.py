"""シーンの読み込み・リセット・関節目標の適用。

観測取得・成功判定はフェーズ4以降でここに追加する。
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
