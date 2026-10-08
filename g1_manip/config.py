"""プロジェクト全体の定数・ハイパーパラメータ。

制御周期、関節名リスト、ランダム化範囲、成功判定しきい値などはすべてここに集約する。
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_G1_DIR = PROJECT_ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"
G1_XML = MENAGERIE_G1_DIR / "g1_with_hands.xml"
SCENE_XML = PROJECT_ROOT / "assets" / "scene_table.xml"

# ---------------------------------------------------------------------------
# シミュレーション・制御周期
# ---------------------------------------------------------------------------
SIM_TIMESTEP = 0.002  # [s] scene_table.xml の <option timestep> と一致させること
CONTROL_HZ = 50
N_SUBSTEPS = round(1.0 / (CONTROL_HZ * SIM_TIMESTEP))  # 制御1ステップあたりの mj_step 回数

# ---------------------------------------------------------------------------
# 下半身固定
# ---------------------------------------------------------------------------
# 骨盤の freejoint を削除してベースをワールドに固定する。
FREEJOINT_NAME = "floating_base_joint"
# 脚と腰の関節は削除する（関節ゼロ角＝直立姿勢で剛体として固定される）。
# 対応するアクチュエータも一緒に削除する。
REMOVED_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# ---------------------------------------------------------------------------
# 制御対象の関節（行動＝これらの関節の目標角度（絶対値）[rad]）
# ---------------------------------------------------------------------------
# アクチュエータは全て位置型（menagerie の class="g1": kp=500, dampratio=1, ctrlrange=関節可動域）。
# 並びは state/action ベクトルの次元順そのもの。右手はアクチュエータの定義順（index→middle）と
# qpos の順（middle→index）が異なるため、インデックスは必ず関節名から引くこと（env.joint_*_ids）。
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
CONTROLLED_JOINTS = RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS  # 14 次元
LEFT_ARM_JOINTS = [j.replace("right_", "left_", 1) for j in RIGHT_ARM_JOINTS]
LEFT_HAND_JOINTS = [j.replace("right_", "left_", 1) for j in RIGHT_HAND_JOINTS]

# ---------------------------------------------------------------------------
# 初期姿勢（ホーム）
# ---------------------------------------------------------------------------
# menagerie の "stand" キーフレームの腕・手の値（腕を体側に下ろした姿勢）。
# ここに無い関節は 0。
# 注: stand キーの thumb_1 = ±1.05 は可動域の端（±1.0472）で親指が手のひらに接触したままになるため、
#     ±0.9 に戻して自己接触のない姿勢にしている。
HOME_QPOS = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.28,
    "left_hand_thumb_1_joint": 0.9,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.28,
    "right_hand_thumb_1_joint": -0.9,
}

# ---------------------------------------------------------------------------
# 頭部カメラ（フェーズ8で使用）
# ---------------------------------------------------------------------------
HEAD_CAMERA_NAME = "head"
HEAD_CAMERA_BODY = "torso_link"
HEAD_CAMERA_POS = (0.074, 0.0, 0.39)  # torso_link 座標系。頭部メッシュ前面・目の高さ付近
# 視線の向き。キューブ範囲と目標範囲を合わせた作業領域の中心 (x≈0.30, y≈-0.10) を見る。
HEAD_CAMERA_PITCH_DEG = 56.0  # 水平から下向きの角度
HEAD_CAMERA_YAW_DEG = -23.0  # 正面から右（-y 側）へ向ける角度は負
HEAD_CAMERA_FOVY = 60.0

# ---------------------------------------------------------------------------
# シーン内オブジェクト名（scene_table.xml と一致させること）
# ---------------------------------------------------------------------------
TABLE_TOP_GEOM = "table_top"
CUBE_BODY = "cube"
CUBE_GEOM = "cube"
CUBE_JOINT = "cube_freejoint"
TARGET_BODY = "target"  # mocap ボディ（data.mocap_pos で位置を変える）

# ---------------------------------------------------------------------------
# ランダム化範囲（暫定値。フェーズ4の到達性テストで調整する）
# ---------------------------------------------------------------------------
SEED = 0
CUBE_XY_NOMINAL = np.array([0.30, -0.15])  # 机上のキューブ基準位置（world x, y）
CUBE_XY_RANGE = np.array([0.10, 0.10])  # 基準位置からの ± 範囲 [m]
CUBE_YAW_RANGE_DEG = 30.0  # ± [deg]
TARGET_XY_NOMINAL = np.array([0.30, 0.0])
TARGET_XY_RANGE = np.array([0.08, 0.05])

# ---------------------------------------------------------------------------
# フェーズ3 アクチュエータ追従テスト（scripts/02_check_actuators.py）の合格しきい値
# ---------------------------------------------------------------------------
TRACK_SINE_MAX_ERR = 0.05  # [rad] 右肘の正弦波追従の最大誤差（立ち上がり TEST_SINE_SKIP 秒を除く）
TRACK_STEP_SS_ERR = 0.02  # [rad] ステップ目標に対する定常誤差（TEST_STEP_HOLD 秒保持した時点）
IDLE_MAX_DEV = 0.01  # [rad] 制御対象外の関節（左腕・左手）の、整定後のホーム姿勢からの最大ずれ

# 右肘の正弦波（中心・振幅 [rad]、周波数 [Hz]、長さ [s]）。目標はホーム角から時定数 TEST_SINE_TAU で正弦波へ移る。
TEST_SINE_CENTER, TEST_SINE_AMP, TEST_SINE_HZ, TEST_SINE_SECONDS = 0.9, 0.4, 0.5, 4.0
TEST_SINE_TAU = 0.2  # [s]
TEST_SINE_SKIP = 0.5  # [s] 誤差評価から除く立ち上がり時間
TEST_SETTLE = 0.5  # [s] テスト前にホームで整定させる時間（重力による定常たわみを基準から除くため）
TEST_STEP_HOLD = 1.0  # [s] ステップ目標の保持時間
# ステップ目標。14関節すべてが「ホームから 0.2rad 以上離れ、互いに 0.05rad 以上異なる」値になるよう選んだ
# （アクチュエータの取り違えがあれば必ず誤差として現れる）。腕は机に当たらないよう後ろ・外側へ動かす。
TEST_STEP_ARM_DELTA = [0.30, -0.30, 0.25, -0.35, 0.40, 0.20, -0.30]  # RIGHT_ARM_JOINTS の順、ホームからの差 [rad]
TEST_STEP_HAND_FRACTION = [0.65, 0.35, 0.25, 0.40, 0.45, 0.80, 0.75]  # RIGHT_HAND_JOINTS の順、可動域内の位置 (0〜1)
