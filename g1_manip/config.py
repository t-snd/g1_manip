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
# アクチュエータ（位置制御 = PD）の設定
# ---------------------------------------------------------------------------
# menagerie の g1_with_hands.xml の位置アクチュエータは「needs tuning」（調整前の仮値）で、
# 全関節 kp=500, dampratio=1 になっている。フェーズ3の調査結果から次のように変更する:
# - 腕: kp=500 / dampratio=1 のまま（追従精度を優先）
# - 指: kp=500 だと 50Hz の目標更新ごとにトルク上限（1.4〜2.45Nm）に張り付き、
#       全開・全閉のような動きになるので、低ゲインにする（kv は臨界減衰: 2*sqrt(kp*armature 0.01)）
# - 重力補償: MuJoCo の gravcomp + actuatorgravcomp で、重力による定常たわみ（数 mrad）を無くす
#   （補償トルクはアクチュエータ経由で加わり、トルク上限に含まれる）
FINGER_KP = 20.0
FINGER_KV = 0.9
GRAVITY_COMPENSATION = True

# ---------------------------------------------------------------------------
# 制御対象の関節（行動＝これらの関節の目標角度（絶対値）[rad]）
# ---------------------------------------------------------------------------
# アクチュエータは全て位置型（ctrlrange=関節可動域）。ゲインは上の「アクチュエータの設定」を参照。
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
# 右手先（手のひら）サイトと IK（フェーズ4）
# ---------------------------------------------------------------------------
# Dex3 の右手は手のひらが wrist_yaw_link の +y 側を向き、指は +x 方向に伸びて +y 側へ曲がる。
# サイトは手のひら表面の中心に置き、軸を「x = 指の向き、z = 手のひらの法線（外向き）」にする
# （wrist_yaw_link の x 軸まわりに -90° 回した向き）。
RIGHT_PALM_SITE = "right_palm"
RIGHT_PALM_BODY = "right_wrist_yaw_link"
RIGHT_PALM_POS = (0.085, 0.017, 0.0)
RIGHT_PALM_QUAT = (0.70710678, -0.70710678, 0.0, 0.0)  # (w, x, y, z)

# 手のひらを下に向けた姿勢（サイト z 軸 = 世界の -z。この quat 自体は指 = 世界の +x）。
# IK では法線まわりの回転（yaw）は自由にするので、実際に解ける指の向きは位置によって変わる
# （作業範囲内で世界 x 軸から +28°〜+85°、準備姿勢では約 +79° でほぼ左＝体の中心側を向く）。
PALM_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)
IK_SOLVER = "daqp"
IK_POSITION_COST = 1.0
IK_ORIENTATION_COST = (1.0, 1.0, 0.0)  # サイト座標系の (x, y, z) 軸まわり。z（法線まわり）は 0 = 自由
IK_FRAME_LM_DAMPING = 1.0  # 手先タスクの Levenberg-Marquardt 減衰（mink 公式例と同じ）
# 右腕を準備姿勢付近に保つ弱い正則化。手先タスクとのトレードオフで、到達限界付近（x≈0.38〜0.40）では
# 0.2〜0.4cm の定常的なずれが残る（cost 0 なら 0 になる）。
IK_POSTURE_COST = 1e-2
IK_DAMPING = 1e-3
IK_DT = 0.02  # [s] 反復1回あたりの積分時間（ゲイン1なので反復はほぼガウス・ニュートン法になる）
IK_MAX_ITERS = 100
IK_POS_TOL = 1e-3  # [m]
IK_ORI_TOL = 1e-2  # [rad]
IK_STEP_TOL = 1e-5  # [rad] 反復1回の関節変化がこれ未満なら、誤差が残っていても打ち切る（正則化との釣り合い点に収束済み）

# 準備姿勢: 右手を机の上に上げ、手のひらを下に向けた姿勢。エピソード開始時はここから始める
# （ホームの腕を下ろした姿勢から前に出すと手首が天板の下に引っかかるため）。腕の関節角は起動時に IK で求める。
READY_PALM_POS = (0.20, -0.15, 1.00)

# ---------------------------------------------------------------------------
# フェーズ4 到達テスト（scripts/03_ik_reach_test.py）
# ---------------------------------------------------------------------------
REACH_SEED = 1
REACH_N_CUBE = 10  # キューブの初期位置の範囲（CUBE_XY_*）から取る点の数
REACH_N_TARGET = 10  # 目標マーカーの範囲（TARGET_XY_*）から取る点の数
REACH_APPROACH_HEIGHT = 0.15  # [m] 天板上面からの手のひらの高さ（上方の通過点）
REACH_HEIGHT = 0.07  # [m] 天板上面からの手のひらの高さ（到達点。親指が手のひらから約 6cm 下に出るため）
REACH_MOVE_TIME = 1.0  # [s] 通過点間の水平移動
REACH_DESCEND_TIME = 0.8  # [s] 下降・上昇
REACH_HOLD_TIME = 0.5  # [s] 到達点での静止（この後に誤差を測る）
REACH_CUBE_PARK_XY = (0.58, 0.30)  # テスト中はキューブを机の奥の隅に置いて手と当たらないようにする
REACH_TOL = 0.02  # [m] 合格とする手のひらの位置誤差
REACH_MIN_SUCCESS = 18  # 20 点中の合格点数の下限

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
TRACK_STEP_SS_ERR = 0.02  # [rad] 目標姿勢への移動後の誤差（移動開始から TEST_STEP_HOLD 秒の時点）
TRACK_OVERSHOOT_MAX = 0.05  # 目標姿勢への移動のオーバーシュート（移動量に対する比）
# 目標姿勢への移動中に、指のトルクが上限に張り付いていた時間の割合（物理ステップ単位）の上限。
# 指を kp=500 に戻すと 11〜37% になる（低ゲイン化の回帰検出用）。現設定の指は 2% 以下。
# 腕（kp=500）の手首は 50Hz の目標更新直後に数 ms 飽和し 15% 程度になるが、追従には影響しないので判定対象外。
TRACK_SATURATION_MAX = 0.10
IDLE_MAX_DEV = 0.01  # [rad] 制御対象外の関節（左腕・左手）の、整定後のホーム姿勢からの最大ずれ

# 右肘の正弦波（中心・振幅 [rad]、周波数 [Hz]、長さ [s]）。目標はホーム角から時定数 TEST_SINE_TAU で正弦波へ移る。
TEST_SINE_CENTER, TEST_SINE_AMP, TEST_SINE_HZ, TEST_SINE_SECONDS = 0.9, 0.4, 0.5, 4.0
TEST_SINE_TAU = 0.2  # [s]
TEST_SINE_SKIP = 0.5  # [s] 誤差評価から除く立ち上がり時間
TEST_SETTLE = 0.5  # [s] テスト前にホームで整定させる時間（重力による定常たわみを基準から除くため）
TEST_STEP_HOLD = 1.0  # [s] 目標姿勢への移動（補間＋保持）の時間
# 目標は急なステップではなく、デモ（IK 軌道の補間）と同じく滑らかに動かす（min-jerk 補間）。
# 急なステップはトルク上限に当たってオーバーシュートするが、デモ・方策の出力では起きない前提。
TEST_STEP_RAMP = 0.5  # [s] min-jerk 補間の時間
# 目標姿勢。14関節すべてが「ホームから 0.2rad 以上離れ、互いに 0.05rad 以上異なる」値になるよう選んだ
# （アクチュエータの取り違えがあれば必ず誤差として現れる）。腕は机に当たらないよう後ろ・外側へ動かす。
TEST_STEP_ARM_DELTA = [0.30, -0.30, 0.25, -0.35, 0.40, 0.20, -0.30]  # RIGHT_ARM_JOINTS の順、ホームからの差 [rad]
TEST_STEP_HAND_FRACTION = [0.65, 0.35, 0.25, 0.40, 0.45, 0.80, 0.75]  # RIGHT_HAND_JOINTS の順、可動域内の位置 (0〜1)
