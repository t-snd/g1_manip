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
# 右腕と胴体の衝突回避（RightHandIK(avoid_body_collision=True)）
IK_COLLISION_ARM_BODIES = ["right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
                           "right_wrist_pitch_link", "right_wrist_yaw_link"]
IK_COLLISION_BODY_BODIES = ["torso_link", "pelvis"]
IK_COLLISION_MIN_DIST = 0.02  # [m]
IK_COLLISION_DETECT_DIST = 0.06  # [m]

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
# 画像入力（フェーズ8）: 頭部カメラの RGB 画像を observation.images.head として保存・入力する（計画書: まず 224×224 程度）
IMAGE_KEY = "observation.images.head"
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224

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

# ---------------------------------------------------------------------------
# フェーズ5 タスク・データ生成
# ---------------------------------------------------------------------------
FPS = CONTROL_HZ
SUCCESS_SPEED_TOL = 0.02  # [m/s] 「静止」とみなす速さ（手のひら・キューブ）

# --- リーチングタスク（パイプライン検証用）---
# 準備姿勢から、机の上の箱の中のランダムな点へ手のひらを持っていく（手のひら下向き）。
REACH_TASK_NAME = "reach the target point with the right palm"
REACH_GOAL_X = (0.20, 0.40)  # [m] 目標点の範囲（キューブの範囲と目標マーカーの範囲を合わせた範囲）
REACH_GOAL_Y = (-0.25, 0.05)
REACH_GOAL_Z = (0.07, 0.15)  # [m] 天板上面からの高さ
REACH_EPISODE_STEPS = 150  # 3 秒
REACH_SCRIPT_MOVE_TIME = 1.5  # [s] スクリプト方策が目標点へ移動する時間（min-jerk）
REACH_SUCCESS_TOL = 0.02  # [m] 最終ステップで手のひらが目標点からこの距離以内なら成功
REACH_ENV_STATE_NAMES = ["goal_x", "goal_y", "goal_z"]
# エピソード開始時の手のひら位置を準備姿勢（READY_PALM_POS）から ± この範囲でずらす（IK で腕の関節角を求める）。
# 初期状態が毎回同じだと、評価時に observation.state の配線を間違えても結果に表れにくいため。
READY_PALM_RANDOM = (0.03, 0.03, 0.02)  # [m]

# --- ピック＆プレース ---
# 机の上のキューブを右手で横からつかみ（手のひらを体の内側＝+y 側に向け、親指と中指で指の向きに挟む）、
# 目標マーカーの上に置く。上からつかむと頭部カメラ（フェーズ8）で手がキューブを隠すため横から（ユーザー決定）。
# Dex3 は人差し指と中指が手のひらの幅方向に並ぶので、手のひらを立てると下側の中指だけがキューブに当たる。
PICK_TASK_NAME = "pick the cube and place it on the target"
PICK_ENV_STATE_NAMES = ["cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
                        "target_x", "target_y", "target_z"]
CUBE_HALF = 0.0225  # [m] scene_table.xml のキューブの半辺
# キューブ・目標の範囲（計画書の ±10cm から、横からの把持で腕が届く範囲に狭めた）
PICK_CUBE_X = (0.22, 0.38)
PICK_CUBE_Y = (-0.23, -0.07)
PICK_CUBE_YAW_DEG = 30.0  # ±
PICK_TARGET_X = (0.22, 0.38)
PICK_TARGET_Y = (-0.05, 0.05)
PICK_MIN_DIST = 0.10  # [m] キューブと目標の xy 距離の下限（範囲の境目 y=-0.07/-0.05 付近では 2cm まで近づきうるため）
PICK_EPISODE_STEPS = 450  # 9 秒（スクリプト方策は約 8 秒で終わり、残りは静止）
PICK_SUCCESS_TOL = 0.03  # [m] 最終ステップでキューブ中心が目標から xy でこの距離以内
PICK_SUCCESS_Z_TOL = 0.01  # [m] キューブ中心の高さが天板＋半辺からこの範囲（机の上に平らに置かれている）

# 把持（手のひらサイト座標系: x = 指の向き、y = z×x、z = 手のひらの法線）
GRASP_CUBE_IN_PALM = (0.035, 0.026, 0.035)  # 把持するときのキューブ中心の位置。y は下向きで、手のひらの中心をキューブの中心より 2.6cm 上にする（1.2cm では手首の下端と中指が天板に当たり、手が奥まで入らないことがあった）
GRASP_STANDOFF = 0.09  # [m] 把持位置から法線の逆向きにこれだけ離れた位置まで来てから、法線方向に近づく
GRASP_ABOVE = 0.04  # [m] 最初はさらにこれだけ上に来る（キューブの上から下ろさない）
# 指の目標（RIGHT_HAND_JOINTS の順）。開: 親指を手のひら側の後ろへ逃がす（近づくときに親指の先がキューブに当たらないよう）。
# 閉: 親指と中指・人差し指をキューブより深く指令して挟む（握る力は指令と接触位置の差 × kp）
HAND_OPEN = (0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0)
HAND_CLOSE = (0.0, -0.3, -1.1, 1.1, 1.2, 1.1, 1.2)
# 手の向き（yaw）: 指が世界 +x、法線が世界 +y を 0 として、世界 z まわりの角度。把持では指をキューブの面の法線に
# 合わせたいので「キューブの yaw ＋ 90°の倍数 ＋ ずれ」を候補にし、ずれが小さく GRASP_YAW_PREF_DEG に近い順に、
# 腕が届く（IK の誤差が PICK_FEASIBLE_TOL 以内）最初の候補を使う。ずれは GRASP_YAW_OFFSETS_DEG まで（20°までは把持できることを確認）。
GRASP_YAW_PREF_DEG = -15.0
# 把持の手の向きの範囲。±45° を超えると手のひらの法線が体の内側（+y）より前（+x）や後ろを向き、手がキューブの手前
# （頭部カメラの側）から近づくことになるので、候補から外す（届かない配置はデモに入らない）
GRASP_YAW_RANGE_DEG = (-45.0, 45.0)
GRASP_YAW_OFFSETS_DEG = (0.0, -10.0, 10.0, -20.0, 20.0)
PLACE_YAW_CANDIDATES_DEG = (15.0, 0.0, 30.0, -15.0, -30.0)  # 置くときの手の向き（キューブは手と一緒に回るので自由に選べる）
# 運ぶ間に手首を回す角度の上限。60°以上回すと手の中でキューブが上下にずれ、下ろすときに下側の中指が先に天板に当たって
# キューブが浮いたまま落ちることがあった
PICK_MAX_YAW_CHANGE_DEG = 45.0
PICK_FEASIBLE_TOL = 0.005  # [m] 手先の位置誤差 ＋ PICK_FEASIBLE_ORI_WEIGHT × 姿勢誤差 [rad]
PICK_FEASIBLE_ORI_WEIGHT = 0.05  # [m/rad]
PICK_FEASIBLE_IK_REPEATS = 3  # 経路の確認で、通過点ごとに IK を解く回数（1 回 = 最大 IK_MAX_ITERS 反復）
# 右腕の目標角の 1 制御ステップあたりの変化の上限 [rad]（= 2.5 rad/s）。手先の経路が滑らかでも、IK の解が
# 冗長な自由度（肘の位置など）や手首の特異姿勢の近くで急に組み替わり、1 ステップで 0.3rad 以上跳ぶことがあった
PICK_MAX_JOINT_STEP = 0.05
PICK_PREGRASP_MAX_SHIFT = 0.002  # [m] 指を閉じる前にキューブが動いてよい距離。超えたら（手が押した）デモとして保存しない
LIFT_HEIGHT = 0.10  # [m]
PLACE_CLEARANCE = 0.005  # [m] 置くときのキューブの底と天板のすき間（ここで指を開いて落とす）
RETREAT_BACK = 0.08  # [m] 離した後に法線の逆向きに引く距離
RETREAT_UP = 0.08  # [m]
# 区間の長さ [s]（各区間の中は min-jerk 補間）
PICK_TIMES = dict(approach=1.4, descend=0.5, insert=0.7, close=0.7, settle=0.3, lift=0.7, move=1.4,
                  lower=0.7, release=0.5, back=0.5, up=0.5)

# --- データ生成 ---
DATA_DIR = PROJECT_ROOT / "data"
GEN_SEED_BASE = 0  # データ生成のエピソード i の乱数シード = GEN_SEED_BASE + i
GEN_MAX_ATTEMPTS_FACTOR = 3  # 試行数の上限 = 保存する本数 × これ（足りなければエラー終了）
# リプレイ検証の許容差。成功判定（2cm）は方策の精度（約 0.1cm）より緩く、保存形式のバグを検出できないので、
# 関節角・environment_state の一致判定が検証の本体になる。
REPLAY_STATE_TOL = 1e-3  # [rad]
REPLAY_ENV_STATE_TOL = 1e-5
EVAL_SEED_BASE = 100_000  # 評価用（学習データと重ならないシード）
# 正規化統計（meta/stats.json）の std の下限。ほぼ動かない次元（リーチング中の指など）で
# 正規化後の値が爆発しないようにする（関節角 0.01rad ≈ 0.6°、位置 1cm）。
STATS_STD_FLOOR = 0.01

# ---------------------------------------------------------------------------
# フェーズ7 評価（scripts/06_eval_policy.py）
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # 学習済みモデル・評価結果・動画（git 管理外）
# タスクごとの既定値。モデルは Colab の手順 9 で持ってきたもの、データは学習に使ったもの（正規化統計とシードの照合用）
EVAL_POLICY_DIRS = {"reach": OUTPUTS_DIR / "act_g1_reach_v1" / "pretrained_model",
                    "pick": OUTPUTS_DIR / "act_g1_pick_v1" / "pretrained_model"}
EVAL_DATASET_DIRS = {"reach": DATA_DIR / "g1_reach_200", "pick": DATA_DIR / "g1_pick_200"}
# 画像入力のモデル（フェーズ8）を --task pick で評価するときの学習データ（--dataset-root を省略したとき、モデルの入力に画像があればこちら）
EVAL_IMAGE_DATASET_DIR = DATA_DIR / "g1_pick_img_200"
EVAL_EPISODES = 50
EVAL_MIN_SUCCESS_RATES = {"reach": 0.90, "pick": 0.70}  # 計画書の目安（リーチング 90%、ピック＆プレース 70% 以上）
# ピック＆プレースの失敗の内訳で「持ち上げた」とみなす、キューブ中心の高さの最大値と初期の高さの差 [m]（LIFT_HEIGHT は 10cm）
PICK_EVAL_LIFTED_MIN = 0.03
EVAL_STATS_TOL = 1e-5  # モデルの正規化統計と meta/stats.json の一致判定
# 動画（mujoco.Renderer）。机の向こう側（ロボットの左前）の斜め上から、ロボットと作業領域を見る
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
VIDEO_LOOKAT = (0.30, -0.10, 0.95)
VIDEO_DISTANCE = 1.3
VIDEO_AZIMUTH = -150.0  # [deg]
VIDEO_ELEVATION = -25.0  # [deg]
