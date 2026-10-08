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
