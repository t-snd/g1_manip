"""IK の到達テスト：机上のランダムな点に右手のひらを持っていく（物理シミュレーションで動かす）。

    python scripts/03_ik_reach_test.py            # ビューアなしで実行し、合否を判定
    mjpython scripts/03_ik_reach_test.py --view   # ビューアで表示しながら実行

手順:
  - 準備姿勢（手を机の上に上げ、手のひらを下に向けた姿勢）から始める
  - 点ごとに「上方の通過点（天板＋REACH_APPROACH_HEIGHT）へ水平移動 → 到達点（天板＋REACH_HEIGHT）へ下降
    → REACH_HOLD_TIME 静止 → 手のひらの位置誤差を計測 → 上昇」を行う
  - 経路は手先空間で min-jerk 補間し、50Hz の制御周期ごとに IK（ウォームスタート）で右腕の目標角を求めて
    位置制御に渡す。指はホーム姿勢のまま
  - 点はキューブの初期位置の範囲から REACH_N_CUBE 点、目標マーカーの範囲から REACH_N_TARGET 点
合格条件: 手のひらの位置誤差（物理側）が REACH_TOL 以内の点が REACH_MIN_SUCCESS 点以上。
あわせて、ロボットと机・ロボット同士（右腕と胴体など）の接触の有無を報告する。
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config  # noqa: E402
from g1_manip.env import (  # noqa: E402
    apply_joint_targets,
    joint_actuator_ids,
    load_model,
    reset_to_ready,
)
from g1_manip.ik import RightHandIK  # noqa: E402

DT = 1.0 / config.CONTROL_HZ


def sample_points(rng: np.random.Generator) -> np.ndarray:
    """キューブの範囲と目標マーカーの範囲から到達点の xy を取る。"""
    cube = config.CUBE_XY_NOMINAL + rng.uniform(-1, 1, (config.REACH_N_CUBE, 2)) * config.CUBE_XY_RANGE
    target = config.TARGET_XY_NOMINAL + rng.uniform(-1, 1, (config.REACH_N_TARGET, 2)) * config.TARGET_XY_RANGE
    return np.concatenate([cube, target])


def min_jerk(s: float) -> float:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", action="store_true", help="ビューアで表示しながら実行（mjpython で実行すること）")
    args = parser.parse_args()

    model, data = load_model()
    ik = RightHandIK(model)
    reset_to_ready(model, data, ik)
    # 到達だけを測るため、キューブは手の届かない机の奥の隅に置いておく
    cube_q = model.joint(config.CUBE_JOINT).qposadr[0]
    data.qpos[cube_q:cube_q + 2] = config.REACH_CUBE_PARK_XY
    mujoco.mj_forward(model, data)
    ik.reset(data.qpos)

    arm_act = joint_actuator_ids(model, config.RIGHT_ARM_JOINTS)
    palm = model.site(config.RIGHT_PALM_SITE).id
    table = model.geom(config.TABLE_TOP_GEOM)
    table_top = data.geom_xpos[table.id][2] + table.size[2]
    robot_root = model.body("pelvis").id
    right_arm_bodies = {model.joint(j).bodyid[0] for j in config.RIGHT_ARM_JOINTS + config.RIGHT_HAND_JOINTS}

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    contacts = set()
    max_ik_err = 0.0

    def run_segment(p0, p1, duration):
        """手のひらを p0 → p1 へ min-jerk 補間で動かす（制御周期ごとに IK → 位置制御）。"""
        nonlocal max_ik_err
        n = max(1, int(round(duration / DT)))
        for k in range(n):
            target = p0 + min_jerk((k + 1) / n) * (p1 - p0)
            apply_joint_targets(model, data, arm_act, ik.solve(target))
            max_ik_err = max(max_ik_err, ik.errors()[0])
            # step_control と同じだが、接触を取りこぼさないよう物理ステップごとに調べる
            for _ in range(config.N_SUBSTEPS):
                mujoco.mj_step(model, data)
                for c in data.contact[:data.ncon]:
                    b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                    r1, r2 = model.body_rootid[b1] == robot_root, model.body_rootid[b2] == robot_root
                    if (r1 != r2 or (r1 and r2)) and (b1 in right_arm_bodies or b2 in right_arm_bodies):
                        contacts.add(f"{model.body(b1).name}<->{model.body(b2).name}")
            if viewer is not None:
                if not viewer.is_running():
                    sys.exit("ビューアが閉じられたため、テストを中断しました。")
                t0 = time.time()
                viewer.sync()
                time.sleep(max(0.0, DT - (time.time() - t0)))

    points = sample_points(np.random.default_rng(config.REACH_SEED))
    z_up = table_top + config.REACH_APPROACH_HEIGHT
    z_reach = table_top + config.REACH_HEIGHT
    current = np.array(config.READY_PALM_POS, dtype=float)

    print(f"到達テスト: {len(points)} 点、手のひらの高さ 天板+{config.REACH_HEIGHT}m（通過点 天板+{config.REACH_APPROACH_HEIGHT}m）、"
          f"合格 {config.REACH_TOL * 100:.0f}cm 以内")
    print(f"{'#':>3} {'範囲':<6} {'x':>6} {'y':>6} {'誤差[cm]':>8} {'IK誤差[cm]':>10}  判定")
    errors = []
    for i, (x, y) in enumerate(points):
        above = np.array([x, y, z_up])
        goal = np.array([x, y, z_reach])
        run_segment(current, above, config.REACH_MOVE_TIME)
        run_segment(above, goal, config.REACH_DESCEND_TIME)
        run_segment(goal, goal, config.REACH_HOLD_TIME)
        err = np.linalg.norm(data.site_xpos[palm] - goal)
        ik_err = np.linalg.norm(ik.palm_pose().translation() - goal)
        errors.append(err)
        region = "cube" if i < config.REACH_N_CUBE else "target"
        print(f"{i:>3} {region:<6} {x:>6.3f} {y:>6.3f} {err * 100:>8.2f} {ik_err * 100:>10.2f}  "
              f"{'OK' if err <= config.REACH_TOL else 'NG'}")
        run_segment(goal, above, config.REACH_DESCEND_TIME)
        current = above

    errors = np.array(errors)
    n_ok = int((errors <= config.REACH_TOL).sum())
    print(f"\n成功 {n_ok}/{len(points)} 点（誤差 平均 {errors.mean() * 100:.2f}cm, 最大 {errors.max() * 100:.2f}cm）、"
          f"経路中の IK 誤差の最大 {max_ik_err * 100:.2f}cm")
    print(f"右腕・右手の接触（机・胴体など）: {sorted(contacts) if contacts else 'なし'}")
    ok = n_ok >= config.REACH_MIN_SUCCESS
    print(f"\n  [{'OK' if ok else 'NG'}] {config.REACH_TOL * 100:.0f}cm 以内に到達した点が "
          f"{config.REACH_MIN_SUCCESS}/{len(points)} 点以上")

    if viewer is not None:
        print("ビューアを閉じると終了します。")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
