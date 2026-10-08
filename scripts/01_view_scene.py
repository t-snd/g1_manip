"""シーンを読み込み、ビューアで表示する。

    mjpython scripts/01_view_scene.py              # ビューア（macOS は mjpython 必須）
    python scripts/01_view_scene.py --headless 5   # ビューアなしで 5 秒シミュレーションして静止を確認
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config  # noqa: E402
from g1_manip.env import load_model  # noqa: E402


def print_summary(model: mujoco.MjModel) -> None:
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
          f"timestep={model.opt.timestep} (制御 {config.CONTROL_HZ}Hz = {config.N_SUBSTEPS} substeps)")
    print("cameras:", [model.camera(i).name for i in range(model.ncam)])


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> bool:
    """一定時間シミュレーションし、G1 が固定されたままでキューブが机上に静止しているか確認する。"""
    pelvis = model.body("pelvis").id
    cube = model.body(config.CUBE_BODY).id
    table_top_z = model.geom(config.TABLE_TOP_GEOM)
    table_top_z = data.geom_xpos[table_top_z.id][2] + table_top_z.size[2]
    cube_half = model.geom(config.CUBE_GEOM).size[2]
    free_joints = [model.joint(j).name for j in range(model.njnt)
                   if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]

    cube0 = data.xpos[cube].copy()
    arm_qadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    q0 = data.qpos[arm_qadr].copy()

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)

    cube_v = np.linalg.norm(data.cvel[cube])
    cube_z_err = data.xpos[cube][2] - (table_top_z + cube_half)
    checks = {
        "G1 のベースがワールドに固定 (freejoint はキューブのみ)": free_joints == [config.CUBE_JOINT],
        "キューブの移動 < 5mm": np.linalg.norm(data.xpos[cube] - cube0) < 5e-3,
        "キューブが机上 (高さ誤差 < 3mm)": abs(cube_z_err) < 3e-3,
        "キューブが静止 (速度 < 1e-3)": cube_v < 1e-3,
        "腕・手がホーム姿勢を保持 (最大誤差 < 0.05rad)": np.abs(data.qpos[arm_qadr] - q0).max() < 0.05,
    }
    print(f"t={data.time:.2f}s  pelvis={data.xpos[pelvis].round(4)}  cube={data.xpos[cube].round(4)}  "
          f"cube速度={cube_v:.2e}  腕関節の最大ずれ={np.abs(data.qpos[arm_qadr] - q0).max():.4f}rad  "
          f"接触数={data.ncon}")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'NG'}] {name}")
    return all(checks.values())


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", type=float, metavar="SECONDS",
                        help="ビューアを開かずに指定秒数シミュレーションし、静止を確認する")
    args = parser.parse_args()

    model, data = load_model()
    print_summary(model)
    if args.headless is not None:
        sys.exit(0 if run_headless(model, data, args.headless) else 1)
    run_viewer(model, data)


if __name__ == "__main__":
    main()
