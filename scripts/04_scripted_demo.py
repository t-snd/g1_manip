"""スクリプト方策のデモを実行して、成功するか確認する（データは保存しない）。

    python scripts/04_scripted_demo.py --episodes 20          # ビューアなしで 20 エピソード、成功率を表示
    mjpython scripts/04_scripted_demo.py --view --episodes 3  # ビューアで表示
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config  # noqa: E402
from g1_manip.env import ReachEnv  # noqa: E402
from g1_manip.scripted import ScriptedReachPolicy  # noqa: E402

TASKS = {"reach": (ReachEnv, ScriptedReachPolicy, config.REACH_EPISODE_STEPS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="reach")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.GEN_SEED_BASE, help="最初のエピソードの乱数シード")
    parser.add_argument("--view", action="store_true", help="ビューアで表示（mjpython で実行すること）")
    args = parser.parse_args()

    env_cls, policy_cls, n_steps = TASKS[args.task]
    env = env_cls()
    policy = policy_cls(env)

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(env.model, env.data)

    n_ok = 0
    dt = 1.0 / config.CONTROL_HZ
    for ep in range(args.episodes):
        seed = args.seed + ep
        env.reset(seed)
        policy.reset()
        for _ in range(n_steps):
            env.step(policy.act())
            if viewer is not None:
                if not viewer.is_running():
                    sys.exit("ビューアが閉じられたため中断しました。")
                t0 = time.time()
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - t0)))
        ok = env.is_success()
        n_ok += ok
        err = np.linalg.norm(env.palm_pos() - env.goal)
        print(f"episode {ep:3d} (seed {seed}): goal={env.goal.round(3)}  手のひら誤差 {err * 100:.2f}cm  "
              f"速さ {env.palm_speed():.4f}m/s  {'成功' if ok else '失敗'}")
    print(f"\n成功率 {n_ok}/{args.episodes} = {n_ok / args.episodes * 100:.1f}%")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
