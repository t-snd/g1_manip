"""スクリプト方策のデモを実行して、成功するか確認する（データは保存しない。フェーズ5）。

    python -m g1_manip demo --episodes 20                 # ビューアなしで 20 エピソード、成功率を表示
    python -m g1_manip demo --task pick --episodes 20
    mjpython -m g1_manip demo --view --episodes 3         # ビューアで表示
"""

import numpy as np

from g1_manip import config, viewer
from g1_manip.env import PickPlaceEnv, ReachEnv
from g1_manip.scripted import ScriptedPickPlacePolicy, ScriptedReachPolicy

TASKS = {"reach": (ReachEnv, ScriptedReachPolicy, config.REACH_EPISODE_STEPS),
         "pick": (PickPlaceEnv, ScriptedPickPlacePolicy, config.PICK_EPISODE_STEPS)}


def add_arguments(parser) -> None:
    parser.add_argument("--task", choices=sorted(TASKS), default="reach")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.GEN_SEED_BASE, help="最初のエピソードの乱数シード")
    parser.add_argument("--view", action="store_true", help="ビューアで表示（mjpython で実行すること）")


def run(args) -> int:
    env_cls, policy_cls, n_steps = TASKS[args.task]
    env = env_cls()
    policy = policy_cls(env)
    v = viewer.launch(env.model, env.data) if args.view else None

    n_ok = 0
    dt = 1.0 / config.CONTROL_HZ
    for ep in range(args.episodes):
        seed = args.seed + ep
        env.reset(seed)
        policy.reset()
        for _ in range(n_steps):
            env.step(policy.act())
            if v is not None:
                viewer.sync(v, dt)
        ok = env.is_success()
        n_ok += ok
        if args.task == "pick":
            print(f"episode {ep:3d} (seed {seed}): 手の向き 把持 {policy.grasp_yaw:+.0f}° / 置く {policy.place_yaw:+.0f}°"
                  f"{'' if policy.feasible else '（届く候補なし）'}  キューブの位置誤差 {env.place_error() * 100:.2f}cm  "
                  f"高さ {(env.cube_pos()[2] - env.table_top_z - config.CUBE_HALF) * 100:+.2f}cm  速さ {env.cube_speed():.4f}m/s  "
                  f"手との接触 {sorted(env.hand_cube_contacts()) or 'なし'}  {'成功' if ok else '失敗'}")
        else:
            err = np.linalg.norm(env.palm_pos() - env.goal)
            print(f"episode {ep:3d} (seed {seed}): goal={env.goal.round(3)}  手のひら誤差 {err * 100:.2f}cm  "
                  f"速さ {env.palm_speed():.4f}m/s  {'成功' if ok else '失敗'}")
    print(f"\n成功率 {n_ok}/{args.episodes} = {n_ok / args.episodes * 100:.1f}%")
    if v is not None:
        v.close()
    return 0
