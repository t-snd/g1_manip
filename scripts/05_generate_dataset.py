"""スクリプト方策でデモを生成し、LeRobot 形式で保存する。保存後にリプレイ検証も行う。

    python scripts/05_generate_dataset.py --task reach --num-episodes 50
    python scripts/05_generate_dataset.py --task reach --num-episodes 200 --overwrite
    python scripts/05_generate_dataset.py --task reach --num-episodes 50 --replay-only   # 既存データのリプレイ検証だけ

- 成功したエピソードだけを保存し、試行数と成功率をログに出す
- エピソード i（試行番号）の乱数シードは GEN_SEED_BASE + i。使ったシードは <root>/g1_generation.json に記録する
- リプレイ検証（保存形式のバグ検出）: 保存したデータセットを読み込み、各エピソードを同じ初期状態から
  保存した action だけで再生して、(1) タスクが成功するか、(2) 再生中の関節角が保存した observation.state と一致するか、
  (3) environment_state が初期状態と一致するか、を確かめる
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config, dataset  # noqa: E402
from g1_manip.env import ReachEnv  # noqa: E402
from g1_manip.scripted import ScriptedReachPolicy  # noqa: E402

TASKS = {
    "reach": dict(env=ReachEnv, policy=ScriptedReachPolicy, steps=config.REACH_EPISODE_STEPS,
                  env_state_names=config.REACH_ENV_STATE_NAMES),
}


def generate(args, spec, root: Path) -> None:
    env = spec["env"]()
    policy = spec["policy"](env)
    ds = dataset.create_dataset(root, args.repo_id, spec["env_state_names"], args.overwrite)

    saved_seeds, failed_seeds = [], []
    t0 = time.time()
    attempt = 0
    while len(saved_seeds) < args.num_episodes and attempt < args.max_attempts:
        seed = config.GEN_SEED_BASE + attempt
        attempt += 1
        obs = env.reset(seed)
        policy.reset()
        frames = []
        for _ in range(spec["steps"]):
            action = policy.act()
            frames.append({**obs, "action": action})
            obs = env.step(action)
        if env.is_success():
            dataset.save_episode(ds, frames, env.task_name)
            saved_seeds.append(seed)
        else:
            failed_seeds.append(seed)
        if attempt % 10 == 0:
            print(f"  試行 {attempt}: 保存 {len(saved_seeds)}, 失敗 {len(failed_seeds)} ({time.time() - t0:.0f}s)")
    ds.finalize()
    floored = dataset.apply_std_floor(root)
    print(f"正規化統計の std に下限 {config.STATS_STD_FLOOR} を適用した次元数: {floored}")

    rate = len(saved_seeds) / attempt
    print(f"\n生成完了: 試行 {attempt}, 成功（保存）{len(saved_seeds)}, 失敗 {len(failed_seeds)}, "
          f"成功率 {rate * 100:.1f}%, {time.time() - t0:.0f}s → {root}")
    dataset.write_generation_info(root, {
        "task": args.task, "task_name": env.task_name, "repo_id": args.repo_id, "fps": config.FPS,
        "episode_steps": spec["steps"], "seeds": saved_seeds, "failed_seeds": failed_seeds,
        "attempts": attempt, "success_rate": rate, "stats_std_floor": config.STATS_STD_FLOOR,
    })
    if len(saved_seeds) < args.num_episodes:
        sys.exit(f"試行数の上限 {args.max_attempts} に達し、保存できたのは {len(saved_seeds)}/{args.num_episodes} 本だけ"
                 f"（成功率 {rate * 100:.1f}%）。成功率を上げるか --max-attempts を増やすこと（データは {root} に残っている）")


def replay_check(args, spec, root: Path) -> bool:
    info = dataset.read_generation_info(root)
    episodes = dataset.load_episodes(root, info["repo_id"])
    assert len(episodes) == len(info["seeds"]), "エピソード数と記録したシードの数が一致しない"
    env = spec["env"]()
    n = len(episodes) if args.replay_episodes <= 0 else min(args.replay_episodes, len(episodes))
    n_ok, max_state_diff, max_env_diff = 0, 0.0, 0.0
    for ep, seed in list(zip(episodes, info["seeds"]))[:n]:
        obs = env.reset(seed)
        for t, action in enumerate(ep["action"]):
            max_state_diff = max(max_state_diff, np.abs(obs["observation.state"] - ep["observation.state"][t]).max())
            max_env_diff = max(max_env_diff, np.abs(obs["observation.environment_state"]
                                                    - ep["observation.environment_state"][t]).max())
            obs = env.step(action)
        n_ok += env.is_success()
    print(f"\nリプレイ検証: {n_ok}/{n} エピソード成功、関節角の最大差 {max_state_diff:.2e} rad、"
          f"environment_state の最大差 {max_env_diff:.2e}")
    stats = json.loads((root / "meta" / "stats.json").read_text())
    min_std = min(min(stats[k]["std"]) for k in ("observation.state", "observation.environment_state", "action"))
    checks = {
        f"正規化統計の std が下限 {config.STATS_STD_FLOOR} 以上": min_std >= config.STATS_STD_FLOOR - 1e-12,
        "保存した action のリプレイで全エピソード成功": n_ok == n,
        f"リプレイ中の関節角が保存した observation.state と一致 (差 < {config.REPLAY_STATE_TOL})":
            max_state_diff < config.REPLAY_STATE_TOL,
        f"リプレイ中の environment_state が保存値と全フレームで一致 (差 < {config.REPLAY_ENV_STATE_TOL})":
            max_env_diff < config.REPLAY_ENV_STATE_TOL,
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'NG'}] {name}")
    return all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="reach")
    parser.add_argument("--num-episodes", type=int, default=50, help="保存する成功エピソード数")
    parser.add_argument("--max-attempts", type=int, default=None, help="試行数の上限（既定: 保存数 × GEN_MAX_ATTEMPTS_FACTOR）")
    parser.add_argument("--root", type=Path, default=None, help="保存先（既定: data/g1_<task>_<N>）")
    parser.add_argument("--repo-id", default=None, help="LeRobot の repo_id（既定: local/g1_<task>）")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay-only", action="store_true", help="生成せず、既存データのリプレイ検証だけ行う")
    parser.add_argument("--replay-episodes", type=int, default=0, help="リプレイ検証するエピソード数（0 = 全部）")
    args = parser.parse_args()

    spec = TASKS[args.task]
    root = args.root or config.DATA_DIR / f"g1_{args.task}_{args.num_episodes}"
    args.repo_id = args.repo_id or f"local/g1_{args.task}"
    args.max_attempts = args.max_attempts or config.GEN_MAX_ATTEMPTS_FACTOR * args.num_episodes

    if not args.replay_only:
        generate(args, spec, root)
    sys.exit(0 if replay_check(args, spec, root) else 1)


if __name__ == "__main__":
    main()
