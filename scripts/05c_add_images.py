"""既存のピック＆プレースのデモ（状態のみ）を再生し、頭部カメラの画像を付けたデータセットを作る（フェーズ8）。

    python scripts/05c_add_images.py                         # data/g1_pick_200 → data/g1_pick_img_200
    python scripts/05c_add_images.py --episodes 5 --dst data/g1_pick_img_test --overwrite   # 試しに 5 本
    python scripts/05c_add_images.py --check-only            # 作ったデータセットの検証だけ

- 各エピソードを、記録したシードで初期化し、保存した action をそのまま与えて再生する（シミュレーションは決定的）。
  各ステップの観測（行動を与える前）で頭部カメラ（IMAGE_HEIGHT×IMAGE_WIDTH）を描く
- 保存するのは observation.state・observation.images.head・action。environment_state（キューブ・目標の真値）は入れない
  （計画書: 画像から推定させる）。state と action は元のデータセットの値をそのまま使い、再生した関節角・environment_state が
  元の値と一致することを確かめる（一致しなければ止める）。動きは状態ベースの学習データと同じになる
- エピソードの順番・シードは元と同じ（g1_generation.json に元のデータセットを記録する）
- 検証: (1) state・action が元と完全に一致、(2) 読み込んだ画像が、その場で再生して描いた同じステップの画像と一致し、
  隣のステップの画像とは違う（1 ステップのずれも検出できる）、(3) 画像の正規化統計がある
- 描画は M1 で 1 枚約 45ms（CPU 側の mjr_render が大半）。--workers 本ずつまとめて別プロセスで並列に描き、描き終わってから
  保存する（描画と保存を同時に行うと CPU を取り合い、保存側（PNG の書き出し・parquet への埋め込み）が 1 本 80 秒ほどに遅くなった）
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config, dataset  # noqa: E402
from g1_manip import pick_analysis as pa  # noqa: E402
from g1_manip.env import PickPlaceEnv  # noqa: E402

_env = None  # 描画プロセスごとの環境


def _init_worker() -> None:
    global _env
    _env = PickPlaceEnv(images=True)


def render_episode(job: tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """1 エピソードを再生し、各ステップの画像と、元の値との最大差・成功したかを返す。"""
    row, seed, states, env_states, actions = job
    obs = _env.reset(seed)
    images = np.empty((len(actions), config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=np.uint8)
    d_state = d_env = 0.0
    for t, action in enumerate(actions):
        d_state = max(d_state, float(np.abs(obs["observation.state"] - states[t]).max()))
        d_env = max(d_env, float(np.abs(obs[pa.ENV_KEY] - env_states[t]).max()))
        images[t] = obs[config.IMAGE_KEY]
        obs = _env.step(action)
    return {"row": row, "images": images, "d_state": d_state, "d_env": d_env, "success": _env.is_success()}


def build(args) -> None:
    src = pa.load_episodes(args.src)
    n = len(src["episode_index"]) if args.episodes <= 0 else min(args.episodes, len(src["episode_index"]))
    src_info = dataset.read_generation_info(args.src)
    ds = dataset.create_dataset(args.dst, args.repo_id, None, args.overwrite, image=True, image_writer_threads=4)
    jobs = [(i, int(src["seeds"][i]), src[pa.STATE_KEY][i], src[pa.ENV_KEY][i], src["action"][i]) for i in range(n)]
    t0 = time.time()
    worst = {"d_state": 0.0, "d_env": 0.0}
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        # workers 本ずつ並列に描き（map は順番どおりに返す。エピソードの並びを元と同じにする）、描き終わってから保存する
        results = (r for b in range(0, n, args.workers) for r in pool.map(render_episode, jobs[b:b + args.workers]))
        for r in results:
            i = r["row"]
            for k in worst:
                worst[k] = max(worst[k], r[k])
            if r["d_state"] >= config.REPLAY_STATE_TOL or r["d_env"] >= config.REPLAY_ENV_STATE_TOL or not r["success"]:
                pool.terminate()
                sys.exit(f"エピソード {src['episode_index'][i]}（シード {jobs[i][1]}）の再生が元と合わない: "
                         f"関節角の差 {r['d_state']:.2e}, environment_state の差 {r['d_env']:.2e}, 成功 {r['success']}")
            for t in range(len(r["images"])):
                ds.add_frame({pa.STATE_KEY: src[pa.STATE_KEY][i, t].astype(np.float32),
                              config.IMAGE_KEY: r["images"][t],
                              "action": src["action"][i, t].astype(np.float32),
                              "task": src_info["task_name"]})
            ds.save_episode()
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n} エピソード ({time.time() - t0:.0f}s)", flush=True)
    ds.finalize()
    # PNG の一時ファイルは parquet に埋め込まれた後に消えるが、空のフォルダが残るので消す（画像は parquet にあるので中身は要らない）
    shutil.rmtree(args.dst / "images", ignore_errors=True)
    floored = dataset.apply_std_floor(args.dst)
    print(f"正規化統計の std に下限 {config.STATS_STD_FLOOR} を適用した次元数: {floored}")
    print(f"再生と元の値の最大差: 関節角 {worst['d_state']:.2e} rad, environment_state {worst['d_env']:.2e}")
    dataset.write_generation_info(args.dst, {
        **{k: src_info[k] for k in ("task", "task_name", "fps", "episode_steps", "stats_std_floor")},
        "repo_id": args.repo_id, "seeds": [int(s) for s in src["seeds"][:n]],
        "failed_seeds": src_info["failed_seeds"],  # 元のデータの生成で失敗したシード（評価のシードとの重なりの確認用）
        # 元のデータセットはこのデータセットのフォルダの親からの相対パスで記録する（worktree などで絶対パスが変わるため）
        "source_dataset": os.path.relpath(args.src.resolve(), args.dst.resolve().parent),
        "source_episode_index": [int(e) for e in src["episode_index"][:n]],
        "image_key": config.IMAGE_KEY, "image_size": [config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
        "camera": config.HEAD_CAMERA_NAME,
    })
    print(f"完了: {n} エピソード, {time.time() - t0:.0f}s → {args.dst}")


def check(args) -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = dataset.read_generation_info(args.dst)
    src = pa.load_episodes(args.src)
    n = len(info["seeds"])
    files = sorted((args.dst / "data").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=[pa.STATE_KEY, "action", "episode_index", "frame_index", "index"])
                    for f in files]).sort_values("index")
    T = info["episode_steps"]
    d_state = np.abs(np.stack(df[pa.STATE_KEY].to_numpy()).reshape(n, T, -1) - src[pa.STATE_KEY][:n]).max()
    d_action = np.abs(np.stack(df["action"].to_numpy()).reshape(n, T, -1) - src["action"][:n]).max()
    same_order = (info["source_episode_index"] == [int(e) for e in src["episode_index"][:n]]
                  and info["seeds"] == [int(s) for s in src["seeds"][:n]])

    # 画像: ランダムなエピソード・ステップを選び、その場で再生して描いた画像と比べる
    ds = LeRobotDataset(info["repo_id"], root=args.dst, video_backend="pyav")
    env = PickPlaceEnv(images=True)
    rng = np.random.default_rng(0)
    rows = rng.choice(n, size=min(args.check_episodes, n), replace=False)
    d_same, d_next = [], []
    for row in rows:
        ts = set(rng.choice(T - 1, size=3, replace=False).tolist()) | {0}
        obs = env.reset(info["seeds"][row])
        rendered = {}
        for t in range(T):
            if t in ts or t - 1 in ts:
                rendered[t] = obs[config.IMAGE_KEY]
            obs = env.step(src["action"][row, t])
        for t in sorted(ts):
            item = ds[int(row * T + t)]
            assert int(item["episode_index"]) == row and int(item["frame_index"]) == t
            img = item[config.IMAGE_KEY]  # float32 (3, H, W)、0〜1
            stored = np.rint(img.numpy().transpose(1, 2, 0) * 255).astype(np.int16)
            d_same.append(np.abs(stored - rendered[t]).max())
            d_next.append(np.abs(stored - rendered[t + 1]).mean())
    stats = json.loads((args.dst / "meta" / "stats.json").read_text())
    img_stats = stats.get(config.IMAGE_KEY, {})
    print(f"検証: {n} エピソード, {len(df)} フレーム; state の差 {d_state:.1e}, action の差 {d_action:.1e}; "
          f"画像 {len(d_same)} 枚 と再生した画像の最大差 {max(d_same)}（0〜255）、"
          f"隣のステップとの平均差 最小 {min(d_next):.2f}")
    print(f"画像の正規化統計: mean {np.ravel(img_stats.get('mean', [])).round(3).tolist()}, "
          f"std {np.ravel(img_stats.get('std', [])).round(3).tolist()}")
    checks = {
        "フレーム数 = エピソード数 × ステップ数": len(df) == n * T and ds.meta.total_frames == n * T,
        "エピソードの順番・シードが元と同じ": same_order,
        "state・action が元のデータセットと完全に一致": d_state == 0 and d_action == 0,
        "environment_state が入っていない（画像から推定させる）": pa.ENV_KEY not in ds.meta.features,
        "画像が、その場で再生して描いた同じステップの画像と一致（PNG なので完全一致）": max(d_same) == 0,
        "画像が隣のステップの画像とは違う（1 ステップのずれを検出できる）": min(d_next) > 0,
        "画像の正規化統計（mean・std、チャンネルごと）がある": all(k in img_stats for k in ("mean", "std")),
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'NG'}] {name}")
    return all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=config.DATA_DIR / "g1_pick_200")
    parser.add_argument("--dst", type=Path, default=config.DATA_DIR / "g1_pick_img_200")
    parser.add_argument("--repo-id", default="local/g1_pick_img")
    parser.add_argument("--episodes", type=int, default=0, help="先頭から何本使うか（0 = 全部）")
    parser.add_argument("--workers", type=int, default=4, help="描画するプロセスの数")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-episodes", type=int, default=10, help="画像を描き直して比べるエピソード数")
    args = parser.parse_args()
    if not args.check_only:
        build(args)
    sys.exit(0 if check(args) else 1)


if __name__ == "__main__":
    main()
