"""学習済み ACT 方策をシミュレーションで動かし、成功率を出す（フェーズ7）。

    python scripts/06_eval_policy.py                                   # 既定のモデルで 50 エピソード（リーチング）
    python scripts/06_eval_policy.py --task pick                       # ピック＆プレース（既定のモデルは act_g1_pick_v1）
    python scripts/06_eval_policy.py --policy-path outputs/<RUN_NAME>/pretrained_model
    python scripts/06_eval_policy.py --n-action-steps 10                # チャンクのうち 10 個だけ実行して予測し直す
    python scripts/06_eval_policy.py --temporal-ensemble 0.01           # 毎ステップ予測し、重なった予測を平均する
    python scripts/06_eval_policy.py --episodes 5 --video outputs/eval.mp4
    mjpython scripts/06_eval_policy.py --view --episodes 3              # ビューアで表示

- 評価の乱数シードは EVAL_SEED_BASE + i（学習データの生成に使ったシードと重ならないことを確かめる）
- 実行前に、モデルの観測キー・次元が環境の観測と一致するか、モデルの正規化統計が学習データの meta/stats.json と
  一致するか（std の下限 STATS_STD_FLOOR が効いているか）、学習データの最初のエピソードと同じシードでリセットした
  観測が保存した frame 0 と一致するか（関節の並び順の照合）を確かめる
- 結果（エピソードごとの誤差・行動の跳びなど）はモデルのフォルダの隣に eval_<task>_<条件>_<N>ep.json として保存する
- ピック＆プレースでは失敗の内訳（持ち上げていない / 置き位置のずれ / 平らに置けていない / 静止していない / 手が触れたまま）も出す
- 成功率がタスクの EVAL_MIN_SUCCESS_RATES 以上なら終了コード 0、未満なら 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config, dataset  # noqa: E402
from g1_manip.env import PickPlaceEnv, ReachEnv  # noqa: E402

TASKS = {"reach": (ReachEnv, config.REACH_EPISODE_STEPS), "pick": (PickPlaceEnv, config.PICK_EPISODE_STEPS)}
# ピック＆プレースの失敗の内訳（pick_failure が返す値）
PICK_FAILURES = {"not_lifted": "持ち上げていない", "misplaced": "置き位置のずれ", "not_flat": "平らに置けていない",
                 "moving": "静止していない", "hand_contact": "手が触れたまま"}
STATS_KEYS = ("observation.state", "observation.environment_state", "action")


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_policy(path: Path, device: str, n_action_steps: int | None, te_coeff: float | None):
    """モデルと前処理・後処理を読み込む。推論の条件（n_action_steps・temporal ensembling）はここで上書きする。"""
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if te_coeff is not None:
        cfg.temporal_ensemble_coeff = te_coeff
        cfg.n_action_steps = 1  # temporal ensembling は毎ステップ予測する
    elif n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if not 1 <= cfg.n_action_steps <= cfg.chunk_size:
        sys.exit(f"--n-action-steps は 1〜chunk_size（{cfg.chunk_size}）で指定する")
    policy = ACTPolicy.from_pretrained(path, config=cfg)
    policy.eval()
    # Colab で学習したモデルは前処理の device が cuda になっているので、推論するデバイスに上書きする
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post, cfg


def model_norm_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """前処理（normalizer_processor）に保存された正規化統計（mean, std）を読む。"""
    steps = json.loads((path / "policy_preprocessor.json").read_text())["steps"]
    (step,) = [s for s in steps if s["registry_name"] == "normalizer_processor"]
    tensors = load_file(str(path / step["state_file"]))
    return {k: {s: tensors[f"{k}.{s}"].numpy() for s in ("mean", "std")} for k in STATS_KEYS}


def preflight(env, cfg, policy_path: Path, dataset_root: Path, seeds: list[int]) -> None:
    """評価の前に、観測の配線・正規化統計・シードの重なりを確かめる。入出力の形が合わなければ終了する。"""
    obs = env.observation()
    want = {k: tuple(f.shape) for k, f in cfg.input_features.items()}
    have = {k: v.shape for k, v in obs.items()}
    action_shape = tuple(cfg.output_features["action"].shape)
    print(f"モデルの入力 {want} / 出力 action {action_shape}")
    if want != have or action_shape != (len(config.CONTROLLED_JOINTS),):
        sys.exit(f"モデルの入出力と環境が合わない: 環境の観測 {have}, 行動 {len(config.CONTROLLED_JOINTS)} 次元")
    # ACT の既定の normalization_mapping には ENV が無く、その場合 environment_state は正規化されずに入る（学習時も同じ）
    env_norm = cfg.normalization_mapping.get("ENV")
    print("environment_state の正規化:",
          env_norm.value if env_norm else "なし（生の値のまま入力。normalization_mapping に ENV が無い）")

    checks = {}
    stats = model_norm_stats(policy_path)
    min_std = min(float(stats[k]["std"].min()) for k in STATS_KEYS)
    checks[f"モデルの正規化統計の std が下限 {config.STATS_STD_FLOOR} 以上（最小 {min_std:.4f}）"] = (
        min_std >= config.STATS_STD_FLOOR - 1e-6)
    if (dataset_root / "meta" / "stats.json").exists():
        ds_stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
        # 次元の違うデータ（--task とデータの取り違え）は一致しないとみなす
        diff = max(float(np.abs(stats[k][s] - np.asarray(ds_stats[k][s])).max())
                   if stats[k][s].shape == np.shape(ds_stats[k][s]) else np.inf
                   for k in STATS_KEYS for s in ("mean", "std"))
        checks[f"モデルの正規化統計が {dataset_root.name}/meta/stats.json と一致（最大差 {diff:.1e}）"] = (
            diff < config.EVAL_STATS_TOL)
    else:
        print(f"⚠️ {dataset_root}/meta/stats.json が無いので、正規化統計の照合を飛ばす")
    if (dataset_root / dataset.GEN_INFO_FILE).exists():
        info = dataset.read_generation_info(dataset_root)
        if info["task_name"] != env.task_name:
            sys.exit(f"{dataset_root} は別のタスク（{info['task_name']}）のデータ。--task と --dataset-root の組み合わせを確かめること")
        overlap = sorted(set(seeds) & set(info["seeds"] + info["failed_seeds"]))
        if overlap:
            sys.exit(f"評価のシードが学習データの生成に使ったシードと重なっている: {overlap[:10]}")
        checks[f"評価のシード（{seeds[0]}〜{seeds[-1]}）が学習データのシードと重ならない"] = True
        # 関節の並び順の照合（形と統計だけでは並び順の取り違えを検出できない）: 学習データの最初のエピソードと
        # 同じシードでリセットした観測が、保存した frame 0 と一致するか
        # 動画は無いので、デコーダは pyav に固定する（既定の torchcodec は FFmpeg が無いと長い警告を出す）
        item = LeRobotDataset(info["repo_id"], root=dataset_root, video_backend="pyav")[0]
        obs0 = env.reset(info["seeds"][int(item["episode_index"])])
        diff0 = max(float(np.abs(obs0[k] - item[k].numpy()).max()) for k in obs0)
        frame0_ok = int(item["frame_index"]) == 0 and diff0 < config.REPLAY_ENV_STATE_TOL
        checks[f"学習データの最初のエピソードと同じシードの初期観測が、保存した frame 0 と一致（最大差 {diff0:.1e}）"] = frame0_ok
        if not frame0_ok:
            for name, ok in checks.items():
                print(f"  [{'OK' if ok else 'NG'}] {name}")
            sys.exit("観測の作り方（関節の並び順など）が学習データの生成時と違う。g1_manip/env.py・config.py の変更を確かめること")
    else:
        print(f"⚠️ {dataset_root}/{dataset.GEN_INFO_FILE} が無いので、シードと初期観測の照合を飛ばす")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'NG'}] {name}")
    if not all(checks.values()):
        print("⚠️ 照合に NG がある。学習に使ったデータとモデルの組み合わせを確かめること（評価は続ける）")


class VideoWriter:
    """mujoco.Renderer で描いた画像を mp4（H.264）に書き出す。"""

    def __init__(self, path: Path, model):
        import av
        import mujoco

        path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=config.VIDEO_HEIGHT, width=config.VIDEO_WIDTH)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self.camera.lookat[:] = config.VIDEO_LOOKAT
        self.camera.distance = config.VIDEO_DISTANCE
        self.camera.azimuth = config.VIDEO_AZIMUTH
        self.camera.elevation = config.VIDEO_ELEVATION
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=config.FPS)
        self.stream.width, self.stream.height = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self._frame_cls = av.VideoFrame

    def add(self, data) -> None:
        self.renderer.update_scene(data, camera=self.camera)
        frame = self._frame_cls.from_ndarray(self.renderer.render(), format="rgb24")
        self.container.mux(self.stream.encode(frame))

    def close(self) -> None:
        self.container.mux(self.stream.encode())
        self.container.close()
        self.renderer.close()


def pick_failure(env: PickPlaceEnv, max_lift: float) -> str | None:
    """失敗の内訳（成功なら None）。持ち上げていなければ not_lifted、それ以外は PickPlaceEnv.is_success() の条件のうち
    最初に満たしていないもの。"""
    if env.is_success():
        return None
    if max_lift < config.PICK_EVAL_LIFTED_MIN:
        return "not_lifted"
    if env.place_error() > config.PICK_SUCCESS_TOL:
        return "misplaced"
    if abs(env.cube_pos()[2] - (env.table_top_z + config.CUBE_HALF)) > config.PICK_SUCCESS_Z_TOL:
        return "not_flat"
    if env.cube_speed() > config.SUCCESS_SPEED_TOL:
        return "moving"
    return "hand_contact"


def task_metrics(env, max_lift: float) -> dict:
    """タスクごとの最終状態。final_error_m はリーチングなら手のひらと目標点の距離、ピック＆プレースならキューブと目標の xy 距離。"""
    if isinstance(env, PickPlaceEnv):
        cube = env.cube_pos()
        return {
            "target": env.target.round(4).tolist(),
            "final_error_m": env.place_error(),
            "cube_final": cube.round(4).tolist(),
            "cube_z_error_m": float(cube[2] - (env.table_top_z + config.CUBE_HALF)),
            "cube_speed_mps": env.cube_speed(),
            "hand_contacts": sorted(env.hand_cube_contacts()),
            "max_lift_m": max_lift,
            "failure": pick_failure(env, max_lift),
        }
    return {
        "goal": env.goal.round(4).tolist(),
        "final_error_m": float(np.linalg.norm(env.palm_pos() - env.goal)),
        "final_speed_mps": env.palm_speed(),
    }


@torch.inference_mode()
def run_episode(env, policy, pre, post, seed: int, n_steps: int, on_step=None) -> dict:
    obs = env.reset(seed)
    policy.reset()  # 行動チャンクのキュー（temporal ensembling なら重ね合わせの途中結果）を空にする
    pick = isinstance(env, PickPlaceEnv)
    start = {}
    if pick:
        cube0, quat0 = env.cube_pos(), env.cube_quat()
        start = {"cube_start": cube0.round(4).tolist(),
                 "cube_start_yaw_deg": float(np.rad2deg(2 * np.arctan2(quat0[3], quat0[0])))}
    max_lift = 0.0  # キューブ中心の高さの最大値と初期の高さの差（ピック＆プレースのみ）
    actions = []
    infer_s = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        batch = pre({k: torch.from_numpy(v) for k, v in obs.items()})
        action = post(policy.select_action(batch)).squeeze(0).numpy()
        infer_s.append(time.perf_counter() - t0)
        actions.append(action)
        obs = env.step(action)
        if pick:
            max_lift = max(max_lift, float(env.cube_pos()[2] - cube0[2]))
        if on_step is not None:
            on_step()
    # 制御ステップ間の行動の跳び（右腕7関節の最大）。チャンクの切り替わりでカクつくかの目安
    jumps = np.abs(np.diff(np.stack(actions)[:, :len(config.RIGHT_ARM_JOINTS)], axis=0)).max(axis=1)
    return {
        "seed": seed,
        "success": bool(env.is_success()),
        **start,
        **task_metrics(env, max_lift),
        "max_action_jump_rad": float(jumps.max()),
        "max_action_jump_step": int(jumps.argmax()) + 1,
        "infer_ms_per_step": 1000 * float(np.mean(infer_s)),
        "infer_ms_max": 1000 * max(infer_s),  # 予測するステップの重さ（n_action_steps > 1 では平均に埋もれる）
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="reach")
    parser.add_argument("--policy-path", type=Path, default=None,
                        help="pretrained_model フォルダ（config.json, model.safetensors, policy_*processor* がある）。"
                             "既定: タスクごとの EVAL_POLICY_DIRS")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="学習に使ったデータセット（正規化統計とシードの照合用。無ければ照合を飛ばす）。"
                             "既定: タスクごとの EVAL_DATASET_DIRS")
    parser.add_argument("--episodes", type=int, default=config.EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED_BASE, help="最初のエピソードの乱数シード")
    parser.add_argument("--device", default="auto", help="auto（MPS があれば mps、無ければ cpu）/ mps / cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n-action-steps", type=int, default=None,
                       help="1 回の予測（chunk_size 個）のうち実際に実行する個数（既定: 学習時の設定）")
    group.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                       help="temporal ensembling の係数（ACT 論文は 0.01）。n_action_steps は 1 になる")
    parser.add_argument("--video", type=Path, default=None, help="mp4 の保存先（全エピソードを続けて録る）")
    parser.add_argument("--view", action="store_true", help="ビューアで表示（mjpython で実行すること）")
    parser.add_argument("--out", type=Path, default=None, help="結果の JSON（既定: モデルのフォルダの隣の eval_<task>_<条件>_<N>ep.json）")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes は 1 以上")

    policy_path = (args.policy_path or config.EVAL_POLICY_DIRS[args.task]).resolve()
    dataset_root = args.dataset_root or config.EVAL_DATASET_DIRS[args.task]
    min_rate = config.EVAL_MIN_SUCCESS_RATES[args.task]
    if not (policy_path / "model.safetensors").exists():
        sys.exit(f"{policy_path} にモデルが無い。Colab の手順 9 の zip を outputs/<RUN_NAME>/ に展開すること")
    env_cls, n_steps = TASKS[args.task]
    env = env_cls()
    seeds = [args.seed + i for i in range(args.episodes)]

    device = pick_device(args.device)
    policy, pre, post, cfg = load_policy(policy_path, device, args.n_action_steps, args.temporal_ensemble)
    cond = (f"te{cfg.temporal_ensemble_coeff:g}" if cfg.temporal_ensemble_coeff is not None
            else f"n{cfg.n_action_steps}")
    print(f"モデル {policy_path}（device {device}, chunk_size {cfg.chunk_size}, n_action_steps {cfg.n_action_steps}, "
          f"temporal_ensemble_coeff {cfg.temporal_ensemble_coeff}）")
    preflight(env, cfg, policy_path, dataset_root, seeds)
    # 1 回空で推論しておく（MPS は最初の呼び出しだけ準備に 100ms 以上かかり、推論時間の最大値がそれになるため）
    with torch.inference_mode():
        policy.select_action(pre({k: torch.from_numpy(v) for k, v in env.reset(seeds[0]).items()}))

    viewer = None
    if args.view:  # 動画より先に開く（mjpython 以外で失敗したとき、開いた動画ファイルを残さない）
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(env.model, env.data)
    video = VideoWriter(args.video, env.model) if args.video else None

    def on_step():
        if video is not None:
            video.add(env.data)
        if viewer is not None:
            if not viewer.is_running():
                sys.exit("ビューアが閉じられたため中断しました。")
            t0 = time.time()
            viewer.sync()
            time.sleep(max(0.0, 1.0 / config.CONTROL_HZ - (time.time() - t0)))

    results = []
    t_start = time.time()
    try:
        for ep, seed in enumerate(seeds):
            r = run_episode(env, policy, pre, post, seed, n_steps, on_step)
            results.append(r)
            jump = f"行動の跳び 最大 {r['max_action_jump_rad']:.3f}rad（ステップ {r['max_action_jump_step']}）"
            if args.task == "pick":
                print(f"episode {ep:3d} (seed {seed}): cube={np.round(r['cube_start'][:2], 3)} "
                      f"yaw {r['cube_start_yaw_deg']:+.0f}° → target={np.round(r['target'][:2], 3)}  "
                      f"置き誤差 {r['final_error_m'] * 100:.2f}cm  持ち上げ {r['max_lift_m'] * 100:.1f}cm  {jump}  "
                      f"{'成功' if r['success'] else '失敗（' + PICK_FAILURES[r['failure']] + '）'}")
            else:
                print(f"episode {ep:3d} (seed {seed}): goal={np.round(r['goal'], 3)}  手のひら誤差 {r['final_error_m'] * 100:.2f}cm  "
                      f"速さ {r['final_speed_mps']:.4f}m/s  {jump}  {'成功' if r['success'] else '失敗'}")
    finally:
        if video is not None:
            video.close()
        if viewer is not None:
            viewer.close()

    n_ok = sum(r["success"] for r in results)
    rate = n_ok / len(results)
    errs = np.array([r["final_error_m"] for r in results])
    summary = {
        "policy_path": str(policy_path), "device": device, "task": args.task, "episodes": len(results),
        "seeds": [seeds[0], seeds[-1]], "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
        "temporal_ensemble_coeff": cfg.temporal_ensemble_coeff, "success_rate": rate,
        "final_error_m": {"mean": float(errs.mean()), "median": float(np.median(errs)), "max": float(errs.max())},
        "max_action_jump_rad": max(r["max_action_jump_rad"] for r in results),
        "infer_ms_per_step": float(np.mean([r["infer_ms_per_step"] for r in results])),
        "infer_ms_max": max(r["infer_ms_max"] for r in results),
        "wall_time_s": time.time() - t_start,
    }
    if args.task == "pick":
        summary["failures"] = {k: sum(r["failure"] == k for r in results) for k in PICK_FAILURES}
    out = args.out or policy_path.parent / f"eval_{args.task}_{cond}_{len(results)}ep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2, ensure_ascii=False))

    print(f"\n成功率 {n_ok}/{len(results)} = {rate * 100:.1f}%（目安 {min_rate * 100:.0f}% 以上）")
    err_stats = f"平均 {errs.mean() * 100:.2f}cm, 中央値 {np.median(errs) * 100:.2f}cm, 最大 {errs.max() * 100:.2f}cm"
    if args.task == "pick":
        print(f"最終ステップのキューブと目標の xy 距離: {err_stats}（成功は {config.PICK_SUCCESS_TOL * 100:.0f}cm 以内で、"
              f"平らに置かれて静止し、手が離れている）")
        print("失敗の内訳: " + ", ".join(f"{PICK_FAILURES[k]} {n}" for k, n in summary["failures"].items()))
    else:
        print(f"最終ステップの手のひら誤差: {err_stats}（成功は {config.REACH_SUCCESS_TOL * 100:.0f}cm 以内かつ静止）")
    print(f"行動の跳び（右腕、制御ステップ間）: 最大 {summary['max_action_jump_rad']:.3f}rad, "
          f"推論 平均 {summary['infer_ms_per_step']:.1f}ms/ステップ（最大 {summary['infer_ms_max']:.1f}ms）, 経過 {summary['wall_time_s']:.0f}s")
    failed = [r["seed"] for r in results if not r["success"]]
    if failed:
        print(f"失敗したシード: {failed}")
    print(f"結果を保存: {out}")
    if video is not None:
        print(f"動画を保存: {args.video}")
    sys.exit(0 if rate >= min_rate else 1)


if __name__ == "__main__":
    main()
