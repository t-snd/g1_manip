"""ピック＆プレースの学習済み方策を、学習データ・教師（スクリプト方策）と比べる（フェーズ7の追加分析）。

    python scripts/07_analyze_pick.py dist                         # 学習データと評価の初期配置の分布・近さ
    python scripts/07_analyze_pick.py traj                         # 方策の軌跡を教師・最も近い学習エピソードと比べる（50 シード）
    python scripts/07_analyze_pick.py traj --temporal-ensemble 0.01
    python scripts/07_analyze_pick.py grid --with-scripted         # キューブの位置を格子状に振って成功マップを作る

- dist: 学習データ 200 本の初期配置（キューブ xy・yaw・目標 xy）と、06_eval_policy.py の結果 JSON の初期配置を並べ、
  各評価エピソードに最も近い学習エピソードとの距離（正規化した environment_state の距離・キューブ xy の距離）を出す
- traj: 評価シードごとに (1) 方策の推論 (2) 同じシードの教師（スクリプト方策、＝その配置での正解の軌跡）
  (3) 最近傍リプレイ（初期配置が最も近い学習エピソードの行動列を、その配置でそのまま再生する。方策が学習データを
  丸暗記して再生しているだけなら、これと同じくらいの結果になる）を動かし、成功と手のひらの軌跡を比べる。
  手のひらの位置は observation.state から順運動学で求める。参考に、学習データの全エピソードのうち把持の区間で
  教師に最も近いもの（正解を知って選んだ上限で、方策にもリプレイにも選べない）とも比べる
- grid: キューブの yaw・目標・ロボットの初期姿勢（シード）を固定し、キューブの xy を格子状に置いて方策（と教師・最近傍リプレイ）を動かす。
  学習データの範囲の外も含めて、どこまで取れるかを見る
- 出力はモデルのフォルダの隣の analysis/（図 PNG・結果 JSON・軌跡 npz）。--out-dir で変えられる
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config  # noqa: E402
from g1_manip import pick_analysis as pa  # noqa: E402
from g1_manip.env import PickPlaceEnv  # noqa: E402
from g1_manip.scripted import ScriptedPickPlacePolicy  # noqa: E402

# モデルの読み込み・失敗の分類は評価スクリプトと同じものを使う（ファイル名が数字で始まるので import 文では読めない）
_spec = importlib.util.spec_from_file_location("eval_policy", Path(__file__).with_name("06_eval_policy.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)



def _seg_end(last: str) -> int:
    """教師（スクリプト方策）の区間 last が終わるステップ（PICK_TIMES の先頭からの和）。"""
    names = list(config.PICK_TIMES)
    return int(round(sum(config.PICK_TIMES[k] for k in names[:names.index(last) + 1]) * config.CONTROL_HZ))


# 軌跡を比べる区間（教師の区間で決める）。APPROACH: 準備姿勢からキューブの横へ近づき、手を差し入れる（0〜2.6 秒）。
# HOLD: 手のひらが把持の位置でほぼ止まり、指を閉じる（2.6〜3.6 秒）。T_GRASP は HOLD の終わり（教師の lift 開始の直前）
T_APPROACH_END = _seg_end("insert")
T_GRASP = _seg_end("settle")
HOLD = slice(T_APPROACH_END, T_GRASP)
APPROACH = slice(0, T_APPROACH_END)

# 色（dataviz スキルの既定パレットの 1〜3 番。学習データは無彩色）
C_TEACHER, C_POLICY, C_NN, C_TRAIN = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
C_OK, C_NG = C_TEACHER, C_POLICY
plt.rcParams.update({
    "font.family": ["Hiragino Sans", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False, "figure.dpi": 110,
})


def env_state0(cube_start, yaw_deg: float, target) -> np.ndarray:
    """評価結果 JSON の初期配置から、リセット直後の environment_state（10）を組み立てる（初期のキューブは z まわりの回転だけ）。"""
    h = np.deg2rad(yaw_deg) / 2
    return np.array([*cube_start, np.cos(h), 0.0, 0.0, np.sin(h), *target])


def setup_topview(ax, title: str, target_range: bool = True) -> None:
    """上から見た図。縦軸 x（上がロボットの前方）、横軸 y（右がロボットの右手側 = -y）。学習時の範囲を破線で描く。"""
    ax.set_title(title, loc="left")
    ax.set_xlabel("y [m]（右がロボットの右手側）")
    ax.set_ylabel("x [m]（上がロボットの前方）")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ranges = [(config.PICK_CUBE_X, config.PICK_CUBE_Y, "キューブの初期位置の範囲")]
    if target_range:
        ranges.append((config.PICK_TARGET_X, config.PICK_TARGET_Y, "置き場所（目標）の範囲"))
    for (x0, x1), (y0, y1), label in ranges:
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ls="--", lw=0.9, ec="#52514e", zorder=5))
        ax.annotate(label, (y0, x1), textcoords="offset points", xytext=(-2, 3), ha="right", fontsize=7, color="#52514e",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# ---------------------------------------------------------------------------
# dist
# ---------------------------------------------------------------------------
def cmd_dist(args, policy_path: Path) -> None:
    nb = pa.TrainingNeighbors(args.dataset_root)
    tr = nb.env0
    ref = nb.self_nn()
    tr_xy_nn = np.linalg.norm(tr[:, None, :2] - tr[None, :, :2], axis=2)
    np.fill_diagonal(tr_xy_nn, np.inf)
    out_dir = args.out_dir or policy_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 既定では 06_eval_policy.py をシードどおりの配置で回した結果だけを使う（--cube / --target を付けた結果は除く）
    evals = args.evals or [p for p in sorted(policy_path.parent.glob("eval_pick_*_50ep.json"))
                           if not json.loads(p.read_text())["summary"].get("place")]
    report = {"dataset": str(args.dataset_root), "train_episodes": len(tr),
              "train_self_nn_norm": {"median": float(np.median(ref)), "max": float(ref.max())},
              "train_self_nn_cube_xy_m": {"median": float(np.median(tr_xy_nn.min(1))), "max": float(tr_xy_nn.min(1).max())},
              "evals": {}}
    print(f"学習データ {len(tr)} 本: 学習エピソードどうしの最近傍 正規化距離 中央値 {np.median(ref):.2f} / 最大 {ref.max():.2f}, "
          f"キューブ xy 中央値 {np.median(tr_xy_nn.min(1)) * 100:.1f}cm / 最大 {tr_xy_nn.min(1).max() * 100:.1f}cm")
    rows_by_eval = {}
    for path in evals:
        eps = json.loads(Path(path).read_text())["episodes"]
        rows = []
        for e in eps:
            nn = nb.nearest(env_state0(e["cube_start"], e["cube_start_yaw_deg"], e["target"]))
            rows.append({"seed": e["seed"], "success": e["success"], "cube_x": e["cube_start"][0],
                         "cube_y": e["cube_start"][1], "yaw_deg": e["cube_start_yaw_deg"],
                         "target_x": e["target"][0], "target_y": e["target"][1], **nn})
        rows_by_eval[Path(path).stem] = rows
        d = np.array([r["dist_norm"] for r in rows])
        ok = np.array([r["success"] for r in rows])
        xy = np.array([r["min_cube_xy_m"] for r in rows])
        s = {"episodes": len(rows), "success": int(ok.sum()),
             "dist_norm_median": float(np.median(d)), "dist_norm_max": float(d.max()),
             "dist_norm_median_success": float(np.median(d[ok])) if ok.any() else None,
             "dist_norm_median_fail": float(np.median(d[~ok])) if (~ok).any() else None,
             "beyond_train_max_nn": int((d > ref.max()).sum()),
             "min_cube_xy_m_median": float(np.median(xy)), "min_cube_xy_m_max": float(xy.max()),
             "rows": rows}
        report["evals"][Path(path).stem] = s
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{Path(path).stem}: 最近傍の正規化距離 中央値 {s['dist_norm_median']:.2f}（成功 {fmt(s['dist_norm_median_success'])}, "
              f"失敗 {fmt(s['dist_norm_median_fail'])}）, 最大 {s['dist_norm_max']:.2f}, 学習どうしの最大 {ref.max():.2f} を超える {s['beyond_train_max_nn']} 件, "
              f"キューブ xy の最近傍 中央値 {s['min_cube_xy_m_median'] * 100:.1f}cm / 最大 {s['min_cube_xy_m_max'] * 100:.1f}cm")

    # 図: 代表として args.show の評価（既定は最初の JSON）を色分けする
    show = args.show or next(iter(rows_by_eval))
    rows = rows_by_eval[show]
    ok = np.array([r["success"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    cy, cx = np.array([r["cube_y"] for r in rows]), np.array([r["cube_x"] for r in rows])
    ty, tx = np.array([r["target_y"] for r in rows]), np.array([r["target_x"] for r in rows])
    ax = axes[0]
    setup_topview(ax, "(a) 初期配置（上から見た図）: キューブ ● と置き場所 ▲")
    ax.scatter(tr[:, 1], tr[:, 0], s=10, c=C_TRAIN, label=f"学習データ（{len(tr)} 本）", zorder=2)
    ax.scatter(tr[:, 8], tr[:, 7], s=12, c=C_TRAIN, marker="^", zorder=2)
    ax.scatter(cy[ok], cx[ok], s=36, c=C_OK, ec="white", lw=0.8, label=f"評価: 方策が成功（{ok.sum()}）", zorder=3)
    ax.scatter(ty[ok], tx[ok], s=40, c=C_OK, marker="^", ec="white", lw=0.8, zorder=3)
    ax.scatter(cy[~ok], cx[~ok], s=46, c=C_NG, marker="X", ec="white", lw=0.6, label=f"評価: 方策が失敗（{(~ok).sum()}、× はキューブ）", zorder=4)
    ax.scatter(ty[~ok], tx[~ok], s=40, c=C_NG, marker="^", ec="white", lw=0.6, zorder=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5)
    ax = axes[1]
    ax.set_title("(b) キューブの初期位置の x と yaw", loc="left")
    ax.scatter(tr[:, 0], pa.cube_yaw_deg(tr), s=10, c=C_TRAIN, label="学習データ")
    yaw = np.array([r["yaw_deg"] for r in rows])
    ax.scatter(cx[ok], yaw[ok], s=36, c=C_OK, ec="white", lw=0.8, label="評価: 方策が成功")
    ax.scatter(cx[~ok], yaw[~ok], s=46, c=C_NG, marker="X", ec="white", lw=0.6, label="評価: 方策が失敗")
    ax.set_xlabel("キューブ x [m]（大きいほどロボットから遠い＝奥）")
    ax.set_ylabel("キューブ yaw [deg]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.5)
    ax = axes[2]
    ax.set_title("(c) 初期配置が最も近い学習エピソードまでの距離\n（キューブの位置・向きと置き場所をまとめた距離）", loc="left")
    rng = np.random.default_rng(0)
    groups = [("学習どうし", ref, C_TRAIN, "o"), ("評価 成功", np.array([r["dist_norm"] for r in rows])[ok], C_OK, "o"),
              ("評価 失敗", np.array([r["dist_norm"] for r in rows])[~ok], C_NG, "X")]
    for i, (label, vals, c, m) in enumerate(groups):
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=22 if m == "o" else 34, c=c, marker=m,
                   ec="white", lw=0.5, zorder=3)
        if len(vals):
            ax.plot([np.median(vals)] * 2, [i - 0.3, i + 0.3], c="#0b0b0b", lw=1.5, zorder=4)
    ax.set_yticks(range(len(groups)), ["学習エピソード\nどうし", "評価: 方策が成功", "評価: 方策が失敗"])
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("正規化した environment_state の距離\n（モデルの入力と同じ正規化。黒い縦線は中央値）")
    ax.grid(axis="y", visible=False)
    fig.suptitle(f"学習データ（{len(tr)} 本）と評価（{show}）の初期配置", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = out_dir / "dist.png"
    fig.savefig(png, bbox_inches="tight")  # (c) の 2 行の見出しが右端で切れないように
    out = out_dir / "dist.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"図を保存: {png}\n結果を保存: {out}")


# ---------------------------------------------------------------------------
# traj / grid 共通
# ---------------------------------------------------------------------------
def rollout(env: PickPlaceEnv, act, n_steps: int) -> dict:
    """env はリセット済み。act(obs) -> 行動 で n_steps 進め、各ステップの観測時点の関節角とキューブの位置を記録する。"""
    obs = env.observation()
    cube0 = env.cube_pos()
    states, cubes, max_lift = [], [], 0.0
    for _ in range(n_steps):
        states.append(obs[pa.STATE_KEY].astype(np.float64))
        cubes.append(env.cube_pos())
        obs = env.step(act(obs))
        max_lift = max(max_lift, float(env.cube_pos()[2] - cube0[2]))
    return {"states": np.array(states), "cube": np.array(cubes), "success": bool(env.is_success()),
            "failure": ev.pick_failure(env, max_lift), "place_error_m": env.place_error(), "max_lift_m": max_lift}


class Runner:
    """方策と教師を同じ配置で動かす。"""

    def __init__(self, args, policy_path: Path):
        self.env = PickPlaceEnv()
        self.device = ev.pick_device(args.device)
        self.policy, self.pre, self.post, cfg = ev.load_policy(policy_path, self.device, args.n_action_steps,
                                                               args.temporal_ensemble)
        self.cond = (f"te{cfg.temporal_ensemble_coeff:g}" if cfg.temporal_ensemble_coeff is not None
                     else f"n{cfg.n_action_steps}")
        self.teacher = ScriptedPickPlacePolicy(self.env)

    @torch.inference_mode()
    def policy_rollout(self, seed: int, place: dict) -> dict:
        self.env.reset(seed, **place)
        self.policy.reset()
        to_t = lambda o: {k: torch.from_numpy(v) for k, v in o.items()}  # noqa: E731
        return rollout(self.env, lambda o: self.post(self.policy.select_action(self.pre(to_t(o)))).squeeze(0).numpy(),
                       config.PICK_EPISODE_STEPS)

    def replay_rollout(self, seed: int, place: dict, actions: np.ndarray) -> dict:
        """学習エピソードの行動列（絶対目標角）を、観測を見ずにそのまま再生する。"""
        self.env.reset(seed, **place)
        it = iter(actions)
        return rollout(self.env, lambda o: next(it), len(actions))

    def teacher_rollout(self, seed: int, place: dict) -> dict:
        self.env.reset(seed, **place)
        self.teacher.reset()
        r = rollout(self.env, lambda o: self.teacher.act(), config.PICK_EPISODE_STEPS)
        r["teacher_feasible"] = bool(self.teacher.feasible)
        return r


# ---------------------------------------------------------------------------
# traj
# ---------------------------------------------------------------------------
def dev_stats(d: np.ndarray) -> dict:
    return {"mean_m": float(d.mean()), "max_m": float(d.max()), "at_grasp_m": float(d[T_GRASP - 1]),
            "approach_mean_m": float(d[APPROACH].mean()), "hold_mean_m": float(d[HOLD].mean())}


def best_train(train_palms: np.ndarray, p_teacher: np.ndarray) -> dict:
    """学習データの全エピソードのうち、把持の区間（HOLD）で手のひらが教師に最も近いもの。
    方策が学習データのどれか 1 本を再生しているだけなら、HOLD での教師とのずれはこれより小さくならない。"""
    d_hold = np.linalg.norm(train_palms[:, HOLD] - p_teacher[HOLD], axis=2).mean(axis=1)
    i = int(d_hold.argmin())
    return {"train_row": i, "hold_mean_m": float(d_hold[i]),
            "at_grasp_m": float(np.linalg.norm(train_palms[i, T_GRASP - 1] - p_teacher[T_GRASP - 1]))}


def cmd_traj(args, policy_path: Path) -> None:
    if args.replot:  # 保存した結果（JSON と npz）から図だけ描き直す
        res = json.loads(args.replot.read_text())
        npz = np.load(args.replot.with_suffix(".npz"))
        palms = {r["seed"]: {k.split("_", 1)[1]: npz[k] for k in npz.files if k.split("_", 1)[0] == str(r["seed"])}
                 for r in res["episodes"]}
        plot_traj(res["episodes"], palms, res["summary"]["condition"], args.replot.parent, args.examples)
        return
    run = Runner(args, policy_path)
    nb = pa.TrainingNeighbors(args.dataset_root)
    env = run.env
    seeds = [args.seed + i for i in range(args.episodes)]
    out_dir = args.out_dir or policy_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_palms = np.stack([pa.palm_positions(env, st) for st in nb.episodes[pa.STATE_KEY]])  # (エピソード数, T, 3)
    rows, palms = [], {}
    t0 = time.time()
    for seed in seeds:
        obs0 = env.reset(seed)
        target = env.target.copy()
        nn = nb.nearest(obs0[pa.ENV_KEY])
        pol = run.policy_rollout(seed, {})
        tea = run.teacher_rollout(seed, {})
        p_pol = pa.palm_positions(env, pol["states"])
        p_tea = pa.palm_positions(env, tea["states"])
        rep = run.replay_rollout(seed, {}, nb.episodes["action"][nn["train_row"]])
        p_rep = pa.palm_positions(env, rep["states"])
        p_nn = train_palms[nn["train_row"]]
        best = best_train(train_palms, p_tea)
        d_rt = np.linalg.norm(p_rep - p_tea, axis=1)
        d_pt = np.linalg.norm(p_pol - p_tea, axis=1)
        d_pn = np.linalg.norm(p_pol - p_nn, axis=1)
        d_tn = np.linalg.norm(p_tea - p_nn, axis=1)
        # 右腕 7 関節の角度の差（RMS over joints）
        arm = slice(0, len(config.RIGHT_ARM_JOINTS))
        j_pt = np.sqrt(((pol["states"][:, arm] - tea["states"][:, arm]) ** 2).mean(1))
        j_pn = np.sqrt(((pol["states"][:, arm] - nb.episodes[pa.STATE_KEY][nn["train_row"]][:, arm]) ** 2).mean(1))
        row = {"seed": seed, "policy_success": pol["success"], "policy_failure": pol["failure"],
               "policy_place_error_m": pol["place_error_m"], "teacher_success": tea["success"],
               "teacher_feasible": tea["teacher_feasible"], "train_nn": nn,
               "replay_success": rep["success"], "replay_failure": rep["failure"], "replay_place_error_m": rep["place_error_m"],
               "palm_replay_vs_teacher": dev_stats(d_rt),
               "palm_policy_vs_teacher": dev_stats(d_pt), "palm_policy_vs_nn": dev_stats(d_pn),
               "palm_teacher_vs_nn": dev_stats(d_tn),
               "best_train_vs_teacher": {**best, "train_episode": int(nb.episodes["episode_index"][best["train_row"]])},
               "beats_best_train_hold": bool(d_pt[HOLD].mean() < best["hold_mean_m"]),
               "arm_rms_policy_vs_teacher_rad": float(j_pt.mean()), "arm_rms_policy_vs_nn_rad": float(j_pn.mean()),
               "closer_to_teacher": bool(d_pt.mean() < d_pn.mean())}
        rows.append(row)
        palms[seed] = {"policy": p_pol, "teacher": p_tea, "nn": p_nn, "replay": p_rep, "cube_replay": rep["cube"],
                       "best": train_palms[best["train_row"]], "d_rt": d_rt,
                       "cube_policy": pol["cube"],
                       "cube_teacher": tea["cube"], "target": target, "d_pt": d_pt, "d_pn": d_pn, "d_tn": d_tn}
        print(f"seed {seed}: 方策 {'成功' if pol['success'] else '失敗'} / 教師 {'成功' if tea['success'] else '失敗'} / "
              f"最近傍リプレイ {'成功' if rep['success'] else '失敗'}  "
              f"手のひらのずれ 平均（最大）: 方策–教師 {d_pt.mean() * 100:.1f}（{d_pt.max() * 100:.1f}）cm, "
              f"方策–最近傍の学習 {d_pn.mean() * 100:.1f}（{d_pn.max() * 100:.1f}）cm, "
              f"教師–最近傍の学習 {d_tn.mean() * 100:.1f}cm  [最近傍 ep {nn['train_episode']}, 正規化距離 {nn['dist_norm']:.2f}]  "
              f"把持区間の教師とのずれ: 方策 {d_pt[HOLD].mean() * 100:.2f}cm, 最近傍リプレイ {d_rt[HOLD].mean() * 100:.2f}cm, "
              f"学習 {len(train_palms)} 本中の最良 {best['hold_mean_m'] * 100:.2f}cm")

    def agg(key, sub, sel=None):
        v = np.array([r[key][sub] for r in rows if sel is None or sel(r)])
        return {"median_m": float(np.median(v)), "max_m": float(v.max())} if len(v) else None

    keys3 = ("palm_policy_vs_teacher", "palm_replay_vs_teacher", "palm_policy_vs_nn", "palm_teacher_vs_nn")

    ok = lambda r: r["policy_success"]  # noqa: E731
    summary = {"condition": run.cond, "seeds": [seeds[0], seeds[-1]], "episodes": len(rows),
               "t_approach_end_step": T_APPROACH_END, "t_grasp_step": T_GRASP, "train_episodes": len(train_palms),
               "policy_success": sum(r["policy_success"] for r in rows),
               "teacher_success": sum(r["teacher_success"] for r in rows),
               "replay_success": sum(r["replay_success"] for r in rows),
               "replay_failures": {k: sum(r["replay_failure"] == k for r in rows) for k in ev.PICK_FAILURES},
               "closer_to_teacher": sum(r["closer_to_teacher"] for r in rows),
               "closer_to_teacher_success_only": sum(r["closer_to_teacher"] for r in rows if ok(r)),
               "beats_best_train_hold": sum(r["beats_best_train_hold"] for r in rows),
               "beats_best_train_hold_success_only": sum(r["beats_best_train_hold"] for r in rows if ok(r)),
               "palm_mean_dev": {k: agg(k, "mean_m") for k in keys3},
               "palm_approach_dev": {k: agg(k, "approach_mean_m") for k in keys3},
               "palm_hold_dev": {**{k: agg(k, "hold_mean_m") for k in keys3},
                                 "best_train_vs_teacher": agg("best_train_vs_teacher", "hold_mean_m")},
               "palm_mean_dev_success_only": {k: agg(k, "mean_m", ok) for k in ("palm_policy_vs_teacher", "palm_policy_vs_nn")},
               "palm_dev_at_grasp": {**{k: agg(k, "at_grasp_m") for k in keys3},
                                     "best_train_vs_teacher": agg("best_train_vs_teacher", "at_grasp_m")},
               "wall_time_s": time.time() - t0}
    out = out_dir / f"traj_{run.cond}.json"
    out.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2, ensure_ascii=False))
    np.savez_compressed(out_dir / f"traj_{run.cond}.npz",
                        **{f"{s}_{k}": v for s, d in palms.items() for k, v in d.items()})
    pm = summary["palm_mean_dev"]
    print(f"\n成功: 方策 {summary['policy_success']}/{len(rows)}, 最近傍リプレイ {summary['replay_success']}/{len(rows)}, "
          f"教師 {summary['teacher_success']}/{len(rows)}")
    print(f"手のひらのずれ（エピソード平均の中央値）: 方策–教師 {pm['palm_policy_vs_teacher']['median_m'] * 100:.2f}cm, "
          f"方策–最近傍の学習 {pm['palm_policy_vs_nn']['median_m'] * 100:.2f}cm, 教師–最近傍の学習 {pm['palm_teacher_vs_nn']['median_m'] * 100:.2f}cm")
    print(f"方策の軌跡が最近傍の学習エピソードより教師に近い: {summary['closer_to_teacher']}/{len(rows)}"
          f"（成功のみ {summary['closer_to_teacher_success_only']}/{summary['policy_success']}）")
    ph = summary["palm_hold_dev"]
    print(f"把持の区間（{T_APPROACH_END / config.CONTROL_HZ:.1f}〜{T_GRASP / config.CONTROL_HZ:.1f} 秒）の教師とのずれ（中央値）: "
          f"方策 {ph['palm_policy_vs_teacher']['median_m'] * 100:.2f}cm, 最近傍リプレイ {ph['palm_replay_vs_teacher']['median_m'] * 100:.2f}cm, "
          f"学習 {len(train_palms)} 本中の最良（正解を知って選んだ上限） "
          f"{ph['best_train_vs_teacher']['median_m'] * 100:.2f}cm, 最近傍の学習 {ph['palm_teacher_vs_nn']['median_m'] * 100:.2f}cm。"
          f"方策がその最良より教師に近い: {summary['beats_best_train_hold']}/{len(rows)}"
          f"（成功のみ {summary['beats_best_train_hold_success_only']}/{summary['policy_success']}）")

    plot_traj(rows, palms, run.cond, out_dir, args.examples)
    print(f"結果を保存: {out}")


def pick_examples(rows: list[dict], n_ok: int = 2, n_ng: int = 2) -> list[int]:
    """方策の成功のうち、最近傍リプレイが失敗したものを優先して学習データに近い順に n_ok 個と、
    方策の失敗のうち方策–教師のずれが大きいもの n_ng 個。"""
    oks = sorted((r for r in rows if r["policy_success"]), key=lambda r: (r["replay_success"], r["train_nn"]["dist_norm"]))
    ngs = sorted((r for r in rows if not r["policy_success"]), key=lambda r: -r["palm_policy_vs_teacher"]["mean_m"])
    # oks の先頭はリプレイが失敗したもの（その中で学習データから近い順）
    picks = [r["seed"] for r in oks[:n_ok]]
    return picks + [r["seed"] for r in ngs[:n_ng]]


def plot_traj(rows, palms, cond, out_dir: Path, examples) -> None:
    by_seed = {r["seed"]: r for r in rows}
    missing = [s for s in (examples or []) if s not in by_seed]
    if missing:
        sys.exit(f"--examples のシード {missing} は結果に無い（{rows[0]['seed']}〜{rows[-1]['seed']}）")
    examples = examples or pick_examples(rows)
    fig, axes = plt.subplots(2, len(examples), figsize=(3.8 * len(examples), 7.2), squeeze=False,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    t = np.arange(config.PICK_EPISODE_STEPS) / config.CONTROL_HZ
    for j, s in enumerate(examples):
        r, p = by_seed[s], palms[s]
        ax = axes[0, j]
        status = "成功" if r["policy_success"] else f"失敗（{ev.PICK_FAILURES[r['policy_failure']]}）"
        rstat = "成功" if r["replay_success"] else f"失敗（{ev.PICK_FAILURES[r['replay_failure']]}）"
        ax.set_title(f"seed {s}\n方策 {status}\nリプレイ {rstat}\n（学習 ep {r['train_nn']['train_episode']}、"
                     f"正規化距離 {r['train_nn']['dist_norm']:.2f}）", loc="left", fontsize=8.5)
        ax.invert_xaxis()
        ax.set_aspect("equal")
        for key, c, ls, label in (("replay", C_NN, "--", "最近傍リプレイ"), ("teacher", C_TEACHER, "-", "教師（同じ配置）"),
                                  ("policy", C_POLICY, "-", "方策（推論）")):
            ax.plot(p[key][:, 1], p[key][:, 0], c=c, ls=ls, lw=1.6 if key == "policy" else 1.3, label=label)
        cube0, target = p["cube_teacher"][0], p["target"]
        ax.scatter(cube0[1], cube0[0], marker="s", s=60, c="#e34948", ec="white", zorder=5, label="キューブの初期位置")
        ax.scatter(p["cube_policy"][-1, 1], p["cube_policy"][-1, 0], marker="s", s=40, facecolors="none",
                   ec=C_POLICY, lw=1.4, zorder=5, label="キューブの最終位置（方策）")
        ax.scatter(target[1], target[0], marker="o", s=90, c="#1baf7a", ec="white", zorder=4, label="置き場所（目標）")
        ax.set_xlim(0.08, -0.37)
        ax.set_ylim(0.13, 0.40)
        ax.set_xlabel("y [m]")
        if j == 0:
            ax.set_ylabel("x [m]（上から見た手のひらの軌跡）")
        ax = axes[1, j]
        for key, c, ls in (("replay", C_NN, "--"), ("teacher", C_TEACHER, "-"), ("policy", C_POLICY, "-")):
            ax.plot(t, p[key][:, 2], c=c, ls=ls, lw=1.6 if key == "policy" else 1.3)
        ax.axvline(T_GRASP / config.CONTROL_HZ, c="#8a8984", lw=0.8, ls=":")
        ax.annotate("把持の区間の終わり", (T_GRASP / config.CONTROL_HZ, ax.get_ylim()[0]), xytext=(3, 4),
                    textcoords="offset points", fontsize=7, color="#52514e")
        ax.set_xlabel("時間 [s]")
        if j == 0:
            ax.set_ylabel("手のひらの高さ z [m]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"手のひらの軌跡: 方策（{cond}）・教師・最近傍リプレイ", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_dir / f"traj_examples_{cond}.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.set_title(f"(a) 把持の区間（{T_APPROACH_END / config.CONTROL_HZ:.1f}〜{T_GRASP / config.CONTROL_HZ:.1f} 秒）の"
                 "手のひらの、教師とのずれ", loc="left")
    x = np.array([r["palm_policy_vs_teacher"]["hold_mean_m"] for r in rows]) * 100
    y = np.array([r["palm_replay_vs_teacher"]["hold_mean_m"] for r in rows]) * 100
    ok = np.array([r["policy_success"] for r in rows])
    lim = max(x.max(), y.max()) * 1.08
    ax.plot([0, lim], [0, lim], c="#8a8984", lw=0.8, ls="--")
    above = int((x < y).sum())
    ax.annotate(f"線より上（{above}/{len(rows)}）= 方策の方が\n最近傍リプレイより教師（正解）に近い",
                (lim * 0.03, lim * 1.17), fontsize=7.5, color="#52514e", va="top")
    ax.scatter(x[ok], y[ok], s=30, c=C_OK, ec="white", lw=0.6, label=f"方策 成功（{ok.sum()}）", zorder=3)
    ax.scatter(x[~ok], y[~ok], s=42, c=C_NG, marker="X", ec="white", lw=0.5, label=f"方策 失敗（{(~ok).sum()}）", zorder=4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim * 1.2)  # 上の 2 割は注記の場所
    ax.set_xlabel("方策–教師（同じ配置の正解の軌跡）[cm]")
    ax.set_ylabel("最近傍リプレイ–教師 [cm]")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    ax.set_title("(b) 教師（正解）との手のひらのずれの時間変化（全シードの中央値、帯は 25〜75%）", loc="left")
    t = np.arange(config.PICK_EPISODE_STEPS) / config.CONTROL_HZ
    for key, c, ls, label in (("d_rt", C_NN, "-", "最近傍リプレイ–教師"), ("d_pt", C_POLICY, "-", "方策–教師")):
        v = np.stack([palms[r["seed"]][key] for r in rows]) * 100
        med = np.median(v, axis=0)
        ax.plot(t, med, c=c, ls=ls, lw=1.6, label=label)
        if ls == "-":
            ax.fill_between(t, *np.percentile(v, [25, 75], axis=0), color=c, alpha=0.15, lw=0)
    ax.axvspan(T_APPROACH_END / config.CONTROL_HZ, T_GRASP / config.CONTROL_HZ, color="#e4e3df", alpha=0.6, lw=0, zorder=0)
    ax.annotate("把持の区間\n（指を閉じる）", (T_APPROACH_END / config.CONTROL_HZ, ax.get_ylim()[1]), xytext=(3, -4),
                textcoords="offset points", fontsize=7, color="#52514e", va="top")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("手のひらの位置の差 [cm]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"traj_summary_{cond}.png")
    plt.close(fig)
    print(f"図を保存: {out_dir / f'traj_examples_{cond}.png'}, {out_dir / f'traj_summary_{cond}.png'}")


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def frange(lo: float, hi: float, step: float) -> np.ndarray:
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def cmd_grid(args, policy_path: Path) -> None:
    if args.replot:  # 保存した結果（JSON）から図だけ描き直す
        res = json.loads(args.replot.read_text())
        plot_grid(res["summary"], res["cells"], pa.TrainingNeighbors(args.dataset_root).env0, args.replot.with_suffix(".png"))
        return
    run = Runner(args, policy_path)
    nb = pa.TrainingNeighbors(args.dataset_root)
    xs, ys = frange(*args.x), frange(*args.y)
    out_dir = args.out_dir or policy_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run.cond}_yaw{args.yaw:+.0f}_target{args.target[0]:+.2f}{args.target[1]:+.2f}"
    print(f"格子 {len(xs)}×{len(ys)} = {len(xs) * len(ys)} 点, yaw {args.yaw}°, 目標 {args.target}, シード {args.seed}"
          f"{'、教師も動かす' if args.with_scripted else ''}")
    cells = []
    t0 = time.time()
    for x in xs:
        for y in ys:
            place = {"cube": (float(x), float(y), args.yaw), "target": tuple(args.target)}
            obs0 = run.env.reset(args.seed, **place)
            nn = nb.nearest(obs0[pa.ENV_KEY])
            pol = run.policy_rollout(args.seed, place)
            cell = {"x": float(x), "y": float(y), "policy_success": pol["success"], "policy_failure": pol["failure"],
                    "policy_place_error_m": pol["place_error_m"], "policy_max_lift_m": pol["max_lift_m"],
                    "dist_norm": nn["dist_norm"], "min_cube_xy_m": nn["min_cube_xy_m"],
                    "cube_target_dist_m": float(np.hypot(x - args.target[0], y - args.target[1]))}
            if args.with_replay:
                rep = run.replay_rollout(args.seed, place, nb.episodes["action"][nn["train_row"]])
                cell.update(replay_success=rep["success"], replay_failure=rep["failure"], replay_train_episode=nn["train_episode"])
            if args.with_scripted:
                tea = run.teacher_rollout(args.seed, place)
                cell.update(teacher_success=tea["success"], teacher_failure=tea["failure"],
                            teacher_feasible=tea["teacher_feasible"])
            cells.append(cell)
        print(f"  x={x:.2f} まで {len(cells)} 点, 方策 成功 {sum(c['policy_success'] for c in cells)}"
              + (f", 教師 成功 {sum(c['teacher_success'] for c in cells)}" if args.with_scripted else "")
              + (f", 最近傍リプレイ 成功 {sum(c['replay_success'] for c in cells)}" if args.with_replay else "")
              + f"（{time.time() - t0:.0f}s）")

    def in_range(c):
        return (config.PICK_CUBE_X[0] <= c["x"] <= config.PICK_CUBE_X[1]
                and config.PICK_CUBE_Y[0] <= c["y"] <= config.PICK_CUBE_Y[1])

    def rate(sel, key="policy_success"):
        v = [c[key] for c in cells if sel(c)]
        return {"success": int(sum(v)), "n": len(v)}

    near = lambda c: c["cube_target_dist_m"] < config.PICK_MIN_DIST  # noqa: E731  学習には無い近さ
    summary = {"condition": run.cond, "yaw_deg": args.yaw, "target": args.target, "seed": args.seed,
               "x": args.x, "y": args.y, "policy_in_range": rate(in_range), "policy_out_of_range": rate(lambda c: not in_range(c)),
               # キューブ xy は学習時の範囲の内だが、置き場所から PICK_MIN_DIST 未満（学習には無い配置）
               "policy_in_range_near_target": rate(lambda c: in_range(c) and near(c))}
    if args.with_scripted:
        summary["teacher_in_range"] = rate(in_range, "teacher_success")
        summary["teacher_out_of_range"] = rate(lambda c: not in_range(c), "teacher_success")
        summary["policy_given_teacher_success"] = rate(lambda c: c["teacher_success"])
    if args.with_replay:
        summary["replay_in_range"] = rate(in_range, "replay_success")
        summary["replay_out_of_range"] = rate(lambda c: not in_range(c), "replay_success")
    summary["wall_time_s"] = time.time() - t0
    out = out_dir / f"grid_{tag}.json"
    out.write_text(json.dumps({"summary": summary, "cells": cells}, indent=2, ensure_ascii=False))
    print(f"\n学習範囲の内: 方策 {summary['policy_in_range']}, 外: 方策 {summary['policy_out_of_range']}")
    if args.with_scripted:
        print(f"学習範囲の内: 教師 {summary['teacher_in_range']}, 外: 教師 {summary['teacher_out_of_range']}, "
              f"教師が成功した配置での方策 {summary['policy_given_teacher_success']}")
    if args.with_replay:
        print(f"学習範囲の内: 最近傍リプレイ {summary['replay_in_range']}, 外: 最近傍リプレイ {summary['replay_out_of_range']}")

    plot_grid(summary, cells, nb.env0, out_dir / f"grid_{tag}.png")
    print(f"結果を保存: {out}")


def plot_grid(summary: dict, cells: list[dict], tr: np.ndarray, png: Path) -> None:
    """格子の成功マップ。tr は学習データの初期の environment_state（エピソード数, 10）。"""
    target, yaw, seed, cond = summary["target"], summary["yaw_deg"], summary["seed"], summary["condition"]
    xs, ys = np.unique([c["x"] for c in cells]), np.unique([c["y"] for c in cells])
    panels = ([("policy", "方策（推論）")] + ([("replay", "最近傍リプレイ")] if "replay_in_range" in summary else [])
              + ([("teacher", "教師（スクリプト方策）")] if "teacher_in_range" in summary else []))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 7.0), squeeze=False)
    step = min(np.diff(xs).min() if len(xs) > 1 else 0.02, np.diff(ys).min() if len(ys) > 1 else 0.02)
    for ax, (who, title) in zip(axes[0], panels):
        r = summary[f"{who}_in_range"], summary[f"{who}_out_of_range"]
        setup_topview(ax, f"{title}\nキューブの初期位置の範囲の内 {r[0]['success']}/{r[0]['n']}, 外 {r[1]['success']}/{r[1]['n']} 成功",
                      target_range=False)
        for c in cells:
            okc = c[f"{who}_success"]
            ax.add_patch(Rectangle((c["y"] - step / 2 + 0.001, c["x"] - step / 2 + 0.001), step - 0.002, step - 0.002,
                                   fc=C_OK if okc else "white", ec=C_OK if okc else C_NG, lw=1.0, alpha=0.85 if okc else 1))
            if not okc:
                ax.plot(c["y"], c["x"], marker="x", c=C_NG, ms=4, mew=1.2)
        ax.scatter(tr[:, 1], tr[:, 0], s=4, c="#52514e", alpha=0.5, zorder=3, label="学習データのキューブの初期位置")
        ax.scatter(target[1], target[0], marker="o", s=90, c="#1baf7a", ec="white", zorder=4, label="置き場所（目標）")
        ax.add_patch(Circle((target[1], target[0]), config.PICK_MIN_DIST, fill=False, ls=":", lw=0.9, ec="#1baf7a"))
        ax.set_xlim(max(ys.max(), target[1]) + step, min(ys.min(), target[1]) - step)
        ax.set_ylim(min(xs.min(), target[0]) - step, max(xs.max(), target[0]) + step)
        ax.plot([], [], marker="s", ls="", c=C_OK, label="このキューブ位置から成功")
        ax.plot([], [], marker="x", ls="", c=C_NG, label="このキューブ位置から失敗")
        ax.plot([], [], ls=":", c="#1baf7a", label=f"置き場所から {config.PICK_MIN_DIST * 100:.0f}cm（学習ではキューブは必ずこれより遠い）")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"キューブを置いた位置ごとの成功（キューブ yaw {yaw:+.0f}°、置き場所 {tuple(target)}、"
                 f"ロボットの初期姿勢はシード {seed}、{cond}）", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(png)
    plt.close(fig)
    print(f"図を保存: {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", type=Path, default=config.EVAL_POLICY_DIRS["pick"])
    parser.add_argument("--dataset-root", type=Path, default=config.EVAL_DATASET_DIRS["pick"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None, help="出力先（既定: モデルのフォルダの隣の analysis/）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dist", help="学習データと評価の初期配置の分布・近さ")
    p.add_argument("--evals", type=Path, nargs="*", default=None,
                   help="06_eval_policy.py の結果 JSON（既定: モデルのフォルダの隣の eval_pick_*_50ep.json）")
    p.add_argument("--show", default=None, help="図で色分けする評価（JSON のファイル名の stem。既定: 最初の JSON）")

    for name, help_ in (("traj", "方策の成功・軌跡を教師・最近傍リプレイと比べる"), ("grid", "キューブの位置を格子状に振る")):
        p = sub.add_parser(name, help=help_)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--n-action-steps", type=int, default=None)
        g.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF")
        if name == "traj":
            p.add_argument("--seed", type=int, default=config.EVAL_SEED_BASE)
            p.add_argument("--episodes", type=int, default=config.EVAL_EPISODES)
            p.add_argument("--examples", type=int, nargs="*", default=None, help="図に描くシード（既定: 自動で 4 つ）")
            p.add_argument("--replot", type=Path, default=None, help="保存した traj_<条件>.json（と同名の .npz）から図だけ描き直す")
        else:
            p.add_argument("--x", type=float, nargs=3, default=(0.16, 0.44, 0.02), metavar=("LO", "HI", "STEP"))
            p.add_argument("--y", type=float, nargs=3, default=(-0.29, -0.05, 0.02), metavar=("LO", "HI", "STEP"))
            p.add_argument("--yaw", type=float, default=0.0, help="キューブの yaw [deg]")
            p.add_argument("--target", type=float, nargs=2, default=(0.30, 0.0), metavar=("X", "Y"))
            p.add_argument("--seed", type=int, default=config.EVAL_SEED_BASE, help="ロボットの初期姿勢を決めるシード")
            p.add_argument("--with-scripted", action="store_true", help="同じ配置で教師（スクリプト方策）も動かす")
            p.add_argument("--with-replay", action="store_true", help="同じ配置で最近傍リプレイも動かす")
            p.add_argument("--replot", type=Path, default=None, help="保存した grid_<...>.json から図だけ描き直す")
    args = parser.parse_args()
    policy_path = args.policy_path.resolve()
    {"dist": cmd_dist, "traj": cmd_traj, "grid": cmd_grid}[args.cmd](args, policy_path)


if __name__ == "__main__":
    main()
