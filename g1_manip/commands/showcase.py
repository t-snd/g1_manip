"""紹介動画（HyperFrames、video/）の素材を作る: エピソードのクリップ（斜め上の視点・頭部カメラ）と、画面に出す数値の JSON。

    python -m g1_manip showcase                 # video/assets/ に書き出す（M1 で約 10 分）
    cd video && npx hyperframes render -o ../outputs/showcase.mp4

- クリップ: EPISODES を実際に動かした結果をそのまま録る（等倍、50 fps）。エピソードごとに
  <name>_main.mp4（斜め上の視点、MAIN_W×MAIN_H）と <name>_head.mp4（方策に渡す 224×224 の画像を HEAD_SIZE に拡大）。
  最後の HOLD_S 秒は終わった時点の画面で止める
- 静止画: STILLS（タスクの画面に出す、教師デモのフレーム）
- video/assets/data.json: エピソードの条件・配置・結果、学習データの初期配置（散布図用）、成功率（eval の JSON から読む）
- 成功率は、eval で保存した eval_pick_<条件>_50ep.json（学習に使っていない 50 配置）から読む。RATE_FILES が無ければ録画の前に終了する
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from g1_manip import config
from g1_manip.env import PickPlaceEnv
from g1_manip.policy import load_policy, make_actor, pick_device
from g1_manip.rollout import pick_failure
from g1_manip.scripted import ScriptedPickPlacePolicy
from g1_manip.training_data import cube_yaw_deg, load_episodes, state_dataset
from g1_manip.video import overview_camera

MAIN_W, MAIN_H = 1440, 1080
HEAD_SIZE = 448
HOLD_S = 3.0
CRF = "18"
VIDEO_DIR = config.PROJECT_ROOT / "video"
IMG_POLICY = config.OUTPUTS_DIR / "act_g1_pick_img_v1" / "pretrained_model"
IMG_DATASET = config.DATA_DIR / "g1_pick_img_200"
PLOT_PX = (960, 820)  # 散布図（動画では半分の大きさで出す）
PLOT_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
PLOT_COLORS = {"bg": "#14161b", "text": "#eceef2", "sub": "#a0a6b2", "grid": "#4a4f5a", "train": "#5d6370",
               "cube": "#3987e5", "target": "#d95926"}
TE = 0.01  # 評価の例はすべて temporal ensembling（係数 0.01）で動かす
SEED0 = config.EVAL_SEED_BASE

# (名前, 方策, シード, 配置)。方策は "teacher"（スクリプト方策）か "act"（画像入力 ACT、temporal ensembling 0.01）。
# 配置は PickPlaceEnv.reset の cube・target（空ならシードで決まる）
EPISODES = [
    ("teacher", "teacher", 0, {}),  # 学習データのエピソード 0 と同じ配置
    ("ok_far", "act", SEED0 + 0, {}),  # 奥・右の端（x 0.35, y −0.23）
    ("ok_yaw", "act", SEED0 + 25, {}),  # キューブの向き +29°
    ("ok_near", "act", SEED0 + 36, {}),  # 手前・左（x 0.23, y −0.07）
    ("ok_ood", "act", SEED0, {"cube": (0.19, -0.15, 0.0), "target": (0.30, 0.0)}),  # 学習範囲の外（x 0.19）
    ("ng_lift", "act", SEED0 + 19, {}),  # te0.01 の失敗 3 件のうち、持ち上げられなかったもの
    ("ng_place", "act", SEED0 + 15, {}),  # 置き位置のずれ
]

# タスクの画面に出す静止画: エピソードの名前 → {ファイル名: (視点 "main" / "head", ステップ)}
STILLS = {"teacher": {"still_grasp": ("main", 160), "still_head": ("head", 25)}}

# 成功率: (キー, モデルのフォルダ, 条件)。n<k> はチャンクのうち k ステップ実行して予測し直す、te<係数> は temporal ensembling
CKPT10K = config.OUTPUTS_DIR / "act_g1_pick_img_v1_ckpts" / "010000"
RATE_FILES = {
    **{f"n{k}": (IMG_POLICY.parent, f"n{k}") for k in (1, 10, 12, 14, 16, 18, 20, 100)},
    "te0.01": (IMG_POLICY.parent, "te0.01"),
    "state_n100": (config.OUTPUTS_DIR / "act_g1_pick_v1", "n100"),
    "state_te0.01": (config.OUTPUTS_DIR / "act_g1_pick_v1", "te0.01"),
    "ckpt10k_n100": (CKPT10K, "n100"),
    "ckpt10k_te0.01": (CKPT10K, "te0.01"),
}


def add_arguments(parser) -> None:
    parser.add_argument("--out", type=Path, default=VIDEO_DIR / "assets")
    parser.add_argument("--device", default="auto", help="auto（MPS があれば mps、無ければ cpu）/ mps / cpu")
    parser.add_argument("--only", nargs="*", help="録り直すエピソードの名前（省略すると全部）。data.json は全部書き直す")
    parser.add_argument("--plots-only", action="store_true", help="録画せず、data.json から散布図だけ描き直す")


class Clip:
    """フレーム（H, W, 3）uint8 を mp4（H.264, FPS）に書き出す。"""

    def __init__(self, path: Path, w: int, h: int):
        import av

        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=config.FPS, options={"crf": CRF})
        self.stream.width, self.stream.height = w, h
        self.stream.pix_fmt = "yuv420p"
        self._frame_cls = av.VideoFrame

    def add(self, img: np.ndarray, n: int = 1) -> None:
        frame = self._frame_cls.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for _ in range(n):
            self.container.mux(self.stream.encode(frame))

    def close(self) -> None:
        self.container.mux(self.stream.encode())
        self.container.close()


def in_range(cube_xy, yaw: float, target_xy) -> bool:
    """初期配置が学習データの範囲（PICK_CUBE_*・PICK_TARGET_*）の中か。"""
    (cx, cy), (tx, ty) = cube_xy, target_xy
    return bool(config.PICK_CUBE_X[0] <= cx <= config.PICK_CUBE_X[1] and config.PICK_CUBE_Y[0] <= cy <= config.PICK_CUBE_Y[1]
                and abs(yaw) <= config.PICK_CUBE_YAW_DEG
                and config.PICK_TARGET_X[0] <= tx <= config.PICK_TARGET_X[1] and config.PICK_TARGET_Y[0] <= ty <= config.PICK_TARGET_Y[1])


def record(env: PickPlaceEnv, out: Path, name: str, act, seed: int, place: dict) -> dict:
    """1 エピソードを動かして 2 本のクリップに録り、条件と結果を返す。act が None なら教師。"""
    renderer = mujoco.Renderer(env.model, height=MAIN_H, width=MAIN_W)
    camera = overview_camera(env.model)
    camera.distance *= 0.92
    main, head = Clip(out / f"{name}_main.mp4", MAIN_W, MAIN_H), Clip(out / f"{name}_head.mp4", HEAD_SIZE, HEAD_SIZE)
    obs = env.reset(seed, **place)
    cube0, target = env.cube_pos(), env.target.copy()
    yaw0 = float(cube_yaw_deg(env.env_state()))
    if act is None:
        teacher = ScriptedPickPlacePolicy(env)
        teacher.reset()
        act = lambda _obs: teacher.act()  # noqa: E731

    stills = {step: (fname, view) for fname, (view, step) in STILLS.get(name, {}).items()}

    def shoot(n: int = 1, t: int | None = None) -> None:
        renderer.update_scene(env.data, camera=camera)
        frames = {"main": renderer.render(),
                  "head": np.asarray(Image.fromarray(obs[config.IMAGE_KEY]).resize((HEAD_SIZE, HEAD_SIZE), Image.NEAREST))}
        main.add(frames["main"], n)
        head.add(frames["head"], n)
        if t in stills:
            fname, view = stills[t]
            Image.fromarray(frames[view]).save(out / f"{fname}.jpg", quality=92)

    max_lift = 0.0
    try:
        for t in range(config.PICK_EPISODE_STEPS):
            shoot(t=t)
            obs = env.step(act(obs))
            max_lift = max(max_lift, float(env.cube_pos()[2] - cube0[2]))
        shoot(round(HOLD_S * config.FPS))
    finally:
        main.close()
        head.close()
        renderer.close()
    return {"cube": [round(float(v), 3) for v in cube0[:2]], "yaw_deg": round(yaw0, 1),
            "target": [round(float(v), 3) for v in target[:2]],
            "in_train_range": in_range(cube0[:2], yaw0, target[:2]),
            "failure": pick_failure(env, max_lift), "place_error_cm": round(env.place_error() * 100, 1),
            "max_lift_cm": round(max_lift * 100, 1)}


def rates() -> dict:
    out = {}
    for key, (folder, cond) in RATE_FILES.items():
        s = json.loads((folder / f"eval_pick_{cond}_50ep.json").read_text())["summary"]
        out[key] = {"success": round(s["success_rate"] * s["episodes"]), "episodes": s["episodes"], "failures": s["failures"]}
    return out


def training_placements() -> list[dict]:
    """画像入力の学習データ（200 本）の初期配置。environment_state は元の状態のみのデータセットから読む。"""
    root, only = state_dataset(IMG_DATASET)
    ep = load_episodes(root)
    env0 = ep["observation.environment_state"][:, 0]
    keep = np.isin(ep["episode_index"], only) if only is not None else np.ones(len(env0), bool)
    return [{"cube": [round(float(e[0]), 3), round(float(e[1]), 3)], "yaw_deg": round(float(cube_yaw_deg(e)), 1),
             "target": [round(float(e[7]), 3), round(float(e[8]), 3)]} for e in env0[keep]]


def plot_placements(path: Path, train: list[dict], ep: dict | None) -> None:
    """上から見た初期配置の散布図（PLOT_PX、背景は動画の地の色）。学習データのキューブ・置き場所を灰色で、
    ep（このエピソード）があればそのキューブ・置き場所を色付きで重ねる。横軸はロボットから見た左右（右が −y）、縦軸は奥行き x。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    font_manager.fontManager.addfont(PLOT_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=PLOT_FONT).get_name()
    fig = plt.figure(figsize=(PLOT_PX[0] / 100, PLOT_PX[1] / 100), dpi=100, facecolor=PLOT_COLORS["bg"])
    ax = fig.add_axes((0.14, 0.12, 0.83, 0.72), facecolor=PLOT_COLORS["bg"])
    for (x0, x1), (y0, y1) in ((config.PICK_CUBE_X, config.PICK_CUBE_Y), (config.PICK_TARGET_X, config.PICK_TARGET_Y)):
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ls=(0, (4, 3)), lw=1.6, ec=PLOT_COLORS["grid"]))
    cube, tgt = np.array([t["cube"] for t in train]), np.array([t["target"] for t in train])
    ax.scatter(cube[:, 1], cube[:, 0], s=22, c=PLOT_COLORS["train"], lw=0, label="学習データのキューブ（200）")
    ax.scatter(tgt[:, 1], tgt[:, 0], s=22, facecolors="none", edgecolors=PLOT_COLORS["train"], lw=1.2,
               label="学習データの置き場所（200）")
    if ep is not None:
        (cx, cy), (tx, ty) = ep["cube"], ep["target"]
        ax.annotate("", (ty, tx), (cy, cx), arrowprops=dict(arrowstyle="-|>", color=PLOT_COLORS["text"], lw=2,
                                                            shrinkA=14, shrinkB=16))
        ax.scatter([cy], [cx], s=420, marker="s", c=PLOT_COLORS["cube"], ec=PLOT_COLORS["bg"], lw=2.5, zorder=5,
                   label="今回のキューブ")
        ax.scatter([ty], [tx], s=520, facecolors="none", edgecolors=PLOT_COLORS["target"], lw=4, zorder=5,
                   label="今回の置き場所")
    ax.set_xlim(0.08, -0.27)  # 右が −y（ロボットの右手側）
    ax.set_ylim(0.16, 0.41)
    ax.set_aspect("equal")
    ax.set_xlabel("左右 y [m]（右がロボットの右手側）", color=PLOT_COLORS["sub"], fontsize=18)
    ax.set_ylabel("奥行き x [m]", color=PLOT_COLORS["sub"], fontsize=18)
    ax.tick_params(colors=PLOT_COLORS["sub"], labelsize=16)
    for s in ax.spines.values():
        s.set_color(PLOT_COLORS["grid"])
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.995), ncol=2, frameon=False, fontsize=17,
               labelcolor=PLOT_COLORS["text"], handletextpad=0.3, columnspacing=1.0)
    fig.savefig(path, facecolor=PLOT_COLORS["bg"])
    plt.close(fig)


def run(args) -> int:
    needed = [IMG_POLICY / "config.json"] + [f / f"eval_pick_{c}_50ep.json" for f, c in RATE_FILES.values()]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        sys.exit("見つからない（モデルを置き、eval をその条件で回す）:\n  " + "\n  ".join(missing))
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.out / "data.json"
    old = {e["name"]: e for e in json.loads(data_path.read_text())["episodes"]} if data_path.exists() else {}
    if args.plots_only:
        data = json.loads(data_path.read_text())
        write_plots(args.out, data)
        return 0
    env = PickPlaceEnv(images=True)
    g = env.model.vis.global_
    g.offwidth, g.offheight = max(g.offwidth, MAIN_W), max(g.offheight, MAIN_H)
    policy, pre, post, _ = load_policy(IMG_POLICY, pick_device(args.device), None, TE)
    actor = make_actor(policy, pre, post)
    episodes = []
    for name, who, seed, place in EPISODES:
        if args.only and name not in args.only and name in old:
            episodes.append(old[name])
            continue
        policy.reset()
        res = record(env, args.out, name, None if who == "teacher" else actor, seed, place)
        episodes.append({"name": name, "policy": who, "seed": seed, "place_given": bool(place), **res})
        print(f"{name}: {who} seed {seed} {place or ''} → {res['failure'] or '成功'}"
              f"（置き場所との距離 {res['place_error_cm']}cm）", flush=True)
    data = {"fps": config.FPS, "episode_s": config.PICK_EPISODE_STEPS / config.CONTROL_HZ, "hold_s": HOLD_S,
            "te_coeff": TE, "train_range": {"cube_x": config.PICK_CUBE_X, "cube_y": config.PICK_CUBE_Y,
                                            "cube_yaw_deg": config.PICK_CUBE_YAW_DEG,
                                            "target_x": config.PICK_TARGET_X, "target_y": config.PICK_TARGET_Y},
            "episodes": episodes, "rates": rates(), "train": training_placements()}
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    print(f"\n{data_path}（エピソード {len(episodes)}、学習データの配置 {len(data['train'])}）")
    write_plots(args.out, data)
    return 0


def write_plots(out: Path, data: dict) -> None:
    """散布図: placements_train.png（学習データだけ）と、ACT のエピソードごとの placements_<名前>.png。"""
    plot_placements(out / "placements_train.png", data["train"], None)
    for ep in data["episodes"]:
        if ep["policy"] == "act":
            plot_placements(out / f"placements_{ep['name']}.png", data["train"], ep)
    print(f"散布図を {out} に書いた")
