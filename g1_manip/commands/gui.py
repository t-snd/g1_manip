"""ピック＆プレースの推論を GUI で試す（キューブと置き場所を上面図で置いて、方策・教師・最近傍リプレイを動かす）。

    python -m g1_manip gui
    python -m g1_manip gui --policy-path outputs/<RUN_NAME>/pretrained_model

- 左上: 机の上面図。赤い四角（キューブ）と緑の三角（置き場所）をドラッグで動かす。キューブの向きはマウスホイールか
  スライダーで変える。薄い点・三角は学習データの初期配置（印の大きさはこれに合わせた。実寸ではない）、破線は学習時の範囲
- 左下: 実行するもの（方策の推論条件・教師・最近傍リプレイ）、ロボットの初期姿勢のシード、実時間で再生するか、動画を保存するか、
  その下に実行の状態と結果
- 右: 実行中の映像（斜め上の視点と頭部カメラ）。頭部カメラは画像入力版（フェーズ8）で方策が見る視点
- 方策に渡す観測は env.observation() のうちモデルの入力にあるキー（policy.obs_to_batch。データ生成・学習と同じ形）。
  モデルの入力に頭部カメラの画像があれば（フェーズ8）、環境で 224×224 の画像も描いて渡す。モデルの読み込みは ACT 用（policy.load_policy）で、
  入力の形の照合（eval の preflight）はしていない。入力が合わなければ実行時にエラーをステータス欄に出して止まる
- 実行ごとの結果はモデルのフォルダの隣の gui/runs.jsonl に追記する（動画もそこに保存する）
- 描画は mujoco.Renderer（オフスクリーン）なので mjpython は要らない
"""

import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco
import numpy as np
from PIL import Image, ImageTk

from g1_manip import config
from g1_manip import training_data as td
from g1_manip.env import PickPlaceEnv
from g1_manip.policy import load_policy, make_actor, needs_images, pick_device
from g1_manip.rollout import PICK_FAILURES, pick_failure
from g1_manip.scripted import ScriptedPickPlacePolicy
from g1_manip.video import VideoWriter, overview_camera

# 上面図の表示範囲 [m]（縦が x：上がロボットの前方、横が y：右がロボットの右手側）と縮尺
VIEW_X = (0.10, 0.48)  # 下端（ロボット側）・上端
VIEW_Y = (0.20, -0.36)  # 左端・右端
SCALE = 1100.0  # [px/m]
# キューブと置き場所の印の大きさ [px]。学習データの初期配置の点（半径 2px）・三角と同じくらいにする
# （実寸だとキューブ 1 辺 4.5cm・目標の円盤 直径 8cm で、学習データの点に比べて大きすぎ、配置が読み取りにくい）
CUBE_MARK = 4.5  # 正方形の半辺
TARGET_MARK = 6.0  # 三角の中心から頂点まで
IMG_W, IMG_H = 440, 330  # 右側の映像 1 枚の大きさ（2 枚縦に並べて、M1 MacBook の画面に収まる大きさ）
# 実行するもの: 表示名 → (種類, n_action_steps, temporal_ensemble)
MODES = {
    "方策: 既定（100 ステップごとに予測）": ("policy", None, None),
    "方策: temporal ensembling 0.01": ("policy", None, 0.01),
    "方策: 10 ステップごとに予測": ("policy", 10, None),
    "教師（スクリプト方策）": ("teacher", None, None),
    "最近傍リプレイ（最も近い学習エピソードの行動を再生）": ("replay", None, None),
}
DEFAULT_CUBE = (0.28, -0.12, 0.0)  # x, y [m], yaw [deg]（学習時の範囲の中央付近）
DEFAULT_TARGET = (0.30, 0.0)
C_CUBE, C_TARGET, C_TRAIL, C_RANGE = "#e34948", "#1baf7a", "#eb6834", "#8a8984"
STATUS_FONT = ("Hiragino Sans", 12, "bold")
MAX_STEPS_PER_TICK = 10  # 「実時間で再生」で 1 回の描画の間に進めるステップ数の上限（追いつけないときに窓が固まらないように）


def to_px(x: float, y: float) -> tuple[float, float]:
    return (VIEW_Y[0] - y) * SCALE, (VIEW_X[1] - x) * SCALE


def to_world(u: float, v: float) -> tuple[float, float]:
    return VIEW_X[1] - v / SCALE, VIEW_Y[0] - u / SCALE


def square_px(x: float, y: float, yaw_deg: float, half: float) -> list[float]:
    """位置 (x, y) [m] を中心に、yaw だけ回した半辺 half [px] の正方形の頂点（キャンバスの座標）。"""
    u, v = to_px(x, y)
    a = np.deg2rad(yaw_deg)
    pts = []
    for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        # ワールドの (dx, dy) をキャンバスへ: x が上（-v）、y が左（-u）
        wx, wy = np.cos(a) * dx - np.sin(a) * dy, np.sin(a) * dx + np.cos(a) * dy
        pts += [u - wy * half, v - wx * half]
    return pts


class PickGui:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        # モデルの入力に画像があれば、観測に頭部カメラの画像を入れる環境にする（1 ステップごとに描くので少し遅くなる）
        self.images = needs_images(args.policy_path)
        self.env = PickPlaceEnv(images=self.images)
        self.model, self.data = self.env.model, self.env.data
        self.policy_path = args.policy_path.resolve()
        self.out_dir = self.policy_path.parent / "gui"
        self.device = pick_device(args.device)
        self.policies = {}  # (n_action_steps, temporal_ensemble) → (policy, act)
        self.teacher = None
        # 画像入力のデータセットを渡されたら、元の状態のデータセット（environment_state がある）を使う
        state_root, train_eps = td.state_dataset(args.dataset_root)
        self.nb = td.TrainingNeighbors(state_root, only=train_eps) if (state_root / "data").exists() else None
        self.cube = list(DEFAULT_CUBE)
        self.target = list(DEFAULT_TARGET)
        self.running = False
        self.drag = None
        self.trail = []
        self.after_id = None  # 予約した次の tick（停止したら取り消す）
        self.run_id = 0  # 実行ごとに増やす。古い実行の tick が残っていても、番号が違えば何もしない

        self.r_view = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self.r_head = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self.cam = overview_camera(self.model)

        self._build_widgets()
        self.reset_scene()

    # --- 画面 ---
    def _build_widgets(self) -> None:
        root = self.root
        root.title(f"G1 ピック＆プレース 推論テスト（{'画像' if self.images else '状態'}入力のモデル: {self.policy_path.parent.name}）")
        left = ttk.Frame(root, padding=6)
        left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(root, padding=6)
        right.grid(row=0, column=1, sticky="n")

        w = int((VIEW_Y[0] - VIEW_Y[1]) * SCALE)
        h = int((VIEW_X[1] - VIEW_X[0]) * SCALE)
        ttk.Label(left, text="上面図: 赤 ■ = キューブの初期位置（ドラッグで移動、ホイールで向き）、緑 ▲ = 置き場所（ドラッグ）\n"
                             "実行中・実行後: 赤い枠 = いまのキューブ、オレンジの線 = 手のひらの軌跡").pack(anchor="w")
        self.canvas = tk.Canvas(left, width=w, height=h, bg="white", highlightthickness=1, highlightbackground="#c3c2b7")
        self.canvas.pack(anchor="w")  # 操作欄の方が少し幅をとるので、ラベル・結果の文と左端をそろえる
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="キューブの向き yaw [deg]").pack(side="left")
        self.yaw_var = tk.DoubleVar(value=self.cube[2])
        self.yaw_scale = ttk.Scale(row, from_=-45, to=45, variable=self.yaw_var, command=self.on_yaw)
        self.yaw_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.show_train = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="学習データを表示", variable=self.show_train, command=self.draw_topview).pack(side="left")
        self.info = tk.StringVar()
        ttk.Label(left, textvariable=self.info, justify="left", wraplength=w).pack(anchor="w", pady=(4, 0))

        ttk.Label(right, text="斜め上の視点").pack(anchor="w")
        self.img_view = ttk.Label(right)
        self.img_view.pack()
        ttk.Label(right, text="頭部カメラ（画像入力版の方策が見る視点）").pack(anchor="w", pady=(6, 0))
        self.img_head = ttk.Label(right)
        self.img_head.pack()

        # 操作と結果は上面図の下に置く（窓の下端に並べると、右の映像 2 枚より下になって画面からはみ出す）
        ttk.Separator(left).pack(fill="x", pady=6)
        ctrl = ttk.Frame(left)
        ctrl.pack(fill="x")
        row = ttk.Frame(ctrl)
        row.pack(fill="x")
        ttk.Label(row, text="実行するもの").pack(side="left")
        self.mode = tk.StringVar(value=next(iter(MODES)))
        ttk.Combobox(row, textvariable=self.mode, values=list(MODES), state="readonly", width=46).pack(side="left", padx=4)
        self.btn_run = ttk.Button(row, text="実行", command=self.run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(row, text="停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        row = ttk.Frame(ctrl)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="初期姿勢のシード").pack(side="left")
        self.seed = tk.IntVar(value=config.EVAL_SEED_BASE)
        spin = ttk.Spinbox(row, from_=0, to=10**9, textvariable=self.seed, width=9, command=self.reset_scene)
        spin.pack(side="left", padx=4)
        spin.bind("<Return>", self.reset_scene)
        spin.bind("<FocusOut>", self.reset_scene)
        self.realtime = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="実時間で再生", variable=self.realtime).pack(side="left", padx=6)
        self.save_video = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="動画を保存", variable=self.save_video).pack(side="left", padx=6)
        ttk.Button(row, text="配置を最初に戻す", command=self.reset_placement).pack(side="right")
        # 結果の文は上面図の幅で折り返す（窓が画面の高さに収まるよう、動画はファイル名だけ出す。フルパスは runs.jsonl にある）
        self.status = tk.StringVar()
        ttk.Label(left, textvariable=self.status, font=STATUS_FONT, justify="left", wraplength=w).pack(anchor="w", pady=(6, 0))

    def draw_topview(self) -> None:
        c = self.canvas
        c.delete("all")
        # 机（x 0.15〜0.65, y ±0.40）。ロボットは x = 0（表示範囲の下の外）にいる
        x0, y0 = to_px(0.15, 0.40)
        x1, y1 = to_px(0.65, -0.40)
        c.create_rectangle(x0, y0, x1, y1, fill="#efe4d6", outline="#c9b79f")
        cw, ch = int(c["width"]), int(c["height"])
        c.create_rectangle(0, ch - 22, cw, ch, fill="#dedcd5", outline="")
        c.create_text(cw / 2, ch - 11, text="↓ ロボット（手前）　　右がロボットの右手側 →", fill="#52514e")
        for (xa, xb), (ya, yb), label in ((config.PICK_CUBE_X, config.PICK_CUBE_Y, "キューブの範囲（学習時）"),
                                          (config.PICK_TARGET_X, config.PICK_TARGET_Y, "置き場所の範囲（学習時）")):
            pa_, pb = to_px(xb, ya), to_px(xa, yb)
            c.create_rectangle(*pa_, *pb, outline=C_RANGE, dash=(4, 3))
            # 枠の右上の角に右端を合わせ、枠の上に左へ伸ばす（右へ伸ばすと隣の枠に重なる）
            c.create_text(pa_[0] - 2, pa_[1] - 8, text=label, fill=C_RANGE, anchor="e", font=("Hiragino Sans", 9))
        if self.show_train.get() and self.nb is not None:
            for e in self.nb.env0:
                u, v = to_px(e[0], e[1])
                c.create_oval(u - 2, v - 2, u + 2, v + 2, fill="#c3c2b7", outline="")
                u, v = to_px(e[7], e[8])
                c.create_polygon(u, v - 3, u - 3, v + 2, u + 3, v + 2, fill="#d6d5cf", outline="")
        # 置き場所から PICK_MIN_DIST の円（学習ではキューブは必ずこれより遠い）
        u, v = to_px(*self.target)
        r = config.PICK_MIN_DIST * SCALE
        c.create_oval(u - r, v - r, u + r, v + r, outline=C_TARGET, dash=(2, 3))
        # 置き場所（学習データの置き場所と同じ三角）とキューブ（向きつきの正方形）。学習データの点より少し大きいだけ
        r = TARGET_MARK
        c.create_polygon(u, v - r, u - 0.87 * r, v + 0.5 * r, u + 0.87 * r, v + 0.5 * r, fill=C_TARGET, outline="white",
                         tags="target")
        c.create_polygon(*square_px(*self.cube, CUBE_MARK), fill=C_CUBE, outline="white", tags="cube")
        self.draw_dynamic()

    def draw_dynamic(self) -> None:
        """実行中・実行後の手のひらの軌跡と、キューブの現在位置だけを描き直す（実行中は毎回これだけ。
        上面図を全部描き直すと学習データの点 600 個も作り直すことになり、実時間で再生できない）。"""
        c = self.canvas
        c.delete("dynamic")
        if len(self.trail) > 1:
            c.create_line(*[p for xy in self.trail for p in to_px(*xy)], fill=C_TRAIL, width=2, tags="dynamic")
        if self.running or self.trail:
            yaw = float(td.cube_yaw_deg(np.r_[np.zeros(3), self.env.cube_quat()]))
            c.create_polygon(*square_px(*self.env.cube_pos()[:2], yaw, CUBE_MARK + 2), fill="", outline=C_CUBE,
                             width=2, tags="dynamic")

    def render_images(self) -> None:
        self.r_view.update_scene(self.data, camera=self.cam)
        self.r_head.update_scene(self.data, camera=config.HEAD_CAMERA_NAME)
        self._photo_view = ImageTk.PhotoImage(Image.fromarray(self.r_view.render()))
        self._photo_head = ImageTk.PhotoImage(Image.fromarray(self.r_head.render()))
        self.img_view.configure(image=self._photo_view)
        self.img_head.configure(image=self._photo_head)

    def update_info(self) -> None:
        x, y, yaw = self.cube
        lines = [f"キューブ x {x:.3f}  y {y:+.3f}  yaw {yaw:+.0f}°    置き場所 x {self.target[0]:.3f}  y {self.target[1]:+.3f}"]
        warn = []
        if not (config.PICK_CUBE_X[0] <= x <= config.PICK_CUBE_X[1] and config.PICK_CUBE_Y[0] <= y <= config.PICK_CUBE_Y[1]):
            warn.append("キューブが学習時の範囲の外")
        if abs(yaw) > config.PICK_CUBE_YAW_DEG:
            warn.append(f"向きが学習時の範囲（±{config.PICK_CUBE_YAW_DEG:g}°）の外")
        if not (config.PICK_TARGET_X[0] <= self.target[0] <= config.PICK_TARGET_X[1]
                and config.PICK_TARGET_Y[0] <= self.target[1] <= config.PICK_TARGET_Y[1]):
            warn.append("置き場所が学習時の範囲の外")
        d = np.hypot(x - self.target[0], y - self.target[1])
        if d < config.PICK_MIN_DIST:
            warn.append(f"キューブと置き場所が {d * 100:.1f}cm（学習では {config.PICK_MIN_DIST * 100:.0f}cm 以上）")
        if self.nb is not None:
            nn = self.nb.nearest(self.env.env_state())
            ref = self.nb.self_nn()
            lines.append(f"学習データとの近さ: 最も近いエピソード {nn['train_episode']}、正規化距離 {nn['dist_norm']:.2f}"
                         f"（学習エピソードどうしは中央値 {np.median(ref):.2f}・最大 {ref.max():.2f}）、"
                         f"キューブ xy だけなら {nn['min_cube_xy_m'] * 100:.1f}cm")
        if warn:
            lines.append("⚠️ " + "、".join(warn))
        self.info.set("\n".join(lines))

    # --- 操作 ---
    def on_press(self, e) -> None:
        if self.running:
            return
        x, y = to_world(e.x, e.y)
        d_cube = np.hypot(x - self.cube[0], y - self.cube[1])
        d_target = np.hypot(x - self.target[0], y - self.target[1])
        if min(d_cube, d_target) > 0.05:
            self.drag = None
        else:
            self.drag = "cube" if d_cube <= d_target else "target"

    def on_drag(self, e) -> None:
        if self.drag is None or self.running:
            return
        x, y = to_world(e.x, e.y)
        # 机の上で、上面図に見えている範囲（見えない所へ置くとクリックできず戻せなくなる）
        margin = 0.03
        x = float(np.clip(x, max(0.15, VIEW_X[0] + margin), VIEW_X[1] - margin))
        y = float(np.clip(y, VIEW_Y[1] + margin, VIEW_Y[0] - margin))
        if self.drag == "cube":
            self.cube[:2] = [round(x, 3), round(y, 3)]
        else:
            self.target = [round(x, 3), round(y, 3)]
        self.trail = []
        self.draw_topview()

    def on_release(self, e) -> None:
        if self.drag is not None:
            self.drag = None
            self.reset_scene()

    def on_wheel(self, e) -> None:
        if self.running:
            return
        self.yaw_var.set(float(np.clip(self.yaw_var.get() + (5 if e.delta > 0 else -5), -45, 45)))
        self.on_yaw()

    def on_yaw(self, *_) -> None:
        if self.running:
            self.yaw_var.set(self.cube[2])  # 実行中は向きを変えない（スライダーの表示も戻す）
            return
        self.cube[2] = round(float(self.yaw_var.get()))
        self.reset_scene()

    def reset_placement(self) -> None:
        """キューブと置き場所を起動時の配置に戻す。"""
        if self.running:
            return
        self.cube, self.target = list(DEFAULT_CUBE), list(DEFAULT_TARGET)
        self.yaw_var.set(self.cube[2])
        self.reset_scene()

    def _seed(self) -> int | None:
        try:
            return int(self.seed.get())
        except (tk.TclError, ValueError):
            self.status.set("シードは整数で入れる")
            return None

    def reset_scene(self, *_) -> None:
        """指定の配置・シードでリセットし、映像と上面図を描き直す（実行はしない）。"""
        if self.running or (seed := self._seed()) is None:
            return
        self.env.reset(seed, cube=tuple(self.cube), target=tuple(self.target))
        self.trail = []
        self.draw_topview()
        self.render_images()
        self.update_info()
        self.status.set("")

    def _policy(self, n_action_steps, te):
        """(policy, act)。推論の条件ごとに一度だけ読み込む。"""
        key = (n_action_steps, te)
        if key not in self.policies:
            self.status.set("モデルを読み込み中…")
            self.root.update_idletasks()
            policy, pre, post, _ = load_policy(self.policy_path, self.device, n_action_steps, te)
            self.policies[key] = (policy, make_actor(policy, pre, post))
        return self.policies[key]

    def run(self) -> None:
        if self.running:
            return
        if (seed := self._seed()) is None:
            return
        self.run_mode = self.mode.get()  # 実行中にコンボボックスを変えても、表示・記録は実行したものにする
        kind, n_action_steps, te = MODES[self.run_mode]
        place = {"cube": tuple(self.cube), "target": tuple(self.target)}
        obs = self.env.reset(seed, **place)
        nn = self.nb.nearest(obs[td.ENV_KEY]) if self.nb is not None else None
        if kind == "policy":
            try:
                policy, act = self._policy(n_action_steps, te)
            except ValueError as e:
                self.status.set(str(e))
                return
            policy.reset()
        elif kind == "teacher":
            if self.teacher is None:
                self.teacher = ScriptedPickPlacePolicy(self.env)
            self.status.set("教師が経路を計画中…")
            self.root.update_idletasks()
            self.teacher.reset()

            def act(o):
                return self.teacher.act()
        else:
            if nn is None:
                self.status.set("学習データが無いので最近傍リプレイは使えない")
                return
            actions = iter(self.nb.episodes["action"][nn["train_row"]])

            def act(o):
                return next(actions)
        self.act, self.obs, self.t = act, obs, 0
        self.run_info = {"mode": self.run_mode, "seed": seed, **{k: list(v) for k, v in place.items()},
                         "train_nn": nn, "policy_path": str(self.policy_path)}
        self.cube0_z = self.env.cube_pos()[2]
        self.max_lift = 0.0
        self.trail = [tuple(self.env.palm_pos()[:2])]
        self.video = None
        if self.save_video.get():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.video_path = self.out_dir / time.strftime("run_%Y%m%d_%H%M%S.mp4")
            self.video = VideoWriter(self.video_path, self.model)
        self.running = True
        self.run_id += 1
        self.t_start = time.perf_counter()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.tick(self.run_id)

    def tick(self, run_id: int) -> None:
        if not self.running or run_id != self.run_id:
            return
        # 実時間: 経過した実時間ぶんだけ進める（映像の描画が 1 枚約 50ms かかるので、描画は 1 秒に 10 回弱になるが、
        # ロボットの動きの速さは実際と同じ）。実時間でないとき: 描画 1 回ごとに 10 ステップ（できるだけ速く）
        if self.realtime.get():
            n = max(1, int((time.perf_counter() - self.t_start) * config.CONTROL_HZ) - self.t)
            # 1 ステップが制御周期より遅いとき（画像入力のモデルは観測の画像を描くので 1 ステップ約 60ms）は追いつけず、
            # n が tick ごとに増えて 1 回の tick で窓が固まる。上限を付け、できるだけ速く進める（実時間より遅くなる）
            n = min(n, MAX_STEPS_PER_TICK)
        else:
            n = 10
        try:
            for _ in range(n):
                if self.t >= config.PICK_EPISODE_STEPS:
                    break
                self.obs = self.env.step(self.act(self.obs))
                self.t += 1
                self.max_lift = max(self.max_lift, float(self.env.cube_pos()[2] - self.cube0_z))
                if self.video is not None:
                    self.video.add(self.data)
        except Exception as e:  # 方策の入力が合わないときなど。実行中のまま固まらないよう止める
            self.finish(error=f"{type(e).__name__}: {e}")
            return
        self.trail.append(tuple(self.env.palm_pos()[:2]))
        self.render_images()
        self.draw_dynamic()
        self.status.set(f"実行中 {self.t / config.CONTROL_HZ:.1f} / {config.PICK_EPISODE_STEPS / config.CONTROL_HZ:.0f} 秒")
        if self.t >= config.PICK_EPISODE_STEPS:
            self.finish()
            return
        self.after_id = self.root.after(1, self.tick, run_id)

    def finish(self, stopped: bool = False, error: str | None = None) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if self.video is not None:
            self.video.close()
        done = not stopped and error is None
        failure = pick_failure(self.env, self.max_lift) if done else None
        result = {**self.run_info, "steps": self.t, "stopped": stopped, "error": error,
                  "success": done and failure is None, "failure": failure, "place_error_m": self.env.place_error(), "max_lift_m": self.max_lift,
                  "video": str(self.video_path) if self.video is not None else None}
        if error is not None:
            # 結果の欄は折り返すので、長いと窓が画面の下にはみ出す。全文は runs.jsonl の error にある
            short = error if len(error) <= 120 else error[:120] + "…（全文は gui/runs.jsonl）"
            self.status.set(f"エラーで停止（{self.t / config.CONTROL_HZ:.1f} 秒）: {short}")
        elif stopped:
            self.status.set(f"停止（{self.t / config.CONTROL_HZ:.1f} 秒）")
        else:
            verdict = "成功" if failure is None else f"失敗（{PICK_FAILURES[failure]}）"
            self.status.set(f"{verdict}   置き誤差 {result['place_error_m'] * 100:.2f}cm   持ち上げ {self.max_lift * 100:.1f}cm"
                            f"   ［{self.run_mode}、シード {result['seed']}］"
                            + (f"   動画: gui/{self.video_path.name}" if self.video is not None else ""))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "runs.jsonl", "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        self.last_result = result
        self.draw_topview()

    def stop(self) -> None:
        if self.running:
            self.finish(stopped=True)


def snapshot(app: PickGui, path: Path) -> None:
    """上面図（キャンバスの図形を Pillow で描き直したもの）・2 つの映像・情報欄の文字を 1 枚にして保存する（動作確認用）。
    macOS の screencapture は画面収録の権限が無いと使えないため。ボタンなどの部品は写らない。"""
    from PIL import ImageDraw, ImageFont

    c = app.canvas
    w, h = int(c["width"]), int(c["height"])
    font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 12)
    top = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(top)
    for item in c.find_all():
        kind, xy = c.type(item), c.coords(item)
        opt = lambda k: c.itemcget(item, k) or None  # noqa: E731
        if kind == "rectangle":
            d.rectangle(xy, fill=opt("fill"), outline=opt("outline"), width=int(float(opt("width") or 1)))
        elif kind == "oval":
            d.ellipse(xy, fill=opt("fill"), outline=opt("outline"), width=int(float(opt("width") or 1)))
        elif kind == "polygon":
            d.polygon(xy, fill=opt("fill"), outline=opt("outline"))
        elif kind == "line":
            d.line(xy, fill=opt("fill"), width=int(float(opt("width") or 1)))
        elif kind == "text":
            d.text(xy, c.itemcget(item, "text"), fill=opt("fill"), font=font, anchor="lm" if opt("anchor") == "w" else "mm")
    app.r_view.update_scene(app.data, camera=app.cam)
    view = Image.fromarray(app.r_view.render())
    app.r_head.update_scene(app.data, camera=config.HEAD_CAMERA_NAME)
    head = Image.fromarray(app.r_head.render())
    text = app.info.get() + "\n" + app.status.get()
    out = Image.new("RGB", (w + IMG_W + 12, max(h, 2 * IMG_H + 6) + 70), "white")
    out.paste(top, (0, 0))
    out.paste(view, (w + 12, 0))
    out.paste(head, (w + 12, IMG_H + 6))
    ImageDraw.Draw(out).multiline_text((4, max(h, 2 * IMG_H + 6) + 6), text, fill="black", font=font)
    out.save(path)


def selftest(app: PickGui, root: tk.Tk, shot: Path | None) -> None:
    """ドラッグ操作をイベントで送ってキューブ・置き場所を置き、既定の方策で 1 エピソード実行する（動作確認用）。"""
    def drag(src, dst):
        (u0, v0), (u1, v1) = to_px(*src), to_px(*dst)
        app.canvas.event_generate("<ButtonPress-1>", x=int(u0), y=int(v0))
        for s in np.linspace(0, 1, 6)[1:]:
            app.canvas.event_generate("<B1-Motion>", x=int(round(u0 + s * (u1 - u0))), y=int(round(v0 + s * (v1 - v0))))
        app.canvas.event_generate("<ButtonRelease-1>", x=int(u1), y=int(v1))

    def step1():
        drag(app.cube[:2], (0.30, -0.15))
        drag(app.target, (0.28, 0.0))
        app.yaw_var.set(10)
        app.on_yaw()
        app.realtime.set(False)
        print(f"selftest: 配置 cube {app.cube}, target {app.target}")
        root.update_idletasks()
        print(f"selftest: 窓の大きさ {root.winfo_reqwidth()}x{root.winfo_reqheight()}、"
              f"画面 {root.winfo_screenwidth()}x{root.winfo_screenheight()}")
        app.run()
        root.after(200, wait)

    def wait():
        if app.running:
            root.after(200, wait)
            return
        r = app.last_result
        root.update_idletasks()
        print(f"selftest: 結果表示後の窓の大きさ {root.winfo_reqwidth()}x{root.winfo_reqheight()}")
        print(f"selftest: 結果 success={r['success']} failure={r['failure']} place_error={r['place_error_m'] * 100:.2f}cm "
              f"max_lift={r['max_lift_m'] * 100:.1f}cm steps={r['steps']}")
        if shot is not None:
            snapshot(app, shot)
            print(f"selftest: 画面の写し {shot}")
        root.destroy()

    root.after(500, step1)


def add_arguments(parser) -> None:
    parser.add_argument("--policy-path", type=Path, default=config.EVAL_POLICY_DIRS["pick"])
    parser.add_argument("--dataset-root", type=Path, default=config.EVAL_DATASET_DIRS["pick"],
                        help="学習に使ったデータセット（学習データの表示・近さ・最近傍リプレイに使う。無ければ使わない）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selftest", action="store_true", help="操作をイベントで送って 1 エピソード実行し、終了する（動作確認用）")
    parser.add_argument("--screenshot", type=Path, default=None, help="--selftest の最後に画面の写し（上面図・映像・情報欄）を保存する")


def run(args) -> int:
    if not (args.policy_path / "model.safetensors").exists():
        sys.exit(f"{args.policy_path} にモデルが無い")
    root = tk.Tk()
    app = PickGui(root, args)
    if args.selftest:
        selftest(app, root, args.screenshot)
    root.mainloop()
    return 0
