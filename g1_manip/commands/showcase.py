"""紹介動画（約 2 分 20 秒、等倍）を作る: 教師デモ → 画像入力 ACT の推論 → 学習範囲の外の配置 → 失敗例 → 成功率の表。

    python -m g1_manip showcase                          # outputs/showcase.mp4
    python -m g1_manip showcase --out outputs/x.mp4 --device cpu

- 画面（1280×720）: 左に斜め上の視点、右に頭部カメラ（方策に渡す 224×224 の画像を拡大）と、エピソードの条件・配置・経過時間
- 構成と字幕は SEGMENTS。エピソードは実際に動かした結果をそのまま録り、終わりに成功・失敗の内訳を出す。
  学習範囲の外の配置だけは、汎化の確認で試した 8 配置のうち、成功した 4 配置から 3 つを選んでいる（字幕にもそう書く）
- 最後の成功率の表は、eval で保存した eval_pick_<条件>_50ep.json（学習に使っていない 50 配置）から読む。画像入力・座標入力の
  両モデル × n100・te0.01・n10 の 6 個が要る（無ければ録画を始める前に終了する。座標入力のモデル本体は要らない）
- 字幕の日本語フォントは macOS のヒラギノ角ゴシック（FONT_PATHS）
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from g1_manip import config
from g1_manip.env import PickPlaceEnv
from g1_manip.policy import load_policy, make_actor, pick_device
from g1_manip.rollout import PICK_FAILURES, pick_failure
from g1_manip.scripted import ScriptedPickPlacePolicy
from g1_manip.video import overview_camera

W, H = 1280, 720
MAIN_W = 960  # 左の斜め上の視点（960×720）。右の 320 px に頭部カメラと情報
HEAD_SIZE = 288
HOLD_S = 1.0  # エピソードの終わりに結果を出して止める秒数
SECTION_S = 2.5  # 区間の見出しの画面
FONT_PATHS = {"regular": "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
              "bold": "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"}
IMG_POLICY = config.OUTPUTS_DIR / "act_g1_pick_img_v1" / "pretrained_model"
STATE_POLICY = config.EVAL_POLICY_DIRS["pick"]
BG, PANEL, FG, SUB = (18, 20, 26), (28, 31, 38), (236, 238, 242), (160, 166, 178)
OK_COLOR, NG_COLOR, ACCENT = (64, 186, 112), (222, 82, 82), (90, 160, 240)
TE = ("te", 0.01)  # 推論の条件: ("te", 係数) は temporal ensembling、("n", k) はチャンクのうち k 個実行して予測し直す
COND_LABELS = {TE: "temporal ensembling 0.01", ("n", 100): "100 ステップごとに予測", ("n", 10): "10 ステップごとに予測"}

# ("card", 秒, 種類) / ("section", 見出し, 説明の行) / ("episode", 方策, 条件, シード, 配置, 字幕)
# 方策は "teacher"（スクリプト方策）か "img"（画像入力 ACT）。配置は PickPlaceEnv.reset の cube・target（None ならシードで決まる）
SEED0 = config.EVAL_SEED_BASE
OOD_CAPTION = "範囲の外の 8 配置を試し、4 配置で成功、4 配置で失敗した。ここでは成功した 4 配置から 3 つを選んで見せる"
SEGMENTS = [
    ("card", 7.0, "title"),
    ("card", 11.0, "pipeline"),
    ("section", "1. 教師デモ（スクリプト方策）",
     ["mink の IK で手先の経路を解き、横からつかんで目標の円盤に置く",
      "この方策で、ランダムな配置のデモを 200 本作って学習データにした"]),
    ("episode", "teacher", None, SEED0, {}, "教師：キューブの真の位置を使って経路を計画する（学習データを作る側）"),
    ("section", "2. 画像入力の ACT（学習に使っていない配置）",
     ["入力は頭部カメラの画像と、右腕・右手の関節角（14）だけ",
      "キューブ・置き場所の座標は渡さない。画像から読んで動く"]),
    *[("episode", "img", TE, SEED0 + i, {}, "ACT：画像と関節角から、右腕・右手 14 関節の目標角を 50Hz で出す")
      for i in range(5)],
    ("section", "3. 学習範囲の外の配置",
     ["学習データの範囲：キューブ x 0.22〜0.38m、y −0.23〜−0.07m、yaw ±30°",
      "その外に置いても運べることがある（試した 8 配置中 4 配置で成功）"]),
    ("episode", "img", ("n", 100), SEED0, {"cube": (0.19, -0.15, 0.0), "target": (0.30, 0.0)},
     "キューブを範囲の手前（x = 0.19m）に置く。" + OOD_CAPTION),
    ("episode", "img", ("n", 100), SEED0, {"cube": (0.30, -0.26, 0.0), "target": (0.30, 0.0)},
     "キューブを範囲の右（y = −0.26m）に置く。" + OOD_CAPTION),
    ("episode", "img", ("n", 100), SEED0, {"cube": (0.30, -0.15, 40.0), "target": (0.30, 0.0)},
     "キューブを 40° 回す（学習は ±30°）。" + OOD_CAPTION),
    ("section", "4. 失敗例：予測し直す間隔が短いと止まる",
     ["100 ステップ分の予測のうち 10 ステップだけ実行して予測し直すと、",
      "50 配置中 47 配置で持ち上げられず、成功は 0。手が途中で止まったまま動かない"]),
    ("episode", "img", ("n", 10), SEED0, {},
     "この配置では差し込む直前で止まる。そこからの予測は「約 20 ステップ待ってから差し込む」で、最初の 10 ステップ（待つ部分）だけ実行するので観測が変わらない"),
    ("card", 12.0, "results"),
]


def add_arguments(parser) -> None:
    parser.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "showcase.mp4")
    parser.add_argument("--device", default="auto", help="auto（MPS があれば mps、無ければ cpu）/ mps / cpu")


def fmt(v: float) -> str:
    """座標の表示（-0.00 を 0.00 にする）。"""
    return f"{round(float(v), 2) + 0.0:.2f}"


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATHS[weight], size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    """日本語の字幕を 1 文字単位で幅 width に折り返す。"""
    lines, line = [], ""
    for ch in text:
        if draw.textlength(line + ch, font=fnt) > width:
            lines.append(line)
            line = ch
        else:
            line += ch
    return lines + [line] if line else lines


class Writer:
    """PIL の画像（W×H）を mp4（H.264, FPS）に書き出す。"""

    def __init__(self, path: Path):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=config.FPS, options={"crf": "20"})
        self.stream.width, self.stream.height = W, H
        self.stream.pix_fmt = "yuv420p"
        self._frame_cls = av.VideoFrame
        self.n_frames = 0

    def add(self, img: Image.Image, n: int = 1) -> None:
        frame = self._frame_cls.from_ndarray(np.asarray(img.convert("RGB")), format="rgb24")
        for _ in range(n):
            self.container.mux(self.stream.encode(frame))
        self.n_frames += n

    def hold(self, img: Image.Image, seconds: float) -> None:
        self.add(img, round(seconds * config.FPS))

    def close(self) -> None:
        self.container.mux(self.stream.encode())
        self.container.close()


# --- 静止画の画面 ---
def card_title() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 200), "Unitree G1 のピック＆プレースを", font=font(48, "bold"), fill=FG)
    d.text((80, 270), "画像入力の模倣学習（ACT）で", font=font(48, "bold"), fill=FG)
    d.text((80, 380), "MuJoCo シミュレーション ／ 上半身のみ・右手で机の上のキューブをつかみ、緑の円盤の上に置く",
           font=font(24), fill=SUB)
    d.text((80, 420), "動画はすべて等倍（1 倍速）。1 エピソード 9 秒", font=font(24), fill=SUB)
    d.rectangle((80, 340, 240, 344), fill=ACCENT)
    return img


def card_pipeline() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 70), "作ったもの", font=font(40, "bold"), fill=FG)
    steps = [("教師デモ", "mink の IK で手先の経路を解くスクリプト方策。本物の接触で把持（疑似把持なし）。\nランダムな配置で 200 本（9 万フレーム、50Hz）"),
             ("データセット", "デモを再生して頭部カメラの画像（224×224）を付け、LeRobot 形式に。\n入力は画像＋関節角 14、出力は関節目標角 14（キューブの座標は入れない）"),
             ("学習", "LeRobot の ACT（ResNet18 ＋ Transformer、100 ステップの行動チャンク）。\nGoogle Colab（T4）で 50,000 ステップ"),
             ("推論・評価", "Mac（M1）の MuJoCo で、学習に使っていない 50 配置で成功率を測る。\n推論の条件（予測し直す間隔）を変えて比べ、失敗の原因を調べた")]
    y = 160
    for i, (head, body) in enumerate(steps):
        d.ellipse((80, y, 124, y + 44), fill=ACCENT)
        d.text((102, y + 22), str(i + 1), font=font(26, "bold"), fill=BG, anchor="mm")
        d.text((150, y + 2), head, font=font(28, "bold"), fill=FG)
        d.multiline_text((150, y + 44), body, font=font(21), fill=SUB, spacing=8)
        y += 130
    return img


def eval_rate(policy_dir: Path, cond: str) -> str:
    path = policy_dir.parent / f"eval_pick_{cond}_50ep.json"
    s = json.loads(path.read_text())["summary"]
    return f"{round(s['success_rate'] * s['episodes'])}/{s['episodes']}"


def card_results() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 70), "成功率（学習に使っていない 50 配置）", font=font(40, "bold"), fill=FG)
    rows = [("推論の条件", "画像入力（この動画）", "参考：座標の真値を入力")]
    for cond, label in (("n100", "100 ステップ実行して予測し直す"), ("te0.01", "temporal ensembling（毎ステップ予測して平均）"),
                        ("n10", "10 ステップ実行して予測し直す")):
        rows.append((label, eval_rate(IMG_POLICY, cond), eval_rate(STATE_POLICY, cond)))
    xs, y = (80, 700, 940), 170
    for i, row in enumerate(rows):
        f = font(22 if i == 0 else 26, "bold" if i == 0 else "regular")
        for x, cell in zip(xs, row):
            d.text((x, y), cell, font=f, fill=SUB if i == 0 else FG)
        y += 56 if i == 0 else 70
        d.line((80, y - 16, W - 80, y - 16), fill=PANEL, width=2)
    notes = ["・100 ステップごと・temporal ensembling では、座標を渡したモデルと同程度以上に成功した",
             "・画像の中だけ円盤をずらすと、ずれた先に置く。本物のキューブを退避させて画像にだけ描いても、",
             "　手の軌跡はほぼ同じ（0.7cm 以内）。画像から位置を読んでいる",
             "・10 ステップごとに予測し直すと手が途中で止まる。差し込む直前で止まった配置では、そこから",
             "　「待ってから差し込む」予測を繰り返していた。予測し直す間隔を 30 ステップ以上（5 配置で確認）",
             "　にするか、temporal ensembling で避けられる"]
    y += 20
    for n in notes:
        d.text((80, y), n, font=font(22), fill=SUB)
        y += 40
    return img


def section_card(title: str, lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 250), title, font=font(44, "bold"), fill=FG)
    d.rectangle((80, 320, 240, 324), fill=ACCENT)
    for i, line in enumerate(lines):
        d.text((80, 350 + i * 44), line, font=font(26), fill=SUB)
    return img


# --- エピソード ---
class Episode:
    """1 エピソードを動かし、各ステップの画面を Writer に書く。"""

    def __init__(self, env: PickPlaceEnv, device: str):
        self.env, self.device = env, device
        m = env.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, MAIN_W), max(m.vis.global_.offheight, H)
        self.renderer = mujoco.Renderer(m, height=H, width=MAIN_W)
        self.camera = overview_camera(m)
        self.camera.distance *= 0.92
        self.policies = {}
        self.teacher = ScriptedPickPlacePolicy(env)

    def actor(self, cond):
        if cond not in self.policies:
            kind, v = cond
            policy, pre, post, _ = load_policy(IMG_POLICY, self.device, v if kind == "n" else None, v if kind == "te" else None)
            self.policies[cond] = (policy, make_actor(policy, pre, post))
        return self.policies[cond]

    def frame(self, obs, info: list[tuple[str, str]], title: str, caption: str, t: int) -> Image.Image:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        img = Image.new("RGB", (W, H), PANEL)
        img.paste(Image.fromarray(self.renderer.render()), (0, 0))
        img.paste(Image.fromarray(obs[config.IMAGE_KEY]).resize((HEAD_SIZE, HEAD_SIZE), Image.NEAREST), (MAIN_W + 16, 48))
        d = ImageDraw.Draw(img, "RGBA")
        d.text((MAIN_W + 16, 14), "頭部カメラ（方策の入力）", font=font(18, "bold"), fill=FG)
        y = 48 + HEAD_SIZE + 18
        for k, v in info + [("時間", f"{t / config.CONTROL_HZ:4.1f} / {config.PICK_EPISODE_STEPS / config.CONTROL_HZ:.0f} 秒")]:
            d.text((MAIN_W + 16, y), k, font=font(15), fill=SUB)
            for line in wrap(d, v, font(17), W - MAIN_W - 32):
                d.text((MAIN_W + 16, y + 20), line, font=font(17), fill=FG)
                y += 24
            y += 26
        tf = font(22, "bold")
        d.rectangle((0, 0, d.textlength(title, font=tf) + 32, 46), fill=(0, 0, 0, 170))
        d.text((16, 10), title, font=tf, fill=FG)
        d.text((MAIN_W - 12, 12), "等倍", font=font(18, "bold"), fill=FG, anchor="ra")
        cf = font(20)
        lines = wrap(d, caption, cf, MAIN_W - 40)
        top = H - 20 - 30 * len(lines)
        d.rectangle((0, top - 12, MAIN_W, H), fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            d.text((20, top + 30 * i), line, font=cf, fill=FG)
        return img

    def run(self, writer: Writer, who: str, cond, seed: int, place: dict, title: str, caption: str) -> str | None:
        env = self.env
        obs = env.reset(seed, **place)
        cube0 = env.cube_pos()
        yaw0 = place.get("cube", (0, 0, None))[2]
        if yaw0 is None:
            q = env.cube_quat()
            yaw0 = np.rad2deg(2 * np.arctan2(q[3], q[0]))
        if who == "teacher":
            self.teacher.reset()
            act = lambda _obs: self.teacher.act()  # noqa: E731
            info = [("方策", "教師（スクリプト方策、mink IK）")]
        else:
            policy, act = self.actor(cond)
            policy.reset()
            info = [("方策", "ACT（画像＋関節角を入力）"), ("推論", COND_LABELS[cond])]
        info += [("キューブ", f"({fmt(cube0[0])}, {fmt(cube0[1])}) m、yaw {int(round(yaw0)):+d}°"),
                 ("置き場所", f"({fmt(env.target[0])}, {fmt(env.target[1])}) m")]
        max_lift = 0.0
        for t in range(config.PICK_EPISODE_STEPS):
            writer.add(self.frame(obs, info, title, caption, t))
            obs = env.step(act(obs))
            max_lift = max(max_lift, float(env.cube_pos()[2] - cube0[2]))
        fail = pick_failure(env, max_lift)
        last = self.frame(obs, info, title, caption, config.PICK_EPISODE_STEPS)
        d = ImageDraw.Draw(last, "RGBA")
        text = "成功" if fail is None else f"失敗：{PICK_FAILURES[fail]}"
        bf = font(44, "bold")
        w = d.textlength(text, font=bf)
        x0, y0 = (MAIN_W - w) / 2 - 36, 280
        d.rounded_rectangle((x0, y0, x0 + w + 72, y0 + 84), radius=16, fill=(*(OK_COLOR if fail is None else NG_COLOR), 235))
        d.text((MAIN_W / 2, y0 + 42), text, font=bf, fill=(255, 255, 255), anchor="mm")
        d.text((MAIN_W / 2, y0 + 110), f"置き場所との距離 {env.place_error() * 100:.1f}cm ／ 持ち上げた高さ {max_lift * 100:.1f}cm",
               font=font(20), fill=FG, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
        writer.hold(last, HOLD_S)
        return fail


def run(args) -> int:
    missing = [p for p in FONT_PATHS.values() if not Path(p).exists()]
    if missing:
        sys.exit(f"日本語フォントが見つからない: {missing}（FONT_PATHS を書き換える）")
    # 成功率の表に使う eval の結果とモデルは、10 分以上かかる録画の最後で無いと分かると困るので先に確かめる
    needed = [IMG_POLICY / "config.json"]
    needed += [d.parent / f"eval_pick_{c}_50ep.json" for d in (IMG_POLICY, STATE_POLICY) for c in ("n100", "te0.01", "n10")]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        sys.exit("見つからない（モデルを置き、eval を n100・--temporal-ensemble 0.01・--n-action-steps 10 で回す）:\n  "
                 + "\n  ".join(missing))
    device = pick_device(args.device)
    env = PickPlaceEnv(images=True)
    episodes = Episode(env, device)
    writer = Writer(args.out)
    try:
        record(writer, episodes)
    finally:
        writer.close()
    print(f"\n{args.out}: {writer.n_frames} フレーム = {writer.n_frames / config.FPS:.1f} 秒（{config.FPS} fps, {W}×{H}）")
    return 0


def record(writer: Writer, episodes: Episode) -> None:
    cards = {"title": card_title, "pipeline": card_pipeline, "results": card_results}
    section = ""
    for seg in SEGMENTS:
        if seg[0] == "card":
            writer.hold(cards[seg[2]](), seg[1])
        elif seg[0] == "section":
            section = seg[1]
            writer.hold(section_card(seg[1], seg[2]), SECTION_S)
        else:
            _, who, cond, seed, place, caption = seg
            fail = episodes.run(writer, who, cond, seed, place, section, caption)
            print(f"{section} / {who} {cond} seed {seed} {place or ''}: {'成功' if fail is None else fail}", flush=True)
