"""レポートの図: 行動チャンクのうち何ステップ実行して予測し直すか（n）と成功率（画像入力、学習に使っていない 50 配置）。

    .venv/bin/python docs/report/chunk_plot.py      # docs/report/chunk_success.png

eval で保存した outputs/act_g1_pick_img_v1/eval_pick_n<n>_50ep.json と、参照線の eval_pick_te0.01_50ep.json から読む。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "outputs" / "act_g1_pick_img_v1"
OUT = Path(__file__).with_name("chunk_success.png")
NS = (1, 10, 12, 14, 16, 18, 20, 100)
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
INK, SUB, GRID, LINE, REF = "#0b0b0b", "#52514e", "#d9d8d3", "#2a78d6", "#52514e"


def success(cond: str) -> tuple[int, int]:
    s = json.loads((EVAL_DIR / f"eval_pick_{cond}_50ep.json").read_text())["summary"]
    return round(s["success_rate"] * s["episodes"]), s["episodes"]


def main() -> None:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    rows = [(n, *success(f"n{n}")) for n in NS]
    te, te_total = success("te0.01")
    fig, ax = plt.subplots(figsize=(3.6, 2.3), dpi=200)
    xs, ys = list(range(len(rows))), [100 * r[1] / r[2] for r in rows]  # 横軸は n を等間隔に並べる
    ax.axhline(100 * te / te_total, color=REF, lw=1, ls=(0, (4, 3)))
    ax.text(-0.2, 100 * te / te_total + 2.5, f"temporal ensembling {te}/{te_total}", color=SUB, fontsize=6.5)
    ax.plot(xs, ys, color=LINE, lw=1.6, marker="o", ms=4, mec="white", mew=0.8)
    for x, (n, ok, total) in zip(xs, rows):
        ax.annotate(f"{ok}", (x, 100 * ok / total), textcoords="offset points", xytext={18: (-6, 5), 100: (0, -11)}.get(n, (0, 5)),
                    ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(xs, [str(r[0]) for r in rows])
    ax.set_xlim(-0.4, len(rows) - 0.6)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("1 回の予測で実行するステップ数 n（チャンク長 100。等間隔ではない）", fontsize=7, color=SUB)
    ax.set_ylabel("成功率 [%]（点の数字は 50 配置中の成功数）", fontsize=7, color=SUB)
    ax.tick_params(labelsize=6.5, colors=SUB, length=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)
    print(OUT, rows, f"te0.01 {te}/{te_total}")


if __name__ == "__main__":
    main()
