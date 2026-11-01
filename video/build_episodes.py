"""評価のエピソードのシーン（compositions/ep_<名前>.html）を assets/data.json から作る。

    python video/build_episodes.py     # python -m g1_manip showcase の後に実行する

結果（成功・失敗の内訳、置き場所との距離）と配置は data.json（実際に動かした結果）から読む。字幕は CAPTIONS。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILURES = {"not_lifted": "持ち上げられない", "misplaced": "置き位置のずれ", "not_flat": "平らに置けていない",
            "moving": "静止していない", "hand_contact": "手が触れたまま"}
# 名前 → (左上のラベル, 字幕)
CAPTIONS = {
    "ok_far": ("成功例 1", "キューブが学習範囲の奥・右の端にある配置"),
    "ok_yaw": ("成功例 2", "キューブが +29° 回っている配置（学習範囲は ±30°）"),
    "ok_near": ("成功例 3", "キューブが手前・左にあり、置き場所に近い配置"),
    "ok_ood": ("成功例 4（学習範囲の外）", "キューブを学習範囲より 3 cm 手前（x = 0.19 m）に置いた配置。範囲の外は方向によって失敗することもある"),
    "ng_lift": ("失敗例 1", "つかみ損ね、キューブを少し動かしただけで持ち上げられない（50 配置中 1 件）"),
    "ng_place": ("失敗例 2", "置き場所まで運んだが、手を離した後にキューブが動き、6.4 cm ずれた（置き位置のずれは 50 配置中 2 件）"),
}

TEMPLATE = """<template>
  <style>
    #root {{ position: absolute; inset: 0; background: #14161b; font-family: "Noto Sans JP", sans-serif; color: #eceef2; }}
    #{p}-main {{ position: absolute; left: 0; top: 0; width: 1440px; height: 1080px; }}
    #{p}-tag {{ position: absolute; left: 32px; top: 28px; padding: 10px 22px; border-radius: 8px;
      background: rgba(20, 22, 27, 0.82); font-size: 30px; font-weight: 700; }}
    #{p}-tag span {{ color: {tag_color}; }}
    #{p}-speed {{ position: absolute; left: 1300px; top: 34px; padding: 6px 14px; border-radius: 6px;
      background: rgba(20, 22, 27, 0.82); font-size: 22px; font-weight: 700; }}
    #{p}-cap {{ position: absolute; left: 0; top: 950px; width: 1440px; height: 130px; padding: 26px 40px;
      background: rgba(20, 22, 27, 0.82); font-size: 30px; line-height: 1.45; }}
    #{p}-side {{ position: absolute; left: 1440px; top: 0; width: 480px; height: 1080px; background: #1c1f26; }}
    #{p}-plot-h {{ position: absolute; left: 1460px; top: 14px; font-size: 22px; font-weight: 700; color: #a0a6b2; }}
    #{p}-plot {{ position: absolute; left: 1440px; top: 46px; width: 480px; height: 410px; }}
    #{p}-head-h {{ position: absolute; left: 1460px; top: 476px; font-size: 22px; font-weight: 700; color: #a0a6b2; }}
    #{p}-head {{ position: absolute; left: 1456px; top: 510px; width: 448px; height: 448px; }}
    #{p}-info {{ position: absolute; left: 1460px; top: 966px; width: 440px; font-size: 21px; color: #a0a6b2; line-height: 1.45; }}
    #{p}-badge {{ position: absolute; left: 720px; top: 400px; transform: translateX(-50%); padding: 28px 56px 24px;
      border-radius: 18px; background: {badge_color}; text-align: center; white-space: nowrap; }}
    #{p}-badge-big {{ font-size: 60px; font-weight: 700; color: #ffffff; }}
    #{p}-badge-small {{ font-size: 26px; color: #ffffff; margin-top: 6px; font-variant-numeric: tabular-nums; }}
  </style>
  <div id="root" data-composition-id="ep_{name}" data-width="1920" data-height="1080">
    <video id="{p}-main" class="clip" data-start="0" data-duration="{dur}" data-track-index="0"
           src="assets/{name}_main.mp4" muted playsinline></video>
    <div id="{p}-tag">評価 ・ <span>{tag}</span></div>
    <div id="{p}-speed">等倍</div>
    <div id="{p}-cap">{caption}</div>
    <div id="{p}-side"></div>
    <div id="{p}-plot-h">初期配置（上から見た図）</div>
    <img id="{p}-plot" src="assets/placements_{name}.png" alt="" />
    <div id="{p}-head-h">頭部カメラ（方策の入力 224×224）</div>
    <video id="{p}-head" class="clip" data-start="0" data-duration="{dur}" data-track-index="1"
           src="assets/{name}_head.mp4" muted playsinline></video>
    <div id="{p}-info">{origin}<br />キューブ ({cx}, {cy}) m、{yaw:+d}°<br />置き場所 ({tx}, {ty}) m</div>
    <div id="{p}-badge" class="clip" data-start="{badge_at}" data-duration="{badge_dur}" data-track-index="2">
      <div id="{p}-badge-big">{result}</div>
      <div id="{p}-badge-small">置き場所との距離 {err:.1f} cm ／ 持ち上げた高さ {lift:.1f} cm</div>
    </div>
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    tl.fromTo("#{p}-cap", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 0.2);
    tl.fromTo("#{p}-badge-big", {{ opacity: 0, scale: 0.8 }}, {{ opacity: 1, scale: 1, duration: 0.3, ease: "back.out(2)" }}, {badge_at});
    tl.fromTo("#{p}-badge-small", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3 }}, {badge_at} + 0.15);
    window.__timelines["ep_{name}"] = tl;
  </script>
</template>
"""


def main() -> None:
    data = json.loads((ROOT / "assets" / "data.json").read_text())
    dur = data["episode_s"] + data["hold_s"]
    for ep in data["episodes"]:
        if ep["policy"] != "act":
            continue
        name, ok = ep["name"], ep["failure"] is None
        tag, caption = CAPTIONS[name]
        html = TEMPLATE.format(
            p=name.replace("_", ""), name=name, dur=dur, tag=tag, caption=caption,
            origin="配置を指定（シードでは決めていない）" if ep["place_given"] else f"評価用シード {ep['seed']}",
            cx=f"{ep['cube'][0]:.2f}", cy=f"{ep['cube'][1]:.2f}", yaw=int(round(ep["yaw_deg"])) + 0,
            tx=f"{ep['target'][0]:.2f}", ty=f"{ep['target'][1]:.2f}",
            result="成功" if ok else f"失敗：{FAILURES[ep['failure']]}", err=ep["place_error_cm"], lift=ep["max_lift_cm"],
            badge_at=data["episode_s"], badge_dur=data["hold_s"],
            tag_color="#40ba70" if ok else "#e66767", badge_color="rgba(48, 150, 88, 0.94)" if ok else "rgba(200, 64, 64, 0.94)")
        (ROOT / "compositions" / f"ep_{name}.html").write_text(html)
        print(f"ep_{name}.html: {'成功' if ok else ep['failure']}")


if __name__ == "__main__":
    main()
