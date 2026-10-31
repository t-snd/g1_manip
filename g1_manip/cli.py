"""g1_manip のコマンドライン（python -m g1_manip <コマンド> ...）。

コマンドごとのモジュール（g1_manip/commands/）は、呼ばれたものだけを読み込む（torch・lerobot の読み込みに数秒かかるため、
一覧や他のコマンドのヘルプでは読み込まない）。各モジュールは add_arguments(parser) と run(args) を持ち、run の戻り値が終了コードになる。
"""

import argparse
import importlib
import sys

# コマンド名 → (モジュール, 一覧に出す説明)。並びはパイプラインの順
COMMANDS = {
    "scene": ("scene", "シーンをビューアで表示する／ビューアなしで静止を確かめる（フェーズ2）"),
    "actuators": ("actuators", "アクチュエータの一覧と、右腕・右手の追従テスト（フェーズ3）"),
    "ik-test": ("ik_test", "IK で机上の点に手のひらを持っていく到達テスト（フェーズ4）"),
    "demo": ("demo", "スクリプト方策（教師）を動かして成功率を見る。保存はしない（フェーズ5）"),
    "generate": ("generate", "スクリプト方策でデモを生成して LeRobot 形式で保存し、リプレイ検証する（フェーズ5）"),
    "add-images": ("add_images", "状態のみのピック＆プレースのデモを再生し、頭部カメラの画像を付ける（フェーズ8）"),
    "push": ("push", "データセットを Hugging Face Hub の private リポジトリへ上げる（フェーズ6の準備）"),
    "eval": ("evaluate", "学習済み ACT 方策をシミュレーションで動かし、成功率を出す（フェーズ7）"),
    "analyze": ("analyze", "ピック＆プレースの方策を学習データ・教師・最近傍リプレイと比べる（フェーズ7の追加分析）"),
    "gui": ("gui", "上面図でキューブと置き場所を置いて、方策・教師・最近傍リプレイを動かす GUI"),
}
PROG = "python -m g1_manip"


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"使い方: {PROG} <コマンド> [オプション]（コマンドごとのオプションは {PROG} <コマンド> -h）", "", "コマンド:"]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(usage())
    if argv[0] in ("-h", "--help"):
        print(usage())
        sys.exit(0)
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        sys.exit(f"不明なコマンド: {name}\n\n{usage()}")
    module = importlib.import_module(f"g1_manip.commands.{COMMANDS[name][0]}")
    parser = argparse.ArgumentParser(prog=f"{PROG} {name}", description=module.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    module.add_arguments(parser)
    args = parser.parse_args(rest)
    sys.exit(module.run(args) or 0)
