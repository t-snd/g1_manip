"""MuJoCo のビューア（launch_passive）を制御周期に合わせて動かす（macOS では mjpython で実行すること）。"""

import sys
import time


def launch(model, data):
    from mujoco import viewer

    return viewer.launch_passive(model, data)


def sync(viewer, dt: float) -> None:
    """ビューアを更新し、前回の更新から dt 秒たつまで待つ（実時間で表示する）。ビューアが閉じられていたら終了する。"""
    if not viewer.is_running():
        sys.exit("ビューアが閉じられたため中断しました。")
    t0 = time.time()
    viewer.sync()
    time.sleep(max(0.0, dt - (time.time() - t0)))


def wait_until_closed(viewer) -> None:
    """終わった後の姿勢を表示したまま、ビューアが閉じられるまで待つ。"""
    print("ビューアを閉じると終了します。")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
