"""環境を行動の関数で 1 エピソード動かすループと、終わった時点の結果（成功・失敗の内訳）の取り出し（eval・analyze・gui で共通）。"""

import time

import numpy as np

from g1_manip import config
from g1_manip.env import PickPlaceEnv

# ピック＆プレースの失敗の内訳（pick_failure が返す値）
PICK_FAILURES = {"not_lifted": "持ち上げていない", "misplaced": "置き位置のずれ", "not_flat": "平らに置けていない",
                 "moving": "静止していない", "hand_contact": "手が触れたまま"}


def rollout(env, act, n_steps: int, obs: dict | None = None, on_step=None) -> dict:
    """env はリセット済み。act(obs) -> 行動 で n_steps 進める。obs はリセットが返した観測（省略すると env.observation()）。

    返す値: states（各ステップの観測時点の関節角, (T, 14) float64）、actions（(T, 14)）、infer_s（act にかかった秒数）、
    ピック＆プレースでは cube（各ステップの観測時点のキューブの位置, (T, 3)）と max_lift（キューブ中心の高さの最大値と初期の高さの差）。
    on_step は 1 ステップ進めるごとに呼ぶ（動画・ビューア用）。
    """
    obs = env.observation() if obs is None else obs
    pick = isinstance(env, PickPlaceEnv)
    cube0 = env.cube_pos() if pick else None
    states, cubes, actions, infer_s = [], [], [], []
    max_lift = 0.0
    for _ in range(n_steps):
        states.append(obs["observation.state"].astype(np.float64))
        if pick:
            cubes.append(env.cube_pos())
        t0 = time.perf_counter()
        action = act(obs)
        infer_s.append(time.perf_counter() - t0)
        actions.append(action)
        obs = env.step(action)
        if pick:
            max_lift = max(max_lift, float(env.cube_pos()[2] - cube0[2]))
        if on_step is not None:
            on_step()
    out = {"states": np.array(states), "actions": np.stack(actions), "infer_s": infer_s}
    if pick:
        out.update(cube=np.array(cubes), max_lift=max_lift)
    return out


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
