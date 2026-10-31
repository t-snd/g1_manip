"""ピック＆プレースの学習データの読み込みと、評価の初期配置との近さ（eval・analyze・gui・add-images で使う）。

- 学習データは parquet を直接読む（LeRobotDataset で 1 サンプルずつ読むと 9 万フレームで数分かかるため）
- 「近さ」はモデルが見るのと同じ正規化（meta/stats.json の mean/std）をした environment_state の距離で測る。
  キューブの位置・姿勢（quat）・目標の位置をまとめて 1 つの距離にする。キューブ xy だけの距離 [m] も別に出す
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

ENV_KEY = "observation.environment_state"
STATE_KEY = "observation.state"


def load_episodes(root: Path) -> dict[str, np.ndarray]:
    """学習データの全エピソードを {キー: (エピソード数, ステップ数, 次元)} で返す（全エピソード同じ長さが前提）。
    episode_index の順に並べ、g1_generation.json があれば各エピソードの乱数シードも seeds に入れる。"""
    files = sorted((root / "data").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values(["episode_index", "frame_index"])
    lengths = df.groupby("episode_index").size()
    if lengths.nunique() != 1:
        raise ValueError(f"{root}: エピソードの長さが揃っていない（{sorted(lengths.unique())}）")
    n_ep, n_t = len(lengths), int(lengths.iloc[0])
    out = {k: np.stack(df[k].to_numpy()).reshape(n_ep, n_t, -1).astype(np.float64)
           for k in (STATE_KEY, ENV_KEY, "action")}
    out["episode_index"] = lengths.index.to_numpy()
    gen = root / "g1_generation.json"
    if gen.exists():
        seeds = json.loads(gen.read_text())["seeds"]
        out["seeds"] = np.array([seeds[i] for i in out["episode_index"]])
    return out


def state_dataset(root: Path) -> tuple[Path, list[int] | None]:
    """environment_state のあるデータセットと、そのうち root の学習データに入っているエピソード（None なら全部）。
    画像入力のデータセット（add-images で作ったもの。environment_state が無い）なら、g1_generation.json に記録した
    元のデータセット（root の親からの相対パス）と、使ったエピソード（source_episode_index。先頭の一部だけのこともある）を返す。"""
    gen = root / "g1_generation.json"
    if gen.exists() and (src := (info := json.loads(gen.read_text())).get("source_dataset")):
        return (root.parent / src).resolve(), info["source_episode_index"]
    return root, None


def env_stats(root: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads((root / "meta" / "stats.json").read_text())[ENV_KEY]
    return np.asarray(stats["mean"]), np.asarray(stats["std"])


def cube_yaw_deg(env_state: np.ndarray) -> np.ndarray:
    """environment_state（..., 10）のキューブ quat（wxyz, w >= 0）から yaw [deg]。"""
    return np.rad2deg(2 * np.arctan2(env_state[..., 6], env_state[..., 3]))


class TrainingNeighbors:
    """評価時の初期配置に、学習データのどのエピソードの初期配置が近いかを調べる。"""

    def __init__(self, root: Path, episodes: dict[str, np.ndarray] | None = None, only: list[int] | None = None):
        """only: 使うエピソード（episode_index）。None なら全部。正規化の統計は root の meta/stats.json（全エピソード）のまま。"""
        self.episodes = episodes if episodes is not None else load_episodes(root)
        if only is not None:
            keep = np.isin(self.episodes["episode_index"], only)
            self.episodes = {k: v[keep] for k, v in self.episodes.items()}
        self.mean, self.std = env_stats(root)
        self.env0 = self.episodes[ENV_KEY][:, 0]  # (エピソード数, 10)

    def nearest(self, env_state0: np.ndarray) -> dict:
        """正規化した environment_state で最も近い学習エピソードと、その距離・キューブ xy の距離。"""
        z = (self.env0 - self.mean) / self.std
        z0 = (np.asarray(env_state0, dtype=float) - self.mean) / self.std
        d = np.linalg.norm(z - z0, axis=1)
        i = int(d.argmin())
        d_xy = np.linalg.norm(self.env0[:, :2] - env_state0[:2], axis=1)
        return {
            "train_episode": int(self.episodes["episode_index"][i]),
            "train_row": i,
            "dist_norm": float(d[i]),
            # 正規化距離で最も近いエピソードのキューブ xy・yaw・目標 xy の差
            "nn_cube_xy_m": float(d_xy[i]),
            "nn_yaw_deg": float(cube_yaw_deg(self.env0[i]) - cube_yaw_deg(np.asarray(env_state0))),
            "nn_target_xy_m": float(np.linalg.norm(self.env0[i, 7:9] - env_state0[7:9])),
            # キューブ xy だけで最も近い学習エピソードとの距離（目標・yaw は無視）
            "min_cube_xy_m": float(d_xy.min()),
        }

    def self_nn(self) -> np.ndarray:
        """学習エピソードどうしの最近傍の正規化距離（自分を除く）。nearest の dist_norm の目安にする。"""
        z = (self.env0 - self.mean) / self.std
        d = np.linalg.norm(z[:, None] - z[None], axis=2)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)


def palm_positions(env, states: np.ndarray) -> np.ndarray:
    """observation.state（CONTROLLED_JOINTS の関節角, (T, 14)）から順運動学で手のひらサイトの位置（T, 3）を求める。
    ロボットは土台が固定で、手のひらの位置は右腕の関節だけで決まる。env.data は書き換えるので、呼んだ後は reset すること。
    シミュレーション中の env.palm_pos()（site_xpos）は mj_step の積分前の値で物理 1 サブステップ分遅れるため、
    同じ時刻でも 1mm 未満ずれる（実測で最大 0.73mm。こちらが関節角と整合する）。"""
    out = np.empty((len(states), 3))
    for t, q in enumerate(states):
        env.data.qpos[env.state_qpos] = q
        mujoco.mj_kinematics(env.model, env.data)
        out[t] = env.data.site_xpos[env.palm_site]
    return out
