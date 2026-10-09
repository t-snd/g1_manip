"""LeRobot 形式でのデモの保存・読み込み（lerobot 0.6.1 の LeRobotDataset を使う）。

保存の流れ: LeRobotDataset.create → add_frame（1ステップごと）→ save_episode（1エピソードごと）→ finalize。
各フレームは「その時点の観測」と「その観測を見て出した行動（＝そのステップで位置制御に与えた目標角）」の組。
"""

import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g1_manip import config

GEN_INFO_FILE = "g1_generation.json"  # 生成時の乱数シードなどを記録するファイル（データセットのルート直下）


def make_features(env_state_names: list[str]) -> dict:
    joints = list(config.CONTROLLED_JOINTS)
    return {
        "observation.state": {"dtype": "float32", "shape": (len(joints),), "names": joints},
        "observation.environment_state": {"dtype": "float32", "shape": (len(env_state_names),),
                                          "names": env_state_names},
        "action": {"dtype": "float32", "shape": (len(joints),), "names": joints},
    }


def create_dataset(root: Path, repo_id: str, env_state_names: list[str], overwrite: bool) -> LeRobotDataset:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} が既にある（上書きするなら --overwrite）")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.FPS,
        features=make_features(env_state_names),
        root=root,
        robot_type="unitree_g1_fixed_base_right_arm_dex3",
        use_videos=False,
    )


def save_episode(ds: LeRobotDataset, frames: list[dict], task: str) -> None:
    for f in frames:
        ds.add_frame({**f, "task": task})
    ds.save_episode()


def apply_std_floor(root: Path) -> dict[str, int]:
    """meta/stats.json の std に下限（STATS_STD_FLOOR）を付ける。finalize の後に呼ぶ。

    LeRobot の正規化は (x - mean) / (std + 1e-8)。リーチングでは指の関節がほぼ動かないため
    std が 1e-6 程度になり、評価時のわずかなずれが数十〜数百倍に拡大されて学習時に無い入力になる。
    学習（lerobot-train）は meta/stats.json の値を使うので、ここで下限を付けておく。
    戻り値: キーごとに下限を適用した次元の数。
    """
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    floored = {}
    for key in ("observation.state", "observation.environment_state", "action"):
        std = np.asarray(stats[key]["std"], dtype=float)
        floored[key] = int((std < config.STATS_STD_FLOOR).sum())
        stats[key]["std"] = np.maximum(std, config.STATS_STD_FLOOR).tolist()
    path.write_text(json.dumps(stats, indent=4))
    return floored


def write_generation_info(root: Path, info: dict) -> None:
    (root / GEN_INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False))


def read_generation_info(root: Path) -> dict:
    return json.loads((root / GEN_INFO_FILE).read_text())


def load_episodes(root: Path, repo_id: str) -> list[dict[str, np.ndarray]]:
    """保存したデータセットを読み込み、エピソードごとに {キー: (T, dim) 配列} の dict のリストで返す。"""
    ds = LeRobotDataset(repo_id, root=root)
    keys = ["observation.state", "observation.environment_state", "action"]
    episodes: dict[int, dict[str, list]] = {}
    for i in range(len(ds)):
        item = ds[i]
        ep = episodes.setdefault(int(item["episode_index"]), {k: [] for k in keys + ["frame_index"]})
        for k in keys:
            ep[k].append(item[k].numpy())
        ep["frame_index"].append(int(item["frame_index"]))
    out = []
    for idx in sorted(episodes):
        ep = episodes[idx]
        order = np.argsort(ep["frame_index"])
        out.append({k: np.stack(ep[k])[order] for k in keys} | {"episode_index": idx})
    return out
