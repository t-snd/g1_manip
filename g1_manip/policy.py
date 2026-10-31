"""学習済み ACT 方策の読み込みと、環境の観測からの推論（eval・analyze・gui で共通）。"""

from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from g1_manip import config


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def needs_images(path: Path) -> bool:
    """モデルの入力に頭部カメラの画像（IMAGE_KEY）があるか（フェーズ8の画像入力版）。環境を images=True で作るかの判断に使う。"""
    return config.IMAGE_KEY in PreTrainedConfig.from_pretrained(path).input_features


def load_policy(path: Path, device: str, n_action_steps: int | None, te_coeff: float | None):
    """モデルと前処理・後処理を読み込む。推論の条件（n_action_steps・temporal ensembling）はここで上書きする。
    n_action_steps が 1〜chunk_size の外なら ValueError。"""
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if te_coeff is not None:
        cfg.temporal_ensemble_coeff = te_coeff
        cfg.n_action_steps = 1  # temporal ensembling は毎ステップ予測する
    elif n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if not 1 <= cfg.n_action_steps <= cfg.chunk_size:
        raise ValueError(f"--n-action-steps は 1〜chunk_size（{cfg.chunk_size}）で指定する")
    policy = ACTPolicy.from_pretrained(path, config=cfg)
    policy.eval()
    # Colab で学習したモデルは前処理の device が cuda になっているので、推論するデバイスに上書きする
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post, cfg


def condition_name(cfg) -> str:
    """推論の条件の短い名前（結果のファイル名に使う）: temporal ensembling なら te<係数>、それ以外は n<n_action_steps>。"""
    if cfg.temporal_ensemble_coeff is not None:
        return f"te{cfg.temporal_ensemble_coeff:g}"
    return f"n{cfg.n_action_steps}"


def obs_to_batch(obs: dict, input_features) -> dict[str, torch.Tensor]:
    """環境の観測から、モデルの入力にあるキーだけをテンソルにする。画像（H, W, 3）uint8 は、学習データを読み込んだときと
    同じ float32 (3, H, W)、0〜1 にする（LeRobotDataset は PNG をこの形で返す）。"""
    batch = {}
    for k in input_features:
        v = torch.from_numpy(obs[k])
        batch[k] = v.permute(2, 0, 1).float() / 255 if v.dtype == torch.uint8 else v
    return batch


def make_actor(policy, pre, post):
    """観測 → 行動（CONTROLLED_JOINTS の目標角）の関数。エピソードの始めに policy.reset() を呼ぶこと。"""
    @torch.inference_mode()
    def act(obs: dict) -> np.ndarray:
        return post(policy.select_action(pre(obs_to_batch(obs, policy.config.input_features)))).squeeze(0).numpy()
    return act
