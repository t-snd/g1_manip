"""保存したデータセットを Hugging Face Hub の private リポジトリへアップロードする（フェーズ6の準備）。

    python -m g1_manip push --repo-id <ユーザー名>/g1_reach
    python -m g1_manip push --root data/g1_pick_200 --repo-id <ユーザー名>/g1_pick

- 事前に `.venv/bin/hf auth login` で write 権限のトークンでログインしておく
- ローカルの repo_id（local/...）は読み込み時にしか使わないので、Hub 用の repo_id を指定して読み込み直してから push する
- push 後に Hub 上のリポジトリが private であることと、ファイル一覧を確認する。public だったら異常終了する
  （既存の public リポジトリへ push した場合、create_repo(exist_ok=True) は公開設定を変えないため）
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g1_manip import config


def add_arguments(parser) -> None:
    parser.add_argument("--root", type=Path, default=config.DATA_DIR / "g1_reach_200", help="ローカルのデータセット")
    parser.add_argument("--repo-id", required=True, help="Hub の repo_id（例: <ユーザー名>/g1_reach）")


def run(args) -> int:
    ds = LeRobotDataset(args.repo_id, root=args.root)
    print(f"{args.root}: {ds.meta.total_episodes} エピソード, {ds.meta.total_frames} フレーム → {args.repo_id}")
    ds.push_to_hub(private=True, tags=["g1", "mujoco", "simulation"])

    api = HfApi()
    info = api.dataset_info(args.repo_id)
    files = sorted(api.list_repo_files(args.repo_id, repo_type="dataset"))
    print(f"private={info.private}")
    for f in files:
        print(f"  {f}")
    if not info.private:
        sys.exit(f"{args.repo_id} が public になっている。Hub の Settings で private に変更すること")
    return 0
