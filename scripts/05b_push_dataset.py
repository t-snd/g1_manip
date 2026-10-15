"""保存したデータセットを Hugging Face Hub の private リポジトリへアップロードする（フェーズ6の準備）。

    python scripts/05b_push_dataset.py --repo-id <ユーザー名>/g1_reach

- 事前に `.venv/bin/hf auth login` で write 権限のトークンでログインしておく
- ローカルの repo_id（local/...）は読み込み時にしか使わないので、Hub 用の repo_id を指定して読み込み直してから push する
- push 後に Hub 上のリポジトリが private であることと、ファイル一覧を確認する。public だったら異常終了する
  （既存の public リポジトリへ push した場合、create_repo(exist_ok=True) は公開設定を変えないため）
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=config.DATA_DIR / "g1_reach_200", help="ローカルのデータセット")
    parser.add_argument("--repo-id", required=True, help="Hub の repo_id（例: <ユーザー名>/g1_reach）")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
