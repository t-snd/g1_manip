# g1_manip

Unitree G1（下半身固定・上半身のみ）が机上のキューブを右手でつかんで目標位置に置くタスクを、
MuJoCo + mink（IK によるスクリプトデモ）+ LeRobot ACT（模倣学習）で学習させるプロジェクト。
作業手順は `../g1_ACT_plan.md` を参照。

## 環境構築（Mac, Apple Silicon）

```bash
brew install uv                         # 未導入の場合
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# 完全に同じ環境を再現する場合:
# uv pip install --python .venv/bin/python -r requirements-lock.txt

git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie fetch --depth 1 origin 0059d4335f8156206f63a35662313385f7ad6d74
git -C third_party/mujoco_menagerie checkout 0059d4335f8156206f63a35662313385f7ad6d74

source .venv/bin/activate
python -c "import mujoco, mink, lerobot"
```

### バージョン

| パッケージ | バージョン |
|---|---|
| Python | 3.12.15 |
| mujoco | 3.15.0 |
| mink | 1.3.0（QP ソルバ: daqp 0.10.3 / quadprog 0.1.13） |
| lerobot | 0.6.1（extras: `dataset`, `training`） |
| torch / torchvision | 2.10.0 / 0.25.0（MPS 利用可） |
| numpy | 2.2.6 |
| mujoco_menagerie | commit `0059d43` |

- 手順書では Python 3.10/3.11 だが、**lerobot 0.5.0 以降は Python>=3.12 必須**のため 3.12 を採用。
- lerobot 0.6.1 では、データセット機能（`LeRobotDataset`）と学習（`lerobot-train`）の依存が extras に
  分かれている。**素の `lerobot` ではなく `lerobot[dataset,training]==0.6.1` を入れること**（Colab でも同じ）。
  Colab の Python が 3.12 以上であることも確認すること。
- `uv pip install lerobot` をバージョン無指定で実行すると、lerobot 0.4.4 に解決されることがあった。
  必ず `requirements.txt` から入れること。
- 動画のエンコード・デコード（torchcodec）には FFmpeg が必要（フェーズ8の画像入力で使う）：`brew install ffmpeg`。
  未導入だと lerobot の import 時に torchcodec の警告が出るが、状態ベースのデータセットには影響しない。

## シーン構成（フェーズ2）

- `assets/scene_table.xml`：床・ライト・机・キューブ・目標マーカーだけを定義する（G1 本体は含まない）。
- `g1_manip/env.py` の `build_spec()` が `MjSpec` で menagerie の `g1_with_hands.xml` を読み込み、
  骨盤の freejoint と脚・腰の関節（とそのアクチュエータ）を削除して、このシーンに attach する。
  - G1 の `meshdir` は相対パスなので、`<include>` ではなく `MjSpec` でメッシュを絶対パスに直してから組み立てる。
  - 下半身は関節ゼロ角（直立）の剛体としてワールドに固定される。残る関節は腕 7×2 ＋ 手 7×2 の計 28（アクチュエータは全て位置型）。
- 座標系：G1 の骨盤直下が原点、+x が正面。机の天板上面は z=0.85m、手前の縁は x=0.15m。
- 目標マーカーは mocap ボディ（`data.mocap_pos` で動かす）。頭部カメラ `head` は `torso_link` に付けている。
- 寸法・初期姿勢・カメラ設定などの定数は `g1_manip/config.py` に集約している。

```bash
.venv/bin/mjpython scripts/01_view_scene.py              # ビューアで表示
.venv/bin/python scripts/01_view_scene.py --headless 5   # ビューアなしで 5 秒動かし、静止を自動チェック
```

### 後続フェーズへの申し送り（フェーズ2の検証で判明）

- 右手はアクチュエータの順（index→middle）と qpos の順（middle→index）が異なる。state/action は必ず関節名で引く。
- x=0.40 付近は右腕の到達限界に近い（フェーズ4で到達しにくければ、キューブ基準 x や範囲を調整する）。手首の yaw は固定しない。
- ホーム姿勢では手が机より下にある。フェーズ5の approach の前に、手を机より上に上げる中継点を入れる。
- 成功判定は xy 距離で行うか、目標マーカーの z（天板上面）にキューブの半辺を足して比べる。
- キューブと目標の範囲が y=-0.05 で接しているので、初期配置に最小距離の制約を入れる。
- 机やランダム化範囲を変えたら、頭部カメラの視野（`HEAD_CAMERA_YAW_DEG` / `PITCH_DEG`）を再確認する。

## 実行上の注意（macOS）

`mujoco.viewer.launch_passive` は macOS では `python` ではなく **`mjpython`** で実行する必要がある。
ビューアを使うスクリプトはすべて次のように実行する：

```bash
.venv/bin/mjpython scripts/01_view_scene.py
```
