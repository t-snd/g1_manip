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

## アクチュエータと行動（フェーズ3）

- menagerie の G1 はアクチュエータが全て**位置型（PD）**（ctrlrange＝関節可動域）なので、
  PD 制御の自作や置き換えは不要。**行動＝関節の目標角度（絶対値）[rad]** をそのまま `ctrl` に入れる。
- ただし menagerie のゲインは「needs tuning」（全関節 kp=500, dampratio=1）なので、`env.build_spec()` で次のように変えている
  （値は `config.py`）：
  - 腕：kp=500 / dampratio=1 のまま。dampratio は全関節0の姿勢の慣性で kv に換算されるため、
    ホーム姿勢での実際の減衰比は 0.43〜1.42（臨界減衰ではない）。
  - 指：kp=20 / kv=0.9（臨界減衰）。kp=500 だと 50Hz の目標更新ごとにトルク上限（1.4〜2.45Nm）に張り付き、全開・全閉のような動きになるため。
  - 重力補償：MuJoCo の `gravcomp` ＋ `actuatorgravcomp` で、重力による定常たわみを無くす（静止時に state＝action になる）。
    補償トルクはアクチュエータ経由で加わり、トルク上限に含まれる（`actuator_force` ではなく `qfrc_actuator` に現れる）。
  - 調査結果（フェーズ3）：急なステップ目標はトルク上限に当たってオーバーシュートする（現設定でも腕は最大約 40%）。
    滑らかな目標（min-jerk 0.5 秒）ならオーバーシュートは 0.5% 以下。デモ・方策の出力は滑らかな前提で、テストも滑らかな補間にしている。
    ただし腕（kp=500）の手首（上限 5Nm）は、滑らかな目標でも 50Hz の目標更新直後に数 ms 飽和する（移動中の約 15%）。追従には影響しない。
    実機寄りのゲイン（menagerie の `g1_mjx.xml`：腕 kp=75/kv=2、手首 kp=20/kv=2）は、腕の追従遅れが約 0.15rad になり IK の到達に不利なため不採用。
- 制御対象は `config.CONTROLLED_JOINTS`（右腕7関節 `RIGHT_ARM_JOINTS` ＋ 右手7関節 `RIGHT_HAND_JOINTS`、計14次元）。
  この並びが state/action ベクトルの次元順になる。
- インデックスは `env.joint_qpos_ids` / `env.joint_actuator_ids` で関節名から引き、
  目標の適用は `env.apply_joint_targets`、制御1周期（50Hz）の前進は `env.step_control` を使う。

```bash
.venv/bin/python scripts/02_check_actuators.py          # 一覧表示＋追従テスト（合否判定、失敗時は終了コード 1）
.venv/bin/mjpython scripts/02_check_actuators.py --view # 同じテストをビューアで表示
```

## IK（フェーズ4）

- `g1_manip/ik.py` の `RightHandIK` が mink で右手先の IK を解く（mink 公式例 `examples/humanoid_g1.py` と同じ `FrameTask` ＋ `PostureTask` 構成）。
  - 手先は `right_palm` サイト（`right_wrist_yaw_link` に追加、手のひら表面の中心。軸は x＝指の向き、z＝手のひらの法線）。
    Dex3 の右手は手のひらが wrist_yaw_link の +y 側を向き、親指は手のひらから約 6cm 飛び出す。
  - 右腕7関節以外（左腕・両手の指・キューブ）は `DofFreezingTask` の等式制約で厳密に固定し、可動域は `ConfigurationLimit`。
  - 姿勢は「手のひら下向き」（`PALM_DOWN_QUAT`）で、法線まわりの回転（yaw）はコスト 0 で自由にしている（固定すると手前で腕が胴体と干渉するため）。
  - `solve(target_pos, target_quat)` は前回の解から反復し（ウォームスタート）、右腕7関節の目標角を返す。IK は物理とは別の運動学モデル上で解く。
- エピソードは「準備姿勢」（`READY_PALM_POS`、手を机の上に上げて手のひらを下に向けた姿勢、`env.reset_to_ready`）から始める。
- 到達テスト：机上のランダムな20点（キューブの範囲10点＋目標マーカーの範囲10点）に、手のひらを天板＋7cm の高さで到達させる。
  経路は手先空間の min-jerk 補間で、50Hz ごとに IK→位置制御。結果は 20/20 点、誤差は最大 0.26cm（乱数シード 2〜5 と範囲の四隅も全点到達、最大 0.44cm）。

- フェーズ5への申し送り（フェーズ4の検証で判明）:
  - **手の yaw**：yaw が自由なので、指の向きは位置によって世界 x 軸から +28°〜+85° ばらつく。キューブの yaw（±30°）に
    手を合わせるには、yaw に弱いコストを付けて「キューブの yaw ＋ 90°の倍数のうち自然な向きに最も近いもの」を目標にするなどが必要。
    yaw を強く固定すると、手前側で肘・肩・手首が胴体と干渉する。
  - **親指**：ホームの指姿勢では親指の先が手のひらの真下（法線方向に約 6cm）にあり、手のひらの中心をキューブの真上に合わせて
    下ろすと親指がキューブに当たる。親指を外へ開いた「開」姿勢を決めるか、把持点を手のひらの中心からずらす必要がある。
  - 準備姿勢は肘を張った姿勢（肩 roll −1.33rad）だが、接触はなく可動域にも余裕がある。IK は1回平均 2.5ms（最大約 15ms）で 50Hz に収まる。

```bash
.venv/bin/python scripts/03_ik_reach_test.py          # 到達テスト（合否判定）
.venv/bin/mjpython scripts/03_ik_reach_test.py --view # ビューアで表示
```

### 後続フェーズへの申し送り（フェーズ2の検証で判明）

- 右手はアクチュエータの順（index→middle）と qpos の順（middle→index）が異なる。state/action は必ず関節名で引く。
- x=0.40 付近は右腕の到達限界に近いと見ていたが、フェーズ4の到達テストでは範囲の四隅も 0.44cm 以内で到達した（手首の yaw は自由）。
  ただし x=0.38〜0.40 では、姿勢正則化（PostureTask）とのトレードオフで 0.2〜0.4cm の定常的なずれが残る。
- ホーム姿勢では手が机より下にある。フェーズ5の approach の前に、手を机より上に上げる中継点を入れる。
  （フェーズ3で確認：ホームから腕を前に出すだけだと、手首が天板の下に引っかかって止まる。）
- ホーム姿勢（腕を下ろした状態）のまま指を開閉すると、指が固定された脚（hip リンク）に当たる。指の開閉は腕を体から離してから行う。
- 指は関節摩擦（frictionloss 0.3Nm、MuJoCo では低速で粘性のように効く）と kp=20 のため、目標に時定数約 0.5 秒で遅れて寄っていく
  （0.5 秒で閉じる指令の直後は約 0.02rad 遅れ、1 秒後 0.007rad、3 秒後ほぼ 0）。grasp では閉じ終わるまで 1 秒ほど待つか、目標を深めに指令する。
  握る力は「目標を接触位置よりどれだけ深く指令するか」×kp で決まり、上限は 1.4Nm（約 0.07rad 以上深く指令すると上限に達する）。
  静止中は摩擦拘束が ±0.3Nm 程度を負担しうるので、握る力にはその分の幅がある。
- 方策の出力（行動）が急に跳ぶとトルク上限に当たってオーバーシュートする。評価時に跳びが大きい場合は、行動の平滑化（ACT の temporal ensembling など）を検討する。
- 成功判定は xy 距離で行うか、目標マーカーの z（天板上面）にキューブの半辺を足して比べる。
- キューブと目標の範囲が y=-0.05 で接しているので、初期配置に最小距離の制約を入れる。
- 机やランダム化範囲を変えたら、頭部カメラの視野（`HEAD_CAMERA_YAW_DEG` / `PITCH_DEG`）を再確認する。

## 実行上の注意（macOS）

`mujoco.viewer.launch_passive` は macOS では `python` ではなく **`mjpython`** で実行する必要がある。
ビューアを使うスクリプトはすべて次のように実行する：

```bash
.venv/bin/mjpython scripts/01_view_scene.py
```
