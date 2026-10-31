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
- 動画のエンコード・デコード（torchcodec）には FFmpeg が必要：`brew install ffmpeg`。このプロジェクトのデータセットは動画を使わない
  （フェーズ8の画像も PNG で parquet に埋め込む）ので不要。未導入だと lerobot の import 時に torchcodec の長い警告（Traceback）が出るが、影響はない。

## 使い方（コマンド）

すべての操作は `python -m g1_manip <コマンド>` で行う（**リポジトリのルートで実行する**。コマンドごとのオプションは `-h`）。
ビューアを開くもの（`scene` の既定・`--view`）は macOS では `mjpython` で実行する（末尾の「実行上の注意」）。

```bash
.venv/bin/python -m g1_manip --help          # コマンドの一覧
.venv/bin/python -m g1_manip eval -h         # コマンドごとのオプション
```

| コマンド | 内容 | フェーズ |
|---|---|---|
| `scene` | シーンをビューアで表示／`--headless` でビューアなしに静止を確かめる | 2 |
| `actuators` | アクチュエータの一覧と、右腕・右手の追従テスト | 3 |
| `ik-test` | IK で机上の 20 点に手のひらを持っていく到達テスト | 4 |
| `demo` | スクリプト方策（教師）を動かして成功率を見る（保存しない） | 5 |
| `generate` | スクリプト方策でデモを生成して LeRobot 形式で保存し、リプレイ検証する | 5 |
| `add-images` | 状態のみのピック＆プレースのデモを再生し、頭部カメラの画像を付ける | 8 |
| `push` | データセットを Hugging Face Hub の private リポジトリへ上げる | 6 の準備 |
| `eval` | 学習済み ACT 方策をシミュレーションで動かし、成功率を出す | 7 |
| `analyze dist / traj / grid` | ピック＆プレースの方策を学習データ・教師・最近傍リプレイと比べる | 7 |
| `gui` | 上面図でキューブと置き場所を置いて、方策・教師・最近傍リプレイを動かす | 7・8 |

学習（フェーズ6）は Colab のノートブック（`notebooks/`）で行う。

### パッケージの構成

| モジュール | 内容 |
|---|---|
| `g1_manip/config.py` | 定数・ハイパーパラメータ（関節名、範囲、しきい値、既定のパス） |
| `g1_manip/env.py` | シーンの組み立て、`ReachEnv`・`PickPlaceEnv`（リセット・観測・行動・成功判定） |
| `g1_manip/ik.py` | mink による右手先の IK（`RightHandIK`） |
| `g1_manip/scripted.py` | スクリプト方策（教師）：`ScriptedReachPolicy`・`ScriptedPickPlacePolicy` |
| `g1_manip/dataset.py` | LeRobot 形式での保存・読み込み、std の下限、生成時の記録（`g1_generation.json`） |
| `g1_manip/training_data.py` | 学習データの parquet の読み込み、初期配置が最も近い学習エピソード（`TrainingNeighbors`） |
| `g1_manip/policy.py` | 学習済み ACT の読み込み（推論の条件の上書き）、観測 → 行動 |
| `g1_manip/rollout.py` | 1 エピソード動かすループ、失敗の内訳、最終状態の取り出し |
| `g1_manip/video.py`・`viewer.py` | 斜め上の視点の mp4、ビューアの実時間表示 |
| `g1_manip/cli.py`・`commands/` | コマンドライン（呼ばれたコマンドのモジュールだけを読み込む） |

### 旧スクリプトとの対応

2026-10 のリファクタリングで `scripts/` の番号付きスクリプトを上のコマンドにまとめた（オプションは同じ）。
ノートブックのコメントに残っている旧スクリプト名は次のように読み替える。

| 旧スクリプト | コマンド |
|---|---|
| `scripts/01_view_scene.py` | `scene` |
| `scripts/02_check_actuators.py` | `actuators` |
| `scripts/03_ik_reach_test.py` | `ik-test` |
| `scripts/04_scripted_demo.py` | `demo` |
| `scripts/05_generate_dataset.py` | `generate` |
| `scripts/05b_push_dataset.py` | `push` |
| `scripts/05c_add_images.py` | `add-images` |
| `scripts/06_eval_policy.py` | `eval` |
| `scripts/07_analyze_pick.py` | `analyze` |
| `scripts/08_pick_gui.py` | `gui` |

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
.venv/bin/mjpython -m g1_manip scene              # ビューアで表示
.venv/bin/python -m g1_manip scene --headless 5   # ビューアなしで 5 秒動かし、静止を自動チェック
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
.venv/bin/python -m g1_manip actuators          # 一覧表示＋追従テスト（合否判定、失敗時は終了コード 1）
.venv/bin/mjpython -m g1_manip actuators --view # 同じテストをビューアで表示
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
.venv/bin/python -m g1_manip ik-test          # 到達テスト（合否判定）
.venv/bin/mjpython -m g1_manip ik-test --view # ビューアで表示
```

## スクリプト方策とデモ生成（フェーズ5）

計画書 5-b に従い、まず**リーチングタスク**（準備姿勢から机の上のランダムな点へ手のひらを持っていく）でフェーズ5〜7を通す。

- `env.ReachEnv`：リセット（準備姿勢、目標点をランダムに決めて目標マーカーで表示、キューブは机の奥の隅へ）、観測、行動の適用（50Hz）、成功判定。
  - 開始時の手のひら位置も準備姿勢から ±3cm（高さ ±2cm）ずらす（IK で腕の関節角を求める）。初期状態が毎回同じだと、
    評価時に observation.state の配線を間違えても結果に表れにくいため。
  - 目標点：x 0.20〜0.40m、y −0.25〜+0.05m、天板上 7〜15cm。成功：最終ステップで手のひらが目標点から 2cm 以内かつ静止。
- `scripted.ScriptedReachPolicy`：手のひらを目標点へ min-jerk 補間（1.5 秒）で動かし、50Hz ごとに IK で右腕の目標角を出す。指はホームの目標のまま。1エピソード 150 ステップ（3 秒）。
- 保存形式（LeRobot v3.0、`dataset.py`）：

  | キー | 中身 | 次元 |
  |---|---|---|
  | `observation.state` | 右腕7関節＋右手7関節の関節角（`CONTROLLED_JOINTS` の順） | 14 |
  | `observation.environment_state` | 目標点の位置（リーチング） | 3 |
  | `action` | そのステップで位置制御に与えた目標角（絶対値） | 14 |
  | `task` | "reach the target point with the right palm" | — |

  - 成功したエピソードだけ保存する。使った乱数シードは `<root>/g1_generation.json` に記録する（評価は別のシード `EVAL_SEED_BASE` 以降を使う）。
  - `meta/stats.json` の std に下限 0.01 を付けている（`STATS_STD_FLOOR`）。リーチングでは指がほぼ動かず std が 1e-6 程度になり、
    LeRobot の正規化（÷(std+1e-8)）で評価時のわずかなずれが数十〜数百倍に拡大されるのを防ぐため。学習は `meta/stats.json` を使う。
    下限は `meta/stats.json` にだけ入る（`meta/episodes/` のエピソードごとの統計は元のまま）。lerobot のツールでエピソードの
    削除・分割・統合をして統計を作り直した場合は、`dataset.apply_std_floor` を再実行すること。
- リプレイ検証（保存形式のバグ検出）：保存したデータを読み込み、同じ初期状態から保存した action だけで再生して、
  タスク成功・全フレームで関節角が保存した `observation.state` と一致・`environment_state` が一致、を確認する。
  - 成功判定（2cm）は方策の精度（約 0.1cm）より緩いので、成功判定だけでは保存形式のバグを検出できない。**検証の本体は一致判定**。
    action を1ステップずらす・state と取り違える・腕の action を 0.95 倍する・environment_state をずらす、といったバグは一致判定で NG になることを確認済み。
  - リーチングでは指がほぼ動かないので、指の列（index/middle）の取り違えは検出できない（ピック＆プレースで指が動けば検出できる）。
- 結果（リーチング）：スクリプト方策の成功率 100%。50 本・200 本ともリプレイ検証は全エピソード成功、関節角の差 0。
  データは `data/g1_reach_50`、`data/g1_reach_200`（git 管理外）。
- `lerobot-train` で状態のみ（画像なし）の ACT が学習できることも確認済み（CPU で 20 ステップのスモークテスト）。
  学習済みモデルは `ACTPolicy.from_pretrained` ＋ `make_pre_post_processors` で読み込み、`ReachEnv` の観測 dict をそのまま渡して
  `select_action` できる（CPU・MPS で確認）。エピソード開始ごとに `policy.reset()` が必要。
- フェーズ7への申し送り：ACT の既定は chunk_size = n_action_steps = 100 なので、150 ステップのエピソードでは推論が実質2回だけになる。
  観測の配線ミスを見つけやすくするため、評価では n_action_steps を小さくした条件でも試す。

```bash
.venv/bin/python -m g1_manip demo --episodes 20            # スクリプト方策の成功率（保存なし）
.venv/bin/mjpython -m g1_manip demo --view --episodes 3    # ビューアで表示
.venv/bin/python -m g1_manip generate --task reach --num-episodes 200 --overwrite  # 生成＋リプレイ検証
.venv/bin/python -m g1_manip generate --task reach --num-episodes 200 --replay-only  # リプレイ検証だけ
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
- キューブと目標の範囲が y=-0.05 で接しているので、初期配置に最小距離の制約を入れる。（ピック＆プレースでは範囲を変え、`PICK_MIN_DIST` で 10cm 以上離している）
- 机やランダム化範囲を変えたら、頭部カメラの視野（`HEAD_CAMERA_YAW_DEG` / `PITCH_DEG`）を再確認する。

## ピック＆プレース（フェーズ5）

リーチングでフェーズ5〜7を通した後、本来のタスク（机の上のキューブを右手でつかみ、目標マーカーの上に置く）のデモを作る。
**本物の接触把持**（weld などの疑似把持は使っていない）。

- **横からつかむ**（ユーザー決定。上からだと頭部カメラ（フェーズ8）で手がキューブを隠すため）：手のひらを立てて体の内側（+y 側）に向け、
  キューブの右側から法線方向に近づき、親指と中指・人差し指で「指の向き」に挟む。
  - Dex3 は人差し指と中指が手のひらの幅方向に並ぶので、手のひらを立てると指が上下に並び、主に下側の中指がキューブに当たる
    （親指＋中指の 2〜3 点で持つ。持ち上げるとキューブが少し傾く）。
  - 近づくときは親指を後ろへ逃がす（`HAND_OPEN` の thumb_1 = +0.4）。開いたままだと親指の先がキューブに当たって倒す。
  - 手のひらの中心はキューブの中心より 2.6cm 上（`GRASP_CUBE_IN_PALM`）。1.2cm では手首の下端と中指が天板に当たり、手が奥まで入らないことがあった（50 エピソードで成功率 92% → 100%）。
- IK は姿勢の 3 軸すべてにコストを付け（`RightHandIK(orientation_cost=(1, 1, 1))`）、**右腕と胴体・骨盤の衝突回避**（mink の `CollisionAvoidanceLimit`、
  `avoid_body_collision=True`）を入れた。入れないと、体の近くでは前腕が胴体に当たって手が目標から最大 7cm ずれたまま止まる。
- 手の向き（yaw）：指をキューブの面の法線に合わせたいので「キューブの yaw ＋ 90° の倍数 ＋ ずれ（0, ±10°, ±20°）」を候補にし、
  ずれが小さく、`GRASP_YAW_PREF_DEG`（−15°）に近い順に、腕が届く最初の候補を使う（`ScriptedPickPlacePolicy._choose_yaws`、
  リセット時に IK で経路を確かめる）。届く手の向きは位置によって 40° 前後の幅しかない（体の近くは大きめ、奥・右は小さめ）ため。
  ずれが 20° あっても把持できることを確かめた。候補は ±45° の範囲だけ（`GRASP_YAW_RANGE_DEG`）。これを超えると手のひらの法線が
  体の内側より前を向き、手がキューブの手前（頭部カメラの側）から近づくことになるため（制限する前は 200 本中 14 本がそうなっていた）。
  範囲内に届く候補が無い配置（試したシード 401 個中 2 個）はデモに入らない。置くときはキューブが手と一緒に回るだけなので、届く向きを `PLACE_YAW_CANDIDATES_DEG` から選ぶ。
  ただし運ぶ間に手首を回す角度は 45° まで（`PICK_MAX_YAW_CHANGE_DEG`）。60° 以上回すと手の中でキューブが上下にずれ、下ろすときに
  下側の中指が先に天板に当たって、キューブが浮いたまま落ちることがあった。
- 区間：approach → descend → insert（法線方向に 9cm 近づく）→ close → settle → lift（10cm）→ move（目標の上へ、手首を置く向きに回す）
  → lower → release → back → up（約 8 秒）。エピソードは 450 ステップ（9 秒）。置く位置は、持ち上げた後と下ろす前に手の中のキューブの
  位置を測って決める（閉じるときや手首を回すときにキューブが少しずれるため）。
- 右腕の目標角の 1 ステップの変化は 0.05rad（2.5rad/s）まで（`PICK_MAX_JOINT_STEP`）。超えたら全関節を同じ比率で縮め、IK の状態も縮めた値に合わせる。
  関節ごとに別々に切ると関節空間で動く向きが変わって手先が経路から最大 6.7cm 外れ、つかむ前に手首がキューブを押すことがあった。手先の経路が滑らかでも、
  IK の解が冗長な自由度（肘の位置など）や手首の特異姿勢の近くで急に組み替わり、1 ステップで最大 0.34rad 跳ぶことがあった
  （0.1rad を超えるエピソードが 200 本中 53 本）。模倣学習で観測から行動への写像が急に変わる所を作らないため。
- 保存の条件：成功に加えて、指を閉じる前（approach〜insert）に手がキューブに触れず、キューブが 2mm 以上動いていないこと
  （`ScriptedPickPlacePolicy.demo_ok`）。押して動かしたのにたまたま成功したエピソードは、把持の経路がキューブの位置と合っていないため。
- 範囲：キューブ x 0.22〜0.38m、y −0.23〜−0.07m、yaw ±30°（計画書の ±10cm から、横からの把持で届く範囲に狭めた）。
  目標 x 0.22〜0.38m、y −0.05〜+0.05m。キューブと目標は xy で 10cm 以上離す。
- 成功：最終ステップでキューブ中心が目標から xy で 3cm 以内、机の上に平らに置かれ（高さが天板＋半辺から ±1cm）、静止し、手が離れている。
- 保存形式はリーチングと同じキー。`observation.environment_state` は キューブの位置(3) ＋ 姿勢 quat(4, wxyz, w ≥ 0) ＋ 目標の位置(3, z は天板上面) の 10 次元。
  - 目標の z（一定）と親指の thumb_0（一定）は std が 0 なので、std の下限 0.01 が付く。
  - ACT の既定では environment_state は正規化されない（フェーズ7の項を参照）。キューブの quat や目標の位置の
    値の幅が小さいので、ピック＆プレースの学習では `normalization_mapping` に ENV を足した（フェーズ6の「ピック＆プレース」を参照）。
- 結果：データ生成（シード 0〜200）は試行 201 回で成功 200 本（成功率 99.5%。失敗の 1 本は届く手の向きの候補が無い配置）、
  別のシード 1000〜1199 では 199/200（失敗は同じ理由の 1 本）。腕の行動の 1 ステップの変化は全エピソードで 0.05rad 以下。
  `data/g1_pick_50`・`data/g1_pick_200`（約 9 万フレーム）のリプレイ検証は全エピソード成功、関節角・environment_state の差 0。
  リーチングのデータ（`g1_reach_50`）のリプレイ検証と、フェーズ4の到達テスト（20/20、最大 0.26cm）も変更後に再確認した。

```bash
.venv/bin/python -m g1_manip demo --task pick --episodes 20
.venv/bin/mjpython -m g1_manip demo --task pick --view --episodes 3
.venv/bin/python -m g1_manip generate --task pick --num-episodes 200   # 生成＋リプレイ検証
```

## ACT 学習（フェーズ6, Colab）

1. データセットを Hugging Face Hub の **private** リポジトリへ上げる（Mac）：

   ```bash
   .venv/bin/hf auth login                                                # write 権限のトークン
   .venv/bin/python -m g1_manip push --repo-id sanada331/g1_reach   # 既定で data/g1_reach_200 を上げる
   ```

   push 後に Hub 上で private になっていることを確認し、public なら異常終了する。
2. Colab で `notebooks/train_act_colab.ipynb` を開く（ランタイムは T4 GPU、鍵アイコンのシークレットに `HF_TOKEN` を登録）。
   W&B で可視化するなら `WANDB_API_KEY` も登録する（使わないなら手順 2 で `USE_WANDB = False`）。
   **トークンはノートブックに書かない**（このリポジトリは public）。
3. 上から順に実行する。学習の出力は Drive の `MyDrive/g1_act/runs/<RUN_NAME>/`、ログは `MyDrive/g1_act/logs/` に置く。
   切断されたら手順 1〜5 を実行してから手順 7（再開）。最後に手順 9 で `pretrained_model` を zip にしてダウンロードし、
   Mac の `outputs/<RUN_NAME>/pretrained_model/` に展開する。

- 学習は `lerobot-train` をラッパー（ノートブック内で `/content/train_act.py` に書き出す）経由で動かす：
  - Drive のマウント先ではシンボリックリンクを作れない場合があるので、`checkpoints/last` の作成に失敗しても止めず、
    `checkpoints/last_step.txt` に記録する。再開は `train_config.json` のパスを直接指定するので `last` は使わない。
  - ログの `step:10K` は丸められているので、正確なステップ数 `step_exact:<n>` を先頭に付ける（loss 曲線用）。
- Colab に元から入っている `diffusers` は、同じく元から入っている古い `torchao` と合わず import に失敗する
  （`cannot import name 'FqnToConfig' from 'torchao.quantization'`）。lerobot は `diffusers` があると GR00T などの他のポリシーの
  コードで import するので ACT の学習も起動しない。手順 3 で `diffusers` をアンインストールする（ACT は diffusers を使わない。
  Mac の環境にも無い）。
- W&B（`USE_WANDB = True`、project `g1_act`）：`wandb.disable_artifact=true` でチェックポイントは W&B に上げない。
  lerobot は W&B の run ID をチェックポイントの `train_config.json` に保存するので、手順 7 の再開でも同じ run に続けて記録される
  （`wandb/latest-run` のシンボリックリンクが Drive 上で作れなくても困らない）。ただし最後の保存から切断までの区間は、
  W&B では切断前の値が残る（step が戻るログは捨てられる）。
- W&B を有効にすると学習の開始時に `RUN_DIR/wandb` が作られるので、最初のチェックポイントより前に落ちたときは
  Drive の `RUN_DIR` を削除するか `RUN_NAME` を変えてから手順 6 をやり直す。
- 既定の設定：ACT の既定値（chunk_size = n_action_steps = 100、lr 1e-5、kl_weight 10）、50,000 ステップ、batch 64、
  5,000 ステップごとに保存、AMP（`policy.use_amp=true`。CUDA の autocast の既定は float16 なので T4 で動く）。
- 再開は、保存が最後まで終わっている最新のチェックポイントから行う。lerobot は `training_step.json` を先に、約 300MB の
  `optimizer_state.safetensors` と `optimizer_param_groups.json` を最後に書くので、ファイルの有無ではなく
  safetensors のヘッダとファイルサイズの一致・JSON が読めるかで判定する（保存の途中で切れたものは飛ばす）。
- チェックポイント 1 個は約 460MB（`pretrained_model` 154MB ＋ `training_state` 307MB）。既定の設定で Drive を約 4.6GB 使う。
- Mac 上で Colab をまねた環境（Drive を一時フォルダに、CUDA を CPU に置き換え、少ないステップ数）を作り、ノートブックを最初から最後まで
  実行して確認済み：Hub の private データセット取得、学習、`last` のリンク作成に失敗した場合の続行、保存途中のチェックポイントを
  飛ばした再開（optimizer の保存途中で切れた状態を模擬）、loss 曲線、zip 書き出し、書き出したモデル（CPU で学習したもの）の
  Mac での読み込みと `select_action`。GPU（T4・float16）での実行は未確認。
- W&B 対応は、ノートブックのラッパーを Mac で直接実行して確認した（ノートブック全体ではない）：オフラインモードで学習 →
  `wandb/latest-run` を消して途中のチェックポイントから再開 → 同じ run ID で続く。オンラインでの送信は未確認。
- フェーズ7への申し送り：Colab で学習したモデルの `policy_preprocessor.json` には `device: cuda` が入る。Mac で読み込むときは
  `make_pre_post_processors(..., preprocessor_overrides={"device_processor": {"device": "mps"}})` のように上書きが必要になる見込み。

### ピック＆プレース

```bash
.venv/bin/python -m g1_manip push --root data/g1_pick_200 --repo-id sanada331/g1_pick
```

Colab では `notebooks/train_act_pick_colab.ipynb` を使う（手順はリーチング版と同じ。学習の設定も同じで 50,000 ステップ、batch 64）。違いは次の 3 点：

- データセット `sanada331/g1_pick`（private、200 本・9 万フレーム、`environment_state` 10 次元）、`RUN_NAME = "act_g1_pick_v1"`。
- **`environment_state` も正規化する**：`--policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ENV":"MEAN_STD","ACTION":"MEAN_STD"}'`。
  lerobot 0.6.1 の ACT の既定の `normalization_mapping` には ENV が無く、`environment_state` はそのまま入る（リーチングのモデルはそうなっている）。
  ピック＆プレースでは値の幅（std 0.01〜0.17）に比べて平均（キューブの z 0.90、quat の w 0.98 など）が大きいので正規化する。
  Mac で短い学習を回し、保存した `config.json`・`train_config.json` に ENV が入り、前処理で `environment_state` が
  (x − 平均) / std になること、再開後も設定が残ることを確かめた。手順 9 でも書き出すモデルの設定を表示して確かめる。
- **データ読み込みの高速化**（ラッパーの `G1_FAST_DATA=1`、手順 2 で設定）：lerobot は 1 サンプルごとに HF datasets（Arrow）から
  現在の行と 100 ステップ分の行動を引くので遅く、リーチングの T4 での学習では 1 ステップ 0.54 秒のうち 0.40 秒がデータ待ち（`data_s`）だった。
  ラッパーは `DatasetReader.get_item` を差し替え、全列を最初に torch のテンソルへ読み込んで（9 万行で約 17MB、0.01 秒）同じ値を添字で取り出す
  （行動の範囲外はエピソード端の値で埋めて `action_is_pad` を立てる、元の実装と同じ処理）。
  - Mac で確かめたこと：`g1_pick_200` の全 9 万サンプル（`delta_timestamps` は ACT と同じ行動 100 ステップ）と `g1_reach_200` の
    約 3,600 サンプルで、元の実装と全キーの値・dtype・形が一致。1 サンプル 約 3.4ms → 約 0.04ms。CPU での 20 ステップの学習で、ログした 5 ステップごとの
    高速化あり・なしの loss が一致（同じシードで同じバッチになる）。
  - 動画・画像のあるデータセット、エピソードを絞り込んだ読み込み、画像変換があるとき、float32・int64・bool のスカラーか長さの揃った 1 次元リスト以外の列があるときは、自動で元の実装を使う。
  - T4 での速さは未確認。ログの `data_s` が `updt_s` よりずっと小さくなっていれば効いている。

## 評価（フェーズ7, Mac）

Colab の手順 9 で持ってきた `outputs/<RUN_NAME>/pretrained_model/` を `eval` コマンド（`g1_manip/commands/evaluate.py`）でシミュレーション上で動かし、成功率を出す。

```bash
.venv/bin/python -m g1_manip eval                                  # 既定: outputs/act_g1_reach_v1/pretrained_model、50 エピソード
.venv/bin/python -m g1_manip eval --task pick                      # ピック＆プレース（既定: outputs/act_g1_pick_v1/pretrained_model、data/g1_pick_200）
.venv/bin/python -m g1_manip eval --temporal-ensemble 0.01         # 毎ステップ予測して重ねて平均する（ACT 論文の係数）
.venv/bin/python -m g1_manip eval --n-action-steps 10              # チャンクのうち 10 個だけ実行して予測し直す（下の結果を参照）
.venv/bin/python -m g1_manip eval --episodes 5 --video outputs/eval_reach.mp4
.venv/bin/mjpython -m g1_manip eval --view --episodes 3
```

- 読み込みは `ACTPolicy.from_pretrained` ＋ `make_pre_post_processors`。Colab で学習したモデルは前処理の device が `cuda` なので、
  `preprocessor_overrides={"device_processor": {"device": ...}}` で推論するデバイスに上書きする。device は既定で MPS（無ければ CPU、`--device cpu` も可）。
- 50Hz で「観測 → 前処理（正規化）→ `select_action` → 後処理（逆正規化）→ 関節目標に適用 → 制御 1 周期」を繰り返す。
  エピソード開始ごとに `policy.reset()` で行動チャンクのキューを空にする。成功判定はデータ生成と同じ `is_success()`
  （リーチング：最終ステップで手のひらが目標点から 2cm 以内かつ静止。ピック＆プレース：キューブ中心が目標から xy で 3cm 以内、
  机の上に平らに置かれて静止し、手が離れている）。
- 評価のシードは `EVAL_SEED_BASE`（100000）以降。実行前に次を確かめる：
  - モデルの入力キー・次元が環境の観測と一致する（一致しなければ終了）
  - モデルの正規化統計が学習データ（既定 `data/g1_reach_200`、`--task pick` では `data/g1_pick_200`）の `meta/stats.json` と一致し、std の下限 0.01 が効いている
  - 評価のシードが学習データの生成に使ったシード（成功・失敗とも）と重ならない
  - 学習データの最初のエピソードと同じシードでリセットした観測が、保存した frame 0 と一致する（関節の並び順の照合。
    形と統計の照合だけでは並び順の取り違えを検出できないため。関節を 1 組入れ替えると NG になることを確認済み）。
    これが NG のときは環境側の配線が学習時と違い、評価結果に意味が無いので終了する
- 結果はモデルのフォルダの隣に `eval_<task>_<条件>_<N>ep.json` として保存する（エピソードごとの最終誤差・速さ・行動の跳び・推論時間）。
  「行動の跳び」は制御ステップ間の右腕の目標角の変化の最大で、チャンクの切り替わりでのカクつきの目安。
- 成功率がタスクごとの `EVAL_MIN_SUCCESS_RATES`（リーチング 90%、ピック＆プレース 70%：計画書の目安）以上なら終了コード 0、
  未満なら 1（入出力の不一致などで途中終了したときも 1）。
- `--task` ごとに、既定のモデル（`EVAL_POLICY_DIRS`）と照合に使う学習データ（`EVAL_DATASET_DIRS`）が切り替わる。
- **`act_g1_reach_v1` では environment_state は正規化されていない**（`act_g1_pick_v1` は ENV も MEAN_STD で正規化して学習した）：
  lerobot 0.6.1 の ACT の既定の `normalization_mapping` は VISUAL・STATE・ACTION だけで
  ENV が無いため、目標点の座標は生の値（m 単位、`g1_reach_200` での std は 2.4〜8.3cm）のまま入る。学習時も評価時も同じ処理なので食い違いではないが、
  成功率が低いときは、`--policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ENV":"MEAN_STD","ACTION":"MEAN_STD"}'`
  を付けて学習し直すことを検討する。
- 結果（`act_g1_reach_v1`：Colab の T4 で 40,000 ステップ学習したチェックポイント。50,000 ステップの予定が 41,062 ステップで
  ランタイムが切れたが、loss は 30K〜40K でほぼ横ばい（l1_loss 0.025→0.023）だったので 40,000 で評価した。評価のシード 100000〜100049）：

  | 推論の条件 | 成功率 | 最終誤差 平均 / 最大 | 行動の跳び 最大 | 推論 平均 |
  |---|---|---|---|---|
  | 既定（n_action_steps 100：2 秒ごとに予測） | **50/50 = 100%** | 0.42 / 1.55cm | 0.070rad | 1.5ms/ステップ |
  | temporal ensembling（係数 0.01、毎ステップ予測） | **50/50 = 100%** | 0.48 / 1.54cm | 0.031rad | 15.5ms/ステップ |
  | n_action_steps 10 | 43/50 = 86% | 2.28 / 22.38cm | 0.074rad | 2.3ms/ステップ |
  | n_action_steps 1 | 23/50 = 46% | 7.69 / 24.69cm | 0.124rad | 12.7ms/ステップ |

  - 既定の条件で計画書の目安（90%）を満たした。MPS と CPU で同じ結果（verifier が確認）。
  - temporal ensembling は行動の跳びが半分以下になる（チャンクの切り替わりのカクつきが減る）。
  - 予測し直す間隔を短くすると成功率が下がる。n_action_steps 10 の失敗（7 件）は最終ステップでもまだ動いていて（0.016〜0.031m/s）
    到着が遅れ、n_action_steps 1 の失敗（27 件）はほとんど（26 件）が目標の手前で止まっている（最終の速さ ≈ 0）。
    観測は関節角と目標点だけで速度も時刻も無いため、途中から予測し直すと「動き出し」の行動（ほぼ現在の姿勢）を出して
    遅れたり止まったりすると考えられる（推測）。temporal ensembling は過去のチャンクの予測も平均するので、この問題が起きない。
  - 既定の条件の 6 エピソード（18 秒）の録画：`outputs/act_g1_reach_v1/eval_reach_n100.mp4`（git 管理外）。
- ピック＆プレースの結果（`act_g1_pick_v1`：Colab で `sanada331/g1_pick`（200 本）から学習。`train_config.json` の steps は 50,000、
  ENV も MEAN_STD で正規化。評価のシード 100000〜100049、MPS）：

  | 推論の条件 | 成功率 | 置き誤差（xy）中央値 / 最大 | 失敗の内訳 | 行動の跳び 最大（成功のみ） | 推論 平均 |
  |---|---|---|---|---|---|
  | 既定（n_action_steps 100） | **43/50 = 86%** | 0.89 / 28.1cm | 持ち上げていない 4、置き位置のずれ 3 | 0.687rad（0.256rad） | 2.9ms/ステップ |
  | temporal ensembling（係数 0.01） | **42/50 = 84%** | 0.61 / 17.6cm | 持ち上げていない 8 | 0.078rad（0.078rad） | 14.0ms/ステップ |
  | n_action_steps 10 | 38/50 = 76% | 1.14 / 29.3cm | 持ち上げていない 9、置き位置のずれ 2、手が触れたまま 1 | 0.415rad（0.331rad） | 3.6ms/ステップ |

  - 3 条件とも計画書の目安（70%）を満たした。既定の条件の「置き位置のずれ」3 件は持ち上げが 4〜7cm で止まっていて（成功時は約 10cm）、
    途中で落としている。n_action_steps 10 の 2 件（シード 100010・100030）は 10cm 以上持ち上げたが、最終ステップでもキューブは
    手の中にあり（天板から 2.4・4.9cm 上）、目標まで運べていない。
  - 既定と temporal ensembling では、失敗はキューブが奥（x ≥ 0.34、範囲の上限は 0.38）のときに集中する。temporal ensembling では
    x < 0.34 が 39/39、x ≥ 0.34 が 3/11（既定は 37/39 と 6/11）。n_action_steps 10 は x < 0.34 でも 6 件失敗した（33/39 と 5/11）。
  - 指を閉じて settle が終わった時点（持ち上げ直前、180 ステップ目。方策もスクリプト方策と同じ時刻に把持を終える前提）の
    キューブ中心を手のひらサイト座標で見ると、temporal ensembling の失敗 8 件は同じシードのスクリプト方策より、
    手のひらの下向き（y）に +0.3〜+1.4cm、法線（z）に +0.2〜+2.0cm ずれている（差の絶対値の平均は 0.66cm・0.62cm、
    成功 42 件では 0.10cm・0.22cm）。8 件とも同じ向きで、手のひらが高く、手前で止まってキューブの上側をつかみ、
    持ち上げる途中で抜け落ちる（動画でも確認）。
    調べたスクリプト：`outputs/act_g1_pick_v1/grasp_diag.py`（git 管理外。リファクタリング前の `scripts/06_eval_policy.py` を読み込むので、今のままでは動かない）。
  - 既定の条件で 0.6rad を超える跳びは失敗したエピソードのチャンクの切り替わり（300・400 ステップ目）だけで出る（5 件）。
    つかみ損ねて学習データに無い状態になった後に予測し直した結果で、失敗の原因ではなく結果と考えられる（推測）。
  - 録画（git 管理外）：既定の条件の成功 3 件 `outputs/act_g1_pick_v1/eval_pick_n100_seed100000-100002.mp4`、
    失敗（シード 100024）`eval_pick_n100_seed100024_fail.mp4`・`eval_pick_te0.01_seed100024_fail.mp4`。
- 任意の配置での推論：`--cube X Y YAW_DEG`（キューブの位置 [m] と yaw [deg]）・`--target X Y`（置き場所 [m]）で配置を指定できる
  （`--task pick` のみ。全エピソード同じ配置で、ロボットの初期姿勢はシードで変わる。範囲・距離の制約は確かめない）。
  ピック＆プレースでは、各エピソードの初期配置に最も近い学習エピソードとの距離もログと JSON（`train_nn`）に出す。

  ```bash
  .venv/bin/python -m g1_manip eval --task pick --cube 0.30 -0.15 10 --target 0.28 0.0 --episodes 1 --video outputs/pick.mp4
  .venv/bin/mjpython -m g1_manip eval --task pick --cube 0.42 -0.20 0 --target 0.30 0.0 --episodes 1 --view
  ```

### GUI で配置して推論を試す（`gui` コマンド）

```bash
.venv/bin/python -m g1_manip gui                                   # 既定: outputs/act_g1_pick_v1/pretrained_model
.venv/bin/python -m g1_manip gui --policy-path outputs/<RUN_NAME>/pretrained_model
```

- 左の上面図で、赤い四角（キューブ）と緑の三角（置き場所）をドラッグで置く。キューブの向きはマウスホイールかスライダー（±45°）。
  印の大きさは学習データの点に合わせた小さなもの（実寸ではない。実物はキューブ 1 辺 4.5cm、シーンの目標の円盤は直径 8cm）。
  薄い点・三角は学習データの初期配置、破線は学習時の範囲、緑の点線の円は置き場所から 10cm（学習ではキューブは必ずこれより遠い）。
  配置を動かすたびに、学習データとの近さ（eval と同じ正規化距離）と、範囲外などの注意を表示する。
- 上面図の下の「実行するもの」で、方策（既定 / temporal ensembling 0.01 / 10 ステップごとに予測）・教師（スクリプト方策）・
  最近傍リプレイ（最も近い学習エピソードの行動列をそのまま再生）を選んで［実行］。ロボットの初期姿勢はシードで変わる。
- 右に実行中の映像（斜め上の視点と頭部カメラ）、上面図に手のひらの軌跡を描き、終わると操作欄の下に成功・失敗（内訳）・置き誤差を表示する。
  窓は 1136×726〜750px（注意や結果の文の行数で変わる。1440×900 の画面に収まる大きさ。結果の文は上面図の幅で折り返し、
  エラー文は 120 文字で切る）。
  「動画を保存」で斜め上の視点の mp4 を、結果は毎回 `outputs/act_g1_pick_v1/gui/runs.jsonl` に残す（git 管理外）。
- 方策に渡す観測は `env.observation()` のうちモデルの入力にあるキー（データ生成・学習と同じ形）。画像入力のモデル（フェーズ8）では、
  環境で 224×224 の頭部カメラ画像も描いて渡す（下の「画像入力」の節）。右下の頭部カメラの映像がその視点。
  モデルの読み込みは ACT 用で、入力の形の照合はしていない。入力が合わなければ実行時にエラーをステータス欄に出して止まる。
- 「配置を最初に戻す」でキューブと置き場所を起動時の位置に戻す。ドラッグできるのは上面図に見えている範囲だけ。
- 描画は `mujoco.Renderer`（オフスクリーン）なので `mjpython` は要らない。M1 では 1 枚約 50ms かかるので、「実時間で再生」は
  経過した実時間ぶんシミュレーションを進める（ロボットの動きは実際の速さ、映像の更新は 1 秒に 10 回弱。temporal ensembling は推論が
  重く 3 回程度）。9 秒のエピソードは約 9〜10 秒で終わる。外すと描画 1 回ごとに 10 ステップ進める（既定の方策で約 6 秒）。
  「動画を保存」は毎ステップ描画するので、1 エピソード約 30〜35 秒かかる（実測。高速モードの方が少し遅い）。
  結果はどちらでも同じ（シミュレーションは決定的）。
- 動作確認：`--selftest --screenshot <png>` でドラッグ操作をイベントで送ってキューブを (0.30, -0.15, 10°)、置き場所を (0.28, 0.00) に置き、
  既定の方策で 1 エピソード実行する。結果（成功、置き誤差 0.58cm、持ち上げ 10.0cm）は `eval --task pick --cube 0.30 -0.15 10
  --target 0.28 0.0` の シード 100000 と一致。`--screenshot` は上面図・映像・情報欄を 1 枚にした画像（ボタンなどは写らない）。

### 追加分析：学習データ・教師・リプレイとの比較（`analyze` コマンド）

「リプレイ（学習データの再生）ではなく、推論で配置に合わせて動けているか」を確かめる。出力は `outputs/act_g1_pick_v1/analysis/`
（git 管理外。`--out-dir` で変えられる）。

```bash
.venv/bin/python -m g1_manip analyze dist --show eval_pick_te0.01_50ep   # 初期配置の分布（eval の結果 JSON を読むだけ。数秒）
.venv/bin/python -m g1_manip analyze traj                                 # 方策・教師・最近傍リプレイ（50 シード、約 3 分）
.venv/bin/python -m g1_manip analyze traj --temporal-ensemble 0.01        # 約 7 分
.venv/bin/python -m g1_manip analyze grid --with-scripted --with-replay   # キューブの位置を格子状に振る（195 点、約 13 分）
.venv/bin/python -m g1_manip analyze traj --replot outputs/act_g1_pick_v1/analysis/traj_n100.json   # 図だけ描き直す
```

- 比べる相手：
  - **教師**：スクリプト方策を同じシード・同じ配置で動かしたもの（学習データの作り方と同じ。その配置での正解の軌跡）。
  - **最近傍リプレイ**：初期配置が最も近い学習エピソードの行動列（絶対目標角）を、観測を見ずにそのまま再生したもの。
    「方策が学習データを覚えて再生しているだけ」ならこれと同じくらいの結果になる。どのエピソードを再生するかは初期の観測だけで決める
    （正解の軌跡は見ない。方策は毎ステップ観測を見て行動を出す点が違う）。
  - 参考：学習データ 200 本のうち、把持の区間で手のひらが教師に最も近いもの（正解を見て選んだ上限で、方策にもリプレイにも選べない）。
- 「近さ」は、モデルが見るのと同じ正規化（`meta/stats.json` の mean/std）をした environment_state（キューブの位置・姿勢・置き場所）の
  距離で測る（`g1_manip/training_data.py`）。学習データは parquet を直接読む（LeRobotDataset の値と一致を確認）。
- 手のひらの位置は `observation.state` から順運動学で求める（シミュレーション中の `site_xpos` は mj_step の積分前の値で、
  1mm 未満ずれる）。時間の区間は教師の区間から決める：近づく区間 0〜2.6 秒（approach・descend・insert）、
  把持の区間 2.6〜3.6 秒（close・settle：手のひらが把持の位置でほぼ止まり、指を閉じる）。
- 図に matplotlib を使う（`requirements.txt` に追加）。

結果（`act_g1_pick_v1`、評価のシード 100000〜100049）：

- **最近傍リプレイとの比較（`traj_summary_<条件>.png`・`traj_examples_<条件>.png`）**：

  | | 方策 既定（n100） | 方策 temporal ensembling 0.01 | 最近傍リプレイ | 教師 |
  |---|---|---|---|---|
  | 成功 | **43/50** | **42/50** | 26/50 | 50/50 |
  | 把持の区間の、教師との手のひらのずれ（中央値） | **0.37cm** | **0.51cm** | 2.14cm | — |
  | 近づく区間の、教師との手のひらのずれ（中央値） | 1.48cm | 2.36cm | 2.46cm | — |
  | 把持の区間で最近傍リプレイより教師に近いシード | 50/50 | 49/50 | — | — |

  - 方策が成功して最近傍リプレイが失敗したシードは 17（既定・temporal ensembling とも）、逆は 0（既定）・1（temporal ensembling）。
    最近傍リプレイの失敗 24 件のうち 22 件は「置き位置のずれ」（つかんで、学習エピソードの置き場所に運んでいる）。
  - 方策は、配置から決まる正解（教師）の把持位置に 0.4〜0.5cm まで合わせている。最近傍の学習エピソードを再生すると 2.1cm ずれる
    （学習データのキューブは評価の配置から xy で中央値 0.6cm にあるが、置き場所・向きを含めて最も近いエピソードを選ぶと把持位置は 2cm 程度違う）。
  - 上限との比較：学習データ 200 本から把持の区間で教師に最も近いものを正解を見て選ぶと、ずれは中央値 0.52cm。方策（既定 0.37cm）は
    34/50 シード（temporal ensembling 0.51cm では 26/50）でこれより教師に近い。既定の条件では、学習データの軌跡のどれを選んでも
    届かない精度で配置に合わせて把持位置を出しているシードが多い（temporal ensembling ではほぼ半々）。
  - 3.6 秒以降のずれは、動き出す時刻のわずかな違いがそのまま位置の差になる（同じ時刻どうしで比べているため）。時間変化の図の山は
    この影響で、経路自体の違いより大きく見える。
  - 教師は 50 配置すべてで成功（届かない配置は無い）。
- **格子（`grid_n100_yaw+0_target+0.30+0.00.png`）**：キューブの yaw 0°、置き場所 (0.30, 0.00)、ロボットの初期姿勢はシード 100000 に固定し、
  キューブを x 0.16〜0.44・y -0.29〜-0.05 に 2cm 間隔で置いた 195 配置：

  | | 学習時のキューブの範囲の内（81） | 外（114） |
  |---|---|---|
  | 方策（既定） | **71** | **79** |
  | 最近傍リプレイ | 38 | 15 |
  | 教師（スクリプト方策） | 81 | 106 |

  - 「範囲の内」はキューブの xy だけで判定している。そのうち 12 配置は置き場所から 10cm 未満で、学習には無い配置（方策は 11/12 成功）。
  - 方策は学習時の範囲の外でも 7 割近く取れる（外挿）。最近傍リプレイは範囲の内でも 38/81（失敗 43 件はすべて「置き位置のずれ」）、
    外では 15/114。方策が成功してリプレイが失敗した配置は 102、逆は 5。
  - 方策の外での失敗 35 件の内訳は、手前（x ≤ 0.18）13 件、奥（x ≥ 0.40）の置き場所側（y ≥ -0.13）11 件、その他 11 件（最も奥の x 0.44 の列と、
    右手側の端 y -0.29 の角など）。教師が成功した 187 配置のうち方策が成功したのは 149。
  - 方策の範囲の内での失敗 10 件はすべて x ≥ 0.34、y -0.07〜-0.17（奥の、置き場所側）で、正規化距離は 0.55〜0.90（学習データの密度の範囲内）、
    教師は全部成功。50 シードの評価と同じく、奥で把持の精度が足りない。
- **分布（`dist.png`）**：評価の 50 配置はすべて学習データの密度の範囲内。最も近い学習エピソードとの正規化距離は中央値 0.79
  （学習エピソードどうしの最近傍は中央値 0.82、最大 1.41。1.41 を超える評価配置は 0 件）、キューブ xy だけなら最も近い学習配置まで
  中央値 0.6cm・最大 1.7cm。失敗の正規化距離は成功よりやや大きい（中央値 既定 0.87 / 0.77、temporal ensembling 0.83 / 0.76、
  n_action_steps 10 0.88 / 0.76）が、失敗も学習エピソードどうしの範囲（最大 1.41）の内で、学習データが無い配置で起きているのではない。
  temporal ensembling の失敗 8 件はすべてキューブが奥（x 0.359〜0.379）のとき。置き場所は 8 件中 7 件が y < 0（全体では 31/50）だが、
  8 件では偏りかどうか判断できない。
- 動作確認：上の実モデルでの評価のほか、Colab のモデルをまねて device を cuda にした試験用モデル（4 ステップだけ学習したもの）で、
  MPS・CPU での読み込みと実行、`--n-action-steps`・`--temporal-ensemble`・`--video`（H.264, 640×480, 50fps）、
  前処理の正規化と後処理の逆正規化が `meta/stats.json` からの手計算と一致すること、学習データが違うモデル（50 本で学習）で
  統計の照合が NG になることを確認した。`--view`（ビューア）は未確認。

## 画像入力（フェーズ8）

キューブ・目標の真値（`environment_state`）の代わりに、頭部カメラの RGB 画像（224×224、`observation.images.head`）を入力にする。
入力は関節角（14）＋画像、出力は関節目標角（14）。

### データ（`add-images` コマンド）

```bash
.venv/bin/python -m g1_manip add-images                  # data/g1_pick_200 → data/g1_pick_img_200（M1 で約 80 分、約 3.0GB）
.venv/bin/python -m g1_manip add-images --check-only     # 検証だけ
.venv/bin/python -m g1_manip push --root data/g1_pick_img_200 --repo-id sanada331/g1_pick_img
```

- 状態ベースのデモ `g1_pick_200` の各エピソードを、記録したシードで初期化して保存した action のまま再生し、各ステップの観測（行動を与える前）で
  頭部カメラを描く（`PickPlaceEnv(images=True)`）。動き・エピソードの並び・シードは状態ベースの学習データと同じなので、入力の違いだけを比べられる。
- 保存するのは `observation.state`・`observation.images.head`（PNG で parquet に埋め込む）・`action`。`environment_state` は入れない（計画書）。
  state と action は元の値をそのまま使い、再生した関節角・`environment_state` が元と一致しなければ止める。
- 検証：state・action が元と完全に一致、読み込んだ画像がその場で再生して描いた同じステップの画像と完全に一致（PNG なので差 0）し、
  隣のステップの画像とは違う、画像の正規化統計がある。
- 画像は直前の `mj_step` の状態をそのまま描く（`mj_forward` を呼ばない。シミュレーションの結果を変えないため）。評価でも同じ描き方をする。
- **手がキューブを隠す**：エピソード開始時（準備姿勢）は、頭部カメラから見ると右手がキューブを隠している。学習データ 40 本で数えると、
  t=0 でキューブが見えている割合は中央値 27%（30% 未満が 21 本、完全に隠れているのが 1 本）。0.6 秒後には手が視野の外へ動き、
  中央値 100%（最小 39%）。置き場所（緑の円盤）は t=0 で中央値 100%。運ぶ間（3.6〜6.6 秒）は手に持ったキューブの半分ほどが隠れる。
  ユーザー判断で、デモは作り直さずこのまま画像を付けた。推論では temporal ensembling など、こまめに予測し直す条件も比べる。

### 学習（`notebooks/train_act_pick_img_colab.ipynb`, Colab）

状態ベースのピック版のノートブックと手順・設定（50,000 ステップ、batch 64）は同じで、違いは次の点：

- データセット `sanada331/g1_pick_img`、`RUN_NAME = "act_g1_pick_img_v1"`。
- 画像は ImageNet で事前学習した ResNet18（ACT の既定）で特徴にする。画像の正規化は ImageNet の平均・標準偏差
  （lerobot-train の既定 `dataset.use_imagenet_stats=true`。`meta/stats.json` の画像の統計は使われない）。
- データ読み込みの高速化（`G1_FAST_DATA=1`）を画像に対応させた。画像の列は PNG のバイト列のまま（Arrow のメモリマップ）持ち、
  取り出すときに元の実装と同じく PIL で開いて `ToTensor` する。元の実装は行動 100 ステップ分を引くときに画像も読むので非常に遅い。
  - Mac で確かめたこと：8 本（3,600 サンプル）の試作データで、元の実装と全キーの値・dtype・形が一致。1 サンプル 約 131ms → 約 1.5ms。
    MPS での 4 ステップの学習で `data_s` 0.02 秒（`updt_s` 0.5〜0.8 秒、batch 8）。
- チェックポイント 1 個 約 0.6GB（`pretrained_model` 197MB ＋ `training_state` 394MB）。10 個で約 6GB なので Drive の空きに注意。
- T4 での速さは未計測（Mac の MPS では batch 64 で 1 ステップ約 4.6 秒）。最初の数百ステップの `updt_s + data_s` で所要時間を見積もる。

### 評価・GUI

```bash
.venv/bin/python -m g1_manip eval --task pick --policy-path outputs/act_g1_pick_img_v1/pretrained_model
.venv/bin/python -m g1_manip eval --task pick --policy-path outputs/act_g1_pick_img_v1/pretrained_model --temporal-ensemble 0.01
.venv/bin/python -m g1_manip gui --policy-path outputs/act_g1_pick_img_v1/pretrained_model
```

- モデルの入力に `observation.images.head` があれば、環境で頭部カメラを描いて渡す（`obs_to_batch` で学習データの読み込みと同じ
  float32 (3, 224, 224)、0〜1 にする）。`--dataset-root` を省略すると `data/g1_pick_img_200` と照合する。画像の正規化統計は ImageNet の値と比べる。
  学習データとの近さ（最近傍）は、`g1_generation.json` に記録した元の状態のデータセット（`g1_pick_200`）のうち、学習データに入っている
  エピソードで測る（GUI に画像入力のデータセットを `--dataset-root` で渡したときも同じ）。
- 動作確認（4 ステップだけ学習した試験用モデル。学習データは 8 本の試作データ `data/g1_pick_img_test`）：`--dataset-root data/g1_pick_img_test`
  を付けると照合がすべて OK（初期観測の画像も保存した frame 0 と差 0）、評価・GUI が最後まで動く。既定の `g1_pick_img_200` と照合すると、
  学習データが違うので正規化統計の照合が NG になる（取り違えを検出できる）。
  状態ベースのモデル（`act_g1_pick_v1`）の評価結果は変更前と同じ（シード 100000〜100001 の置き誤差・持ち上げ高さが一致）。
- 画像入力のモデルは 1 ステップごとに観測の画像を描くので、1 ステップ約 60ms かかり（画像なしは約 1ms）、実時間（20ms）に追いつけない。
  GUI の「実時間で再生」では描画 1 回の間に進めるステップ数を 10 までにしてあり、9 秒のエピソードが約 35 秒かかる
  （ほかに重い処理を動かしていないとき。描画の間隔は最長約 0.9 秒。上限が無いと 1 回の描画の間に残り全部を進めて窓が固まった）。状態入力のモデルは今までどおり約 9 秒。
- `analyze`（学習データ・教師・リプレイとの比較）は、モデルの入力に画像があれば環境で頭部カメラを描いて渡す（eval・gui と同じ）。
  1 ステップごとに画像を描くので、状態ベースのモデルより大幅に遅い。

## 実行上の注意（macOS）

`mujoco.viewer.launch_passive` は macOS では `python` ではなく **`mjpython`** で実行する必要がある。
ビューアを使うコマンド（`scene` の既定、`actuators`・`ik-test`・`demo`・`eval` の `--view`）はすべて次のように実行する：

```bash
.venv/bin/mjpython -m g1_manip scene
```
