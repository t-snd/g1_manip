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
.venv/bin/python scripts/04_scripted_demo.py --episodes 20            # スクリプト方策の成功率（保存なし）
.venv/bin/mjpython scripts/04_scripted_demo.py --view --episodes 3    # ビューアで表示
.venv/bin/python scripts/05_generate_dataset.py --task reach --num-episodes 200 --overwrite  # 生成＋リプレイ検証
.venv/bin/python scripts/05_generate_dataset.py --task reach --num-episodes 200 --replay-only  # リプレイ検証だけ
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

## ACT 学習（フェーズ6, Colab）

1. データセットを Hugging Face Hub の **private** リポジトリへ上げる（Mac）：

   ```bash
   .venv/bin/hf auth login                                                # write 権限のトークン
   .venv/bin/python scripts/05b_push_dataset.py --repo-id sanada331/g1_reach   # 既定で data/g1_reach_200 を上げる
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

## 評価（フェーズ7, Mac）

Colab の手順 9 で持ってきた `outputs/<RUN_NAME>/pretrained_model/` を `scripts/06_eval_policy.py` でシミュレーション上で動かし、成功率を出す。

```bash
.venv/bin/python scripts/06_eval_policy.py                                  # 既定: outputs/act_g1_reach_v1/pretrained_model、50 エピソード
.venv/bin/python scripts/06_eval_policy.py --temporal-ensemble 0.01         # 毎ステップ予測して重ねて平均する（ACT 論文の係数）
.venv/bin/python scripts/06_eval_policy.py --n-action-steps 10              # チャンクのうち 10 個だけ実行して予測し直す（下の結果を参照）
.venv/bin/python scripts/06_eval_policy.py --episodes 5 --video outputs/eval_reach.mp4
.venv/bin/mjpython scripts/06_eval_policy.py --view --episodes 3
```

- 読み込みは `ACTPolicy.from_pretrained` ＋ `make_pre_post_processors`。Colab で学習したモデルは前処理の device が `cuda` なので、
  `preprocessor_overrides={"device_processor": {"device": ...}}` で推論するデバイスに上書きする。device は既定で MPS（無ければ CPU、`--device cpu` も可）。
- 50Hz で「観測 → 前処理（正規化）→ `select_action` → 後処理（逆正規化）→ 関節目標に適用 → 制御 1 周期」を繰り返す。
  エピソード開始ごとに `policy.reset()` で行動チャンクのキューを空にする。成功判定はデータ生成と同じ `ReachEnv.is_success()`
  （最終ステップで手のひらが目標点から 2cm 以内かつ静止）。
- 評価のシードは `EVAL_SEED_BASE`（100000）以降。実行前に次を確かめる：
  - モデルの入力キー・次元が環境の観測と一致する（一致しなければ終了）
  - モデルの正規化統計が学習データ（既定 `data/g1_reach_200`）の `meta/stats.json` と一致し、std の下限 0.01 が効いている
  - 評価のシードが学習データの生成に使ったシード（成功・失敗とも）と重ならない
  - 学習データの最初のエピソードと同じシードでリセットした観測が、保存した frame 0 と一致する（関節の並び順の照合。
    形と統計の照合だけでは並び順の取り違えを検出できないため。関節を 1 組入れ替えると NG になることを確認済み）。
    これが NG のときは環境側の配線が学習時と違い、評価結果に意味が無いので終了する
- 結果はモデルのフォルダの隣に `eval_<task>_<条件>_<N>ep.json` として保存する（エピソードごとの最終誤差・速さ・行動の跳び・推論時間）。
  「行動の跳び」は制御ステップ間の右腕の目標角の変化の最大で、チャンクの切り替わりでのカクつきの目安。
- 成功率が `EVAL_MIN_SUCCESS_RATE`（90%）以上なら終了コード 0、未満なら 1（入出力の不一致などで途中終了したときも 1）。
- **environment_state は正規化されていない**：lerobot 0.6.1 の ACT の既定の `normalization_mapping` は VISUAL・STATE・ACTION だけで
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
- 動作確認：上の実モデルでの評価のほか、Colab のモデルをまねて device を cuda にした試験用モデル（4 ステップだけ学習したもの）で、
  MPS・CPU での読み込みと実行、`--n-action-steps`・`--temporal-ensemble`・`--video`（H.264, 640×480, 50fps）、
  前処理の正規化と後処理の逆正規化が `meta/stats.json` からの手計算と一致すること、学習データが違うモデル（50 本で学習）で
  統計の照合が NG になることを確認した。`--view`（ビューア）は未確認。

## 実行上の注意（macOS）

`mujoco.viewer.launch_passive` は macOS では `python` ではなく **`mjpython`** で実行する必要がある。
ビューアを使うスクリプトはすべて次のように実行する：

```bash
.venv/bin/mjpython scripts/01_view_scene.py
```
