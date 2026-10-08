"""アクチュエータの一覧表示と、右腕・右手の関節目標角への追従テスト。

    python scripts/02_check_actuators.py            # 一覧表示＋追従テスト（ビューアなし、合否判定）
    mjpython scripts/02_check_actuators.py --view   # 同じテストをビューアで表示しながら実行

テスト内容（制御は 50Hz、行動＝関節の目標角度（絶対値））:
  1. 右肘の目標角を正弦波で動かし、追従誤差を測る
  2. 右腕7関節・右手7関節すべてを目標姿勢へ min-jerk 補間で動かし、オーバーシュートと定常誤差を測る
     （その後ホームに戻す）。目標は14関節すべてで互いに異なり、ホームからも離れているので、
     アクチュエータの取り違えも検出できる
  3. 上記の間、制御対象外の左腕・左手が整定後のホーム姿勢から動かないことを確認する
     （脚・腰は関節ごと削除済みなので、関節が存在しないことを確認する）
テストのパラメータと合格しきい値は config.py（TEST_* / TRACK_* / IDLE_MAX_DEV）にある。
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_manip import config  # noqa: E402
from g1_manip.env import (  # noqa: E402
    apply_joint_targets,
    joint_actuator_ids,
    joint_qpos_ids,
    load_model,
    step_control,
)

DT = 1.0 / config.CONTROL_HZ


def actuator_type(model: mujoco.MjModel, i: int) -> str:
    gain, bias = model.actuator_gaintype[i], model.actuator_biastype[i]
    if model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE or gain != mujoco.mjtGain.mjGAIN_FIXED:
        return "other"
    if bias == mujoco.mjtBias.mjBIAS_NONE:
        return "motor"
    if bias == mujoco.mjtBias.mjBIAS_AFFINE:
        if model.actuator_biasprm[i, 1] != 0:
            return "position"
        if model.actuator_biasprm[i, 2] != 0:
            return "velocity"
    return "other"


def print_actuators(model: mujoco.MjModel) -> None:
    group = {**{j: "R-arm" for j in config.RIGHT_ARM_JOINTS}, **{j: "R-hand" for j in config.RIGHT_HAND_JOINTS},
             **{j: "L-arm" for j in config.LEFT_ARM_JOINTS}, **{j: "L-hand" for j in config.LEFT_HAND_JOINTS}}
    print(f"{'id':>2} {'actuator':<28} {'type':<8} {'joint':<28} {'group':<6} "
          f"{'ctrlrange [rad]':<18} {'kp':>5} {'kv':>6} {'force [Nm]':>10}")
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = model.joint(jid).name
        lo, hi = model.actuator_ctrlrange[i]
        kp = model.actuator_gainprm[i, 0]
        kv = -model.actuator_biasprm[i, 2]
        frc = model.jnt_actfrcrange[jid][1] if model.jnt_actfrclimited[jid] else np.inf
        print(f"{i:>2} {model.actuator(i).name:<28} {actuator_type(model, i):<8} {jname:<28} "
              f"{group.get(jname, '-'):<6} [{lo:+.3f}, {hi:+.3f}] {kp:>5.0f} {kv:>6.2f} {frc:>10.2f}")
    types = {actuator_type(model, i) for i in range(model.nu)}
    print(f"\n合計 {model.nu} 個、種類: {sorted(types)}")


def step_target(model: mujoco.MjModel, home: np.ndarray) -> np.ndarray:
    """ステップテストの目標（CONTROLLED_JOINTS の順）。

    右腕はホームから TEST_STEP_ARM_DELTA だけ動かす（腕を前に出すと手首が机の天板の下に引っかかるので、
    後ろ・外側へ動かす）。右手は各関節の可動域内の TEST_STEP_HAND_FRACTION の位置。
    """
    n_arm = len(config.RIGHT_ARM_JOINTS)
    target = home.copy()
    target[:n_arm] += np.asarray(config.TEST_STEP_ARM_DELTA)
    for k, name in enumerate(config.RIGHT_HAND_JOINTS):
        lo, hi = model.jnt_range[model.joint(name).id]
        target[n_arm + k] = lo + config.TEST_STEP_HAND_FRACTION[k] * (hi - lo)

    # 取り違え（アクチュエータの送り先の誤り）を必ず検出できる目標になっているか確認する
    min_from_home = np.abs(target - home).min()
    min_gap = min(abs(a - b) for a, b in itertools.combinations(target, 2))
    assert min_from_home > 2 * config.TRACK_STEP_SS_ERR and min_gap > 2 * config.TRACK_STEP_SS_ERR, (
        f"ステップ目標が取り違えを検出できない（ホームからの最小差 {min_from_home:.3f}, 関節間の最小差 {min_gap:.3f}）")
    return target


def run_tests(model, data, sync=None) -> bool:
    ctrl_ids = joint_actuator_ids(model, config.CONTROLLED_JOINTS)
    ctrl_q = joint_qpos_ids(model, config.CONTROLLED_JOINTS)
    idle_q = joint_qpos_ids(model, config.LEFT_ARM_JOINTS + config.LEFT_HAND_JOINTS)
    elbow = config.CONTROLLED_JOINTS.index("right_elbow_joint")
    # ホーム目標は config から作る（qpos の対応が誤っていれば、目標と実測のずれとして NG に出る）
    home = np.array([config.HOME_QPOS.get(j, 0.0) for j in config.CONTROLLED_JOINTS])

    def control_step(target):
        apply_joint_targets(model, data, ctrl_ids, target)
        step_control(model, data)
        if sync is not None:
            sync()

    # --- 0. ホームで整定（P 制御の重力による定常たわみ（数 mrad）を基準から除く）---
    for _ in range(int(config.TEST_SETTLE / DT)):
        control_step(home)
    idle_home = data.qpos[idle_q].copy()
    idle_dev = 0.0

    def tracked_step(target):
        nonlocal idle_dev
        control_step(target)
        idle_dev = max(idle_dev, np.abs(data.qpos[idle_q] - idle_home).max())

    # --- 1. 右肘の正弦波 ---
    sine_err = []
    for k in range(int(config.TEST_SINE_SECONDS / DT)):
        t = (k + 1) * DT
        target = home.copy()
        # 目標がホーム角から急に飛ばないよう、中心角とのずれを時定数 TEST_SINE_TAU で減衰させて正弦波に移る
        target[elbow] = (config.TEST_SINE_CENTER
                         + config.TEST_SINE_AMP * np.sin(2 * np.pi * config.TEST_SINE_HZ * t)
                         + (home[elbow] - config.TEST_SINE_CENTER) * np.exp(-t / config.TEST_SINE_TAU))
        tracked_step(target)
        if t > config.TEST_SINE_SKIP:
            sine_err.append(abs(data.qpos[ctrl_q[elbow]] - target[elbow]))
    sine_err = np.array(sine_err)

    # --- 2. ステップ目標（全14関節）→ ホームに戻す ---
    # 接触で止められると追従の評価にならないので、計測時のロボットと環境（机など）の接触も調べる。
    robot_root = model.body("pelvis").id

    def robot_env_contacts():
        names = []
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (model.body_rootid[b1] == robot_root) != (model.body_rootid[b2] == robot_root):
                names.append(f"{model.geom(c.geom1).name or model.body(b1).name}"
                             f"<->{model.geom(c.geom2).name or model.body(b2).name}")
        return sorted(set(names))

    # トルク飽和の計測用：各制御関節の dof と力の上限（重力補償分も含む qfrc_actuator と比べる）
    ctrl_dof = model.jnt_dofadr[model.actuator_trnid[ctrl_ids, 0]]
    frc_lim = np.where(model.jnt_actfrclimited[model.actuator_trnid[ctrl_ids, 0]].astype(bool),
                       model.jnt_actfrcrange[model.actuator_trnid[ctrl_ids, 0], 1], np.inf)

    step_results = []
    env_contacts = []
    n_ramp = int(config.TEST_STEP_RAMP / DT)
    start = home
    for label, target in (("step", step_target(model, home)), ("home", home)):
        peak_progress = np.zeros(len(target))
        n_sat = np.zeros(len(target))
        n_phys = 0
        for k in range(int(config.TEST_STEP_HOLD / DT)):
            s = min(1.0, (k + 1) / n_ramp)
            s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5  # min-jerk
            apply_joint_targets(model, data, ctrl_ids, start + s * (target - start))
            # step_control と同じだが、トルク飽和を物理ステップごとに数えるため1ステップずつ進める
            for _ in range(config.N_SUBSTEPS):
                mujoco.mj_step(model, data)
                n_sat += np.abs(data.qfrc_actuator[ctrl_dof]) >= frc_lim - 1e-6
                n_phys += 1
            if sync is not None:
                sync()
            idle_dev = max(idle_dev, np.abs(data.qpos[idle_q] - idle_home).max())
            peak_progress = np.maximum(peak_progress, (data.qpos[ctrl_q] - start) / (target - start))
        overshoot = np.clip(peak_progress - 1.0, 0.0, None)
        step_results.append((label, target, np.abs(data.qpos[ctrl_q] - target), overshoot, n_sat / n_phys))
        env_contacts += robot_env_contacts()
        start = target

    # --- 結果 ---
    print(f"\n[1] 右肘 正弦波: 中心 {config.TEST_SINE_CENTER} rad, 振幅 {config.TEST_SINE_AMP} rad, "
          f"{config.TEST_SINE_HZ} Hz, {config.TEST_SINE_SECONDS} 秒")
    print(f"    追従誤差（{config.TEST_SINE_SKIP} 秒以降）: 最大 {sine_err.max():.4f} rad, "
          f"RMS {np.sqrt((sine_err ** 2).mean()):.4f} rad")
    print(f"\n[2] 目標姿勢への移動（min-jerk {config.TEST_STEP_RAMP} 秒＋保持、計 {config.TEST_STEP_HOLD} 秒）"
          f"の誤差 [rad] とオーバーシュート [移動量比 %]")
    print(f"    {'joint':<28} {'ホーム':>8} {'目標':>8} {'誤差':>8} {'行過ぎ%':>7} {'飽和%':>6} "
          f"{'戻り誤差':>8} {'行過ぎ%':>7} {'飽和%':>6}")
    for k, name in enumerate(config.CONTROLLED_JOINTS):
        (_, tgt, err_s, ov_s, sat_s), (_, _, err_h, ov_h, sat_h) = step_results
        print(f"    {name:<28} {home[k]:>+8.3f} {tgt[k]:>+8.3f} {err_s[k]:>8.4f} {ov_s[k] * 100:>7.1f} "
              f"{sat_s[k] * 100:>6.1f} {err_h[k]:>8.4f} {ov_h[k] * 100:>7.1f} {sat_h[k] * 100:>6.1f}")
    step_max = max(r[2].max() for r in step_results)
    overshoot_max = max(r[3].max() for r in step_results)
    # 飽和の合否は指だけで判定する（指の低ゲイン化の回帰検出）。腕は kp=500 のままなので、
    # 手首（上限 5Nm）は 50Hz の目標更新直後に数 ms 飽和することがあり、表に参考として出すだけにする。
    n_arm = len(config.RIGHT_ARM_JOINTS)
    saturation_max = max(r[4][n_arm:].max() for r in step_results)
    removed_present = [j for j in config.REMOVED_JOINTS
                       if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) != -1]
    print(f"\n[3] 制御対象外: 左腕・左手の、整定後のホーム姿勢からの最大ずれ {idle_dev:.5f} rad"
          f"（脚・腰の関節: {'存在する ' + str(removed_present) if removed_present else 'なし（削除済み＝固定）'}）")

    checks = {
        f"右肘が正弦波に追従 (最大誤差 < {config.TRACK_SINE_MAX_ERR} rad)": sine_err.max() < config.TRACK_SINE_MAX_ERR,
        f"右腕・右手が目標姿勢に追従 (定常誤差 < {config.TRACK_STEP_SS_ERR} rad)":
            step_max < config.TRACK_STEP_SS_ERR,
        f"目標姿勢への移動のオーバーシュートが小さい (< {config.TRACK_OVERSHOOT_MAX * 100:.0f}%)":
            overshoot_max < config.TRACK_OVERSHOOT_MAX,
        f"移動中に指がトルク上限へ張り付かない (時間割合 < {config.TRACK_SATURATION_MAX * 100:.0f}%)":
            saturation_max < config.TRACK_SATURATION_MAX,
        f"計測時にロボットが環境と接触していない {env_contacts if env_contacts else ''}": not env_contacts,
        f"左腕・左手が静止 (ずれ < {config.IDLE_MAX_DEV} rad)": idle_dev < config.IDLE_MAX_DEV,
        "脚・腰の関節が無い（ワールドに固定）": not removed_present,
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'NG'}] {name}")
    return all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", action="store_true", help="ビューアで表示しながら実行（mjpython で実行すること）")
    args = parser.parse_args()

    model, data = load_model()
    print_actuators(model)

    if not args.view:
        sys.exit(0 if run_tests(model, data) else 1)

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        def sync():
            if not viewer.is_running():
                sys.exit("ビューアが閉じられたため、テストを中断しました。")
            t0 = time.time()
            viewer.sync()
            time.sleep(max(0.0, DT - (time.time() - t0)))

        ok = run_tests(model, data, sync=sync)
        print("\nテスト終了。ビューアを閉じると終了します。")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
