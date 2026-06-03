#!/usr/bin/env python3
"""
First MAPPO training task for mappo-drone: conservative safe hover.

Run from repo root while Pegasus/PX4 is already running:
    python train_pegasus_hover_mappo.py \
        --num_env_steps 800 \
        --episode_length 40 \
        --step_dt_sim_sec 0.5

Important:
- This script uses only one rollout thread because Pegasus/PX4 is a real-time simulator.
- It does first takeoff once, then each episode recovers to home through your env reset().
- If an unrecoverable event happens, it raises and stops training so you can restart Pegasus/PX4.
"""

from __future__ import annotations

import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def ensure_onpolicy_alias():
    """
    Your repo folder is named `mappo/`, but the original MAPPO code imports `onpolicy.*`.
    This runtime alias lets the existing code work without renaming folders.
    """
    repo_root = Path(__file__).resolve().parent
    mappo_dir = repo_root / "mappo"
    if "onpolicy" not in sys.modules:
        pkg = types.ModuleType("onpolicy")
        pkg.__path__ = [str(mappo_dir)]
        sys.modules["onpolicy"] = pkg
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


ensure_onpolicy_alias()

from onpolicy.config import get_config
from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits
from envs.pegasus_mappo_env.config import TwoDroneEnvConfig
from envs.pegasus_mappo_env.safe_hover_env import SafeHoverTwoDroneEnv
from envs.pegasus_mappo_env.mappo_vec_env import PegasusSingleVecEnv
from onpolicy.runner.shared.pegasus_hover_runner import PegasusHoverRunner


def parse_args():
    parser = get_config()

    # Environment-specific args.
    parser.add_argument("--step_dt_sim_sec", type=float, default=0.5)
    parser.add_argument("--ctbr_send_hz", type=int, default=20)
    parser.add_argument("--data_stream_hz", type=int, default=20)
    parser.add_argument("--takeoff_altitude_1", type=float, default=5.0)
    parser.add_argument("--takeoff_altitude_2", type=float, default=9.0)
    parser.add_argument("--init_action_std", type=float, default=0.001)
    parser.add_argument("--pegasus_log_dir", type=str, default="./log_folder")
    parser.add_argument("--no_pegasus_log", action="store_true", default=False)

    all_args = parser.parse_args()

    # Safe defaults for real-time Pegasus/PX4 training.
    all_args.env_name = "PegasusTwoDroneSafeHover"
    all_args.algorithm_name = "mappo"
    all_args.experiment_name = all_args.experiment_name or "safe_hover"

    # Real simulator: exactly one rollout env.
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.n_render_rollout_threads = 1

    # Disable wandb in this first local training script.
    all_args.use_wandb = False
    all_args.use_eval = False

    # Feed-forward policy is simpler and safer for first integration.
    all_args.use_recurrent_policy = False
    all_args.use_naive_recurrent_policy = False

    # Conservative PPO defaults for slow real-time rollouts.
    all_args.ppo_epoch = min(all_args.ppo_epoch, 5)
    all_args.num_mini_batch = 1
    all_args.lr = min(all_args.lr, 3e-4)
    all_args.critic_lr = min(all_args.critic_lr, 3e-4)
    all_args.entropy_coef = min(all_args.entropy_coef, 0.005)
    all_args.hidden_size = min(all_args.hidden_size, 64)

    return all_args


def set_seed(seed: int, cuda_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_env(all_args):
    # Safe-hover 第一阶段：极保守动作范围。
    # 目标不是快速学会飞行，而是先保证不坠毁、不撞机、不触发 PX4 failsafe。
    action_limits = CTBRActionLimits(
        max_roll_rate=0.080,
        max_pitch_rate=0.080,
        max_yaw_rate=0.010,

        hover_thrust=0.60,
        thrust_delta=0.015,
        thrust_min=0.50,
        thrust_max=0.72,
    )

    # Safe-hover 第一阶段：收紧安全边界，避免飞远后 recover 很久。
    safety_limits = SafetyLimits(
        min_altitude=0.35,
        max_altitude=11.0,
        max_tilt_deg=55.0,
        max_body_rate=4.0,
        max_down_speed=3.0,

        max_xy_from_home=5.5,
        max_z_error_from_home=4.0,

        stale_wall_time_sec=2.0,
    )

    env_cfg = TwoDroneEnvConfig(
        data_stream_hz=all_args.data_stream_hz,
        ctbr_send_hz=all_args.ctbr_send_hz,
        step_dt_sim_sec=all_args.step_dt_sim_sec,
        episode_length=all_args.episode_length,

        # 第一阶段建议两机高度差拉大，降低随机探索时的碰撞概率。
        takeoff_altitudes=(all_args.takeoff_altitude_1, all_args.takeoff_altitude_2),

        stabilize_after_takeoff_sim_sec=5.0,
        recover_timeout_sim_sec=25.0,
        recover_tolerance_m=0.5,

        auto_takeoff_on_first_reset=True,
        start_logging=not all_args.no_pegasus_log,
        log_dir=all_args.pegasus_log_dir,

        # Hover curriculum:
        # SafeHoverTwoDroneEnv 会把目标覆盖成各自 home 点。
        goal_xy_radius_min=0.0,
        goal_xy_radius_max=0.0,
        goal_z_delta_max=0.0,

        # 第一阶段保守避碰。
        collision_distance_m=1.0,
        warning_distance_m=2.5,

        # Safe-hover 阶段：
        # timeout 表示安全活到 episode 结束，所以给正奖励。
        reward_alive=0.05,
        reward_control_scale=0.05,
        reward_close_penalty_scale=0.5,
        reward_collision=-30.0,
        reward_crash=-30.0,
        reward_timeout=2.0,

        action_limits=action_limits,
        safety_limits=safety_limits,
    )

    return PegasusSingleVecEnv(
        lambda: SafeHoverTwoDroneEnv(env_cfg, seed=all_args.seed),
        auto_reset=True,
        stop_on_unrecoverable=True,
    )


def main():
    all_args = parse_args()
    set_seed(all_args.seed, all_args.cuda_deterministic)

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        Path("./results")
        / all_args.env_name
        / all_args.algorithm_name
        / all_args.experiment_name
        / f"seed{all_args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    envs = make_env(all_args)

    print("=" * 80)
    print("Pegasus MAPPO safe-hover training")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    print(f"episode_length: {all_args.episode_length}")
    print(f"step_dt_sim_sec: {all_args.step_dt_sim_sec}")
    print(f"num_env_steps: {int(all_args.num_env_steps)}")
    print(f"init_action_std: {all_args.init_action_std}")
    print("=" * 80)

    try:
        runner = PegasusHoverRunner(all_args, envs, device, run_dir)
        runner.run()
    except KeyboardInterrupt:
        print("\n[TRAIN] 收到 Ctrl+C，正在安全关闭环境...")
    finally:
        envs.close()
        print("[TRAIN] 环境已关闭")


if __name__ == "__main__":
    main()
