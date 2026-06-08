#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits
from envs.pegasus_ppo_env.config import SingleDroneEnvConfig
from envs.pegasus_ppo_env.single_drone_hover_env import SingleDroneHoverEnv
from ppo.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a single-drone Pegasus hover PPO checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode_length", type=int, default=80)
    parser.add_argument("--step_dt_sim_sec", type=float, default=0.5)
    parser.add_argument("--ctbr_send_hz", type=int, default=20)
    parser.add_argument("--data_stream_hz", type=int, default=20)
    parser.add_argument("--takeoff_altitude", type=float, default=5.0)
    parser.add_argument("--connection_str", type=str, default="udp:0.0.0.0:14540")
    parser.add_argument("--target_system", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--init_action_std", type=float, default=0.05)
    parser.add_argument("--residual_gain", type=float, default=0.05)
    parser.add_argument("--pegasus_log_dir", type=str, default="./log_folder")
    parser.add_argument("--no_pegasus_log", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(args) -> SingleDroneHoverEnv:
    action_limits = CTBRActionLimits(
        max_roll_rate=0.080,
        max_pitch_rate=0.080,
        max_yaw_rate=0.010,
        hover_thrust=0.60,
        thrust_delta=0.015,
        thrust_min=0.50,
        thrust_max=0.72,
        residual_gain=args.residual_gain,
    )
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
    cfg = SingleDroneEnvConfig(
        connection_str=args.connection_str,
        target_system=args.target_system,
        data_stream_hz=args.data_stream_hz,
        ctbr_send_hz=args.ctbr_send_hz,
        step_dt_sim_sec=args.step_dt_sim_sec,
        episode_length=args.episode_length,
        takeoff_altitude=args.takeoff_altitude,
        stabilize_after_takeoff_sim_sec=5.0,
        recover_timeout_sim_sec=25.0,
        recover_tolerance_m=0.5,
        start_logging=not args.no_pegasus_log,
        log_dir=args.pegasus_log_dir,
        reward_alive=0.05,
        reward_control_scale=0.05,
        reward_crash=-30.0,
        reward_timeout=2.0,
        action_limits=action_limits,
        safety_limits=safety_limits,
    )
    return SingleDroneHoverEnv(cfg, seed=args.seed)


def load_policy(args, obs_dim: int, action_dim: int, device: torch.device) -> ActorCritic:
    policy = ActorCritic(
        obs_dim,
        action_dim,
        hidden_size=args.hidden_size,
        init_std=args.init_action_std,
    ).to(device)
    checkpoint = Path(args.checkpoint).expanduser()
    state_dict = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def actor_mean_action(policy: ActorCritic, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy.distribution(obs_t).mean
    return np.clip(action.squeeze(0).cpu().numpy().astype(np.float32), -1.0, 1.0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    env = make_env(args)
    policy = load_policy(args, env.obs_dim, env.action_dim, device)

    print("=" * 80)
    print("Pegasus single-drone PPO deterministic evaluation")
    print(f"checkpoint: {Path(args.checkpoint).expanduser()}")
    print(f"device: {device}")
    print(f"episodes: {args.episodes}")
    print(f"episode_length: {args.episode_length}")
    print(f"residual_gain: {args.residual_gain}")
    print("=" * 80)

    episode_summaries = []
    done_reasons = Counter()
    try:
        for episode_idx in range(1, args.episodes + 1):
            obs, _ = env.reset()
            rewards = []
            xy_errs = []
            z_errs = []
            speed_xys = []
            final_reason = "not_done"

            for _ in range(args.episode_length):
                action = actor_mean_action(policy, obs, device)
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                if info.get("xy_err") is not None:
                    xy_errs.append(float(info["xy_err"]))
                if info.get("z_err") is not None:
                    z_errs.append(float(info["z_err"]))
                if info.get("speed_xy") is not None:
                    speed_xys.append(float(info["speed_xy"]))

                if done:
                    final_reason = str(info.get("done_reason", "unknown"))
                    break

            done_reasons[final_reason] += 1
            summary = {
                "episode": episode_idx,
                "steps": len(rewards),
                "return": float(np.sum(rewards)) if rewards else 0.0,
                "mean_xy_err": float(np.mean(xy_errs)) if xy_errs else 0.0,
                "max_xy_err": float(np.max(xy_errs)) if xy_errs else 0.0,
                "mean_z_err": float(np.mean(z_errs)) if z_errs else 0.0,
                "max_z_err": float(np.max(z_errs)) if z_errs else 0.0,
                "max_speed_xy": float(np.max(speed_xys)) if speed_xys else 0.0,
                "done_reason": final_reason,
            }
            episode_summaries.append(summary)
            print(
                "[PPO EVAL] "
                f"episode={summary['episode']}, steps={summary['steps']}, "
                f"return={summary['return']:.3f}, "
                f"mean_xy_err={summary['mean_xy_err']:.3f}, "
                f"max_xy_err={summary['max_xy_err']:.3f}, "
                f"mean_z_err={summary['mean_z_err']:.3f}, "
                f"max_z_err={summary['max_z_err']:.3f}, "
                f"max_speed_xy={summary['max_speed_xy']:.3f}, "
                f"done_reason={summary['done_reason']}"
            )

        mean_xy = float(np.mean([s["mean_xy_err"] for s in episode_summaries])) if episode_summaries else 0.0
        max_xy = float(np.max([s["max_xy_err"] for s in episode_summaries])) if episode_summaries else 0.0
        mean_z = float(np.mean([s["mean_z_err"] for s in episode_summaries])) if episode_summaries else 0.0
        max_z = float(np.max([s["max_z_err"] for s in episode_summaries])) if episode_summaries else 0.0
        print("=" * 80)
        print("[PPO EVAL SUMMARY]")
        print(f"  done_reasons: {dict(done_reasons)}")
        print(f"  mean_episode_xy_err: {mean_xy:.3f}")
        print(f"  max_episode_xy_err: {max_xy:.3f}")
        print(f"  mean_episode_z_err: {mean_z:.3f}")
        print(f"  max_episode_z_err: {max_z:.3f}")
    except KeyboardInterrupt:
        print("\n[PPO EVAL] interrupted, closing environment...")
    finally:
        env.close()
        print("[PPO EVAL] environment closed")


if __name__ == "__main__":
    main()
