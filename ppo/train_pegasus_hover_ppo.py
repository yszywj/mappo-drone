#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits
from envs.pegasus_ppo_env.config import SingleDroneEnvConfig
from envs.pegasus_ppo_env.single_drone_hover_env import SingleDroneHoverEnv
from ppo.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser("Single-drone Pegasus hover PPO")
    parser.add_argument("--num_env_steps", type=int, default=800)
    parser.add_argument("--rollout_steps", type=int, default=80)
    parser.add_argument("--episode_length", type=int, default=40)
    parser.add_argument("--step_dt_sim_sec", type=float, default=0.5)
    parser.add_argument("--ctbr_send_hz", type=int, default=20)
    parser.add_argument("--data_stream_hz", type=int, default=20)
    parser.add_argument("--takeoff_altitude", type=float, default=5.0)
    parser.add_argument("--connection_str", type=str, default="udp:0.0.0.0:14540")
    parser.add_argument("--target_system", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--init_action_std", type=float, default=0.05)
    parser.add_argument("--residual_gain", type=float, default=0.02)
    parser.add_argument("--pegasus_log_dir", type=str, default="./log_folder")
    parser.add_argument("--no_pegasus_log", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(args):
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


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    returns = np.zeros_like(rewards, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    return returns, advantages


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    run_dir = Path("./results") / "PegasusSingleDroneHover" / "ppo" / f"seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    policy = ActorCritic(env.obs_dim, env.action_dim, hidden_size=args.hidden_size, init_std=args.init_action_std).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print("=" * 80)
    print("Pegasus single-drone PPO residual hover training")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    print(f"num_env_steps: {args.num_env_steps}")
    print(f"rollout_steps: {args.rollout_steps}")
    print(f"episode_length: {args.episode_length}")
    print(f"residual_gain: {args.residual_gain}")
    print(f"init_action_std: {args.init_action_std}")
    print("=" * 80)

    obs, _ = env.reset()
    total_steps = 0
    update = 0
    try:
        while total_steps < args.num_env_steps:
            obs_buf = []
            action_buf = []
            logprob_buf = []
            reward_buf = []
            done_buf = []
            value_buf = []
            stats_reasons = Counter()
            stats_xy = []
            stats_rewards = []

            for _ in range(args.rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, logprob_t, value_t = policy.act(obs_t)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
                env_action = np.clip(action, -1.0, 1.0)
                next_obs, reward, done, info = env.step(env_action)

                obs_buf.append(obs.copy())
                action_buf.append(action.copy())
                logprob_buf.append(float(logprob_t.item()))
                reward_buf.append(float(reward))
                done_buf.append(bool(done))
                value_buf.append(float(value_t.item()))
                stats_rewards.append(float(reward))
                stats_reasons[info.get("done_reason", "running")] += 1
                if info.get("xy_err") is not None:
                    stats_xy.append(float(info["xy_err"]))

                obs = next_obs
                total_steps += 1
                if done:
                    obs, _ = env.reset()
                if total_steps >= args.num_env_steps:
                    break

            with torch.no_grad():
                last_value = policy.value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()

            rewards = np.asarray(reward_buf, dtype=np.float32)
            dones = np.asarray(done_buf, dtype=np.float32)
            values = np.asarray(value_buf, dtype=np.float32)
            returns, advantages = compute_gae(rewards, dones, values, last_value, args.gamma, args.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_tensor = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
            action_tensor = torch.as_tensor(np.asarray(action_buf), dtype=torch.float32, device=device)
            old_logprob_tensor = torch.as_tensor(np.asarray(logprob_buf), dtype=torch.float32, device=device)
            return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
            adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)

            batch_size = len(obs_buf)
            mini_batch_size = max(1, batch_size // args.num_mini_batch)
            pg_losses = []
            value_losses = []
            entropies = []

            for _ in range(args.ppo_epoch):
                indices = np.arange(batch_size)
                np.random.shuffle(indices)
                for start in range(0, batch_size, mini_batch_size):
                    mb = indices[start:start + mini_batch_size]
                    new_logprob, entropy, value = policy.evaluate_actions(obs_tensor[mb], action_tensor[mb])
                    ratio = torch.exp(new_logprob - old_logprob_tensor[mb])
                    surr1 = ratio * adv_tensor[mb]
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_tensor[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, return_tensor[mb])
                    entropy_loss = entropy.mean()
                    loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_loss.item()))

            update += 1
            torch.save(policy.state_dict(), model_dir / "actor_critic.pt")
            mean_reward = float(np.mean(stats_rewards)) if stats_rewards else 0.0
            mean_xy = float(np.mean(stats_xy)) if stats_xy else 0.0
            max_xy = float(np.max(stats_xy)) if stats_xy else 0.0
            print("=" * 80)
            print(f"[PegasusPPO] update={update}, steps={total_steps}/{args.num_env_steps}")
            print(f"  mean_rollout_reward: {mean_reward:.4f}")
            print(f"  mean_xy_err: {mean_xy:.3f}")
            print(f"  max_xy_err: {max_xy:.3f}")
            print(f"  done_reasons: {dict(stats_reasons)}")
            print(f"  policy_loss: {np.mean(pg_losses):.6f}")
            print(f"  value_loss: {np.mean(value_losses):.6f}")
            print(f"  entropy: {np.mean(entropies):.6f}")
    except KeyboardInterrupt:
        print("\n[PPO TRAIN] interrupted, closing environment...")
    finally:
        env.close()
        print("[PPO TRAIN] environment closed")


if __name__ == "__main__":
    main()
