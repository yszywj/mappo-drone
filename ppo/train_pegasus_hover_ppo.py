#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits
from envs.pegasus_ppo_env.config import SingleDroneEnvConfig
from envs.pegasus_ppo_env.single_drone_hover_env import SingleDroneHoverEnv
from ppo.actor_critic import ActorCritic


UPDATE_METRIC_FIELDS = [
    "update",
    "total_steps",
    "mean_rollout_reward",
    "mean_xy_err",
    "max_xy_err",
    "mean_z_err",
    "max_z_err",
    "mean_signed_z_err",
    "mean_goal_xy_progress",
    "mean_speed_xy",
    "max_speed_xy",
    "mean_speed_z",
    "max_speed_z",
    "success_count",
    "timeout_count",
    "other_done_count",
    "done_reasons",
    "policy_loss",
    "value_loss",
    "entropy",
]

EPISODE_METRIC_FIELDS = [
    "episode",
    "total_steps",
    "episode_steps",
    "return",
    "done_reason",
    "final_goal_xy_err",
    "final_z_err",
    "final_goal_distance",
    "final_goal_rel_x",
    "final_goal_rel_y",
    "final_goal_rel_z",
    "final_signed_z_err",
    "final_speed_xy",
    "final_speed_z",
    "final_cmd_roll_rate",
    "final_cmd_pitch_rate",
    "final_cmd_yaw_rate",
    "final_cmd_thrust",
    "mean_goal_xy_err",
    "max_goal_xy_err",
    "mean_z_err",
    "max_z_err",
    "success",
]


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
    parser.add_argument("--residual_gain", type=float, default=0.8)
    parser.add_argument("--goal_feedback_scale", type=float, default=0.0)
    parser.add_argument("--attitude_feedback_scale", type=float, default=1.0)
    parser.add_argument("--hover_thrust", type=float, default=0.60)
    parser.add_argument("--thrust_delta", type=float, default=0.015)
    parser.add_argument("--thrust_min", type=float, default=0.50)
    parser.add_argument("--thrust_max", type=float, default=0.72)
    parser.add_argument("--z_pos_gain", type=float, default=0.08)
    parser.add_argument("--z_vel_gain", type=float, default=0.025)
    parser.add_argument("--recover_z_tolerance_m", type=float, default=0.25)
    parser.add_argument("--goal_xy_radius_min", type=float, default=0.0)
    parser.add_argument("--goal_xy_radius_max", type=float, default=0.0)
    parser.add_argument("--goal_z_delta_max", type=float, default=0.0)
    parser.add_argument("--goal_tolerance_m", type=float, default=0.25)
    parser.add_argument("--goal_z_tolerance_m", type=float, default=0.35)
    parser.add_argument("--goal_speed_xy_tolerance_mps", type=float, default=0.25)
    parser.add_argument("--goal_speed_z_tolerance_mps", type=float, default=0.25)
    parser.add_argument("--reward_alive", type=float, default=0.0)
    parser.add_argument("--reward_progress_scale", type=float, default=3.0)
    parser.add_argument("--reward_distance_scale", type=float, default=0.10)
    parser.add_argument("--reward_z_scale", type=float, default=0.20)
    parser.add_argument("--reward_success", type=float, default=8.0)
    parser.add_argument("--reward_timeout", type=float, default=0.0)
    parser.add_argument("--pegasus_log_dir", type=str, default="./log_folder")
    parser.add_argument("--no_pegasus_log", action="store_true", default=False)
    parser.add_argument("--no_terminal_log", action="store_true", default=False)
    return parser.parse_args()


def append_csv_row(path: Path, fieldnames, row) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def start_terminal_log(run_dir: Path):
    log_dir = run_dir / "terminal_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "console.log"
    log_file = log_path.open("a", buffering=1, encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"[PPO TRAIN] terminal output is being saved to {log_path}")
    return log_file, original_stdout, original_stderr


def save_svg_line_chart(path: Path, title: str, x_values, series) -> None:
    if not x_values or not series:
        return

    width = 1000
    height = 520
    left = 80
    right = 30
    top = 56
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316"]

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    all_y = []
    for _, values in series:
        all_y.extend(values)
    y_min = min(all_y)
    y_max = max(all_y)
    if y_min == y_max:
        pad = max(1.0, abs(y_min) * 0.1)
        y_min -= pad
        y_max += pad
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    def sx(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - float(y)) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="22" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#374151" stroke-width="1"/>',
    ]

    for i in range(6):
        frac = i / 5
        y = top + frac * plot_h
        value = y_max - frac * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-family="sans-serif" font-size="12" text-anchor="end" fill="#4b5563">{value:.3g}</text>')

    for idx, (label, values) in enumerate(series):
        color = colors[idx % len(colors)]
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = left + idx * 190
        legend_y = height - 25
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-family="sans-serif" font-size="13" fill="#111827">{label}</text>')

    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" font-family="sans-serif" font-size="13" text-anchor="middle" fill="#4b5563">step/update/episode</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_svg_training_plots(run_dir: Path, update_rows, episode_rows) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if update_rows:
        updates = [row["update"] for row in update_rows]
        save_svg_line_chart(
            plots_dir / "training_reward.svg",
            "PPO Mean Rollout Reward",
            updates,
            [("mean_rollout_reward", [row["mean_rollout_reward"] for row in update_rows])],
        )
        save_svg_line_chart(
            plots_dir / "training_goal_xy_err.svg",
            "PPO Goal XY Error",
            updates,
            [
                ("mean_xy_err", [row["mean_xy_err"] for row in update_rows]),
                ("max_xy_err", [row["max_xy_err"] for row in update_rows]),
            ],
        )
        save_svg_line_chart(
            plots_dir / "training_z_and_speed.svg",
            "PPO Z Error And Speed",
            updates,
            [
                ("mean_z_err", [row["mean_z_err"] for row in update_rows]),
                ("max_z_err", [row["max_z_err"] for row in update_rows]),
                ("mean_speed_xy", [row["mean_speed_xy"] for row in update_rows]),
                ("mean_speed_z", [row["mean_speed_z"] for row in update_rows]),
            ],
        )
        save_svg_line_chart(
            plots_dir / "training_progress.svg",
            "PPO Goal XY Progress",
            updates,
            [("mean_goal_xy_progress", [row["mean_goal_xy_progress"] for row in update_rows])],
        )
        save_svg_line_chart(
            plots_dir / "losses.svg",
            "PPO Losses And Entropy",
            updates,
            [
                ("policy_loss", [row["policy_loss"] for row in update_rows]),
                ("value_loss", [row["value_loss"] for row in update_rows]),
                ("entropy", [row["entropy"] for row in update_rows]),
            ],
        )

    if episode_rows:
        episodes = [row["episode"] for row in episode_rows]
        success_flags = np.asarray([1.0 if row["success"] else 0.0 for row in episode_rows], dtype=np.float32)
        cumulative_success = np.cumsum(success_flags) / np.arange(1, len(success_flags) + 1)
        save_svg_line_chart(
            plots_dir / "episode_final_goal_xy_err.svg",
            "Episode Final Goal XY Error",
            episodes,
            [("final_goal_xy_err", [row["final_goal_xy_err"] for row in episode_rows])],
        )
        save_svg_line_chart(
            plots_dir / "episode_success_rate.svg",
            "Episode Cumulative Success Rate",
            episodes,
            [("success_rate", cumulative_success.tolist())],
        )
        save_svg_line_chart(
            plots_dir / "episode_final_z_err.svg",
            "Episode Final Z Error",
            episodes,
            [
                ("final_z_err", [row["final_z_err"] for row in episode_rows]),
                ("final_speed_z", [row["final_speed_z"] for row in episode_rows]),
            ],
        )
        save_svg_line_chart(
            plots_dir / "episode_final_goal_rel_xy.svg",
            "Episode Final Goal Relative XY",
            episodes,
            [
                ("goal_rel_x", [row["final_goal_rel_x"] for row in episode_rows]),
                ("goal_rel_y", [row["final_goal_rel_y"] for row in episode_rows]),
            ],
        )

    print(f"[PPO TRAIN] saved SVG plots to {plots_dir}")


def save_training_plots(run_dir: Path, update_rows, episode_rows) -> None:
    if not update_rows and not episode_rows:
        return

    try:
        matplotlib_cache_dir = run_dir / "matplotlib_cache"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PPO TRAIN] matplotlib unavailable ({exc}); saving SVG plots instead")
        save_svg_training_plots(run_dir, update_rows, episode_rows)
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if update_rows:
        updates = [row["update"] for row in update_rows]
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(updates, [row["mean_rollout_reward"] for row in update_rows], label="mean rollout reward")
        axes[0].set_ylabel("reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(updates, [row["mean_xy_err"] for row in update_rows], label="mean goal xy err")
        axes[1].plot(updates, [row["max_xy_err"] for row in update_rows], label="max goal xy err", alpha=0.75)
        axes[1].set_ylabel("meters")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(updates, [row["mean_z_err"] for row in update_rows], label="mean z err")
        axes[2].plot(updates, [row["mean_signed_z_err"] for row in update_rows], label="mean signed z err")
        axes[2].plot(updates, [row["mean_speed_xy"] for row in update_rows], label="mean speed xy", alpha=0.75)
        axes[2].set_ylabel("meters / mps")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].bar(updates, [row["success_count"] for row in update_rows], label="success")
        axes[3].bar(updates, [row["timeout_count"] for row in update_rows], bottom=[row["success_count"] for row in update_rows], label="timeout")
        axes[3].set_ylabel("terminal episodes")
        axes[3].set_xlabel("update")
        axes[3].grid(True, axis="y", alpha=0.3)
        axes[3].legend()
        fig.tight_layout()
        overview_path = plots_dir / "training_overview.png"
        fig.savefig(overview_path, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(updates, [row["policy_loss"] for row in update_rows], label="policy loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].plot(updates, [row["value_loss"] for row in update_rows], label="value loss", color="tab:orange")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[2].plot(updates, [row["entropy"] for row in update_rows], label="entropy", color="tab:green")
        axes[2].set_xlabel("update")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        fig.tight_layout()
        losses_path = plots_dir / "losses.png"
        fig.savefig(losses_path, dpi=150)
        plt.close(fig)

    if episode_rows:
        episodes = [row["episode"] for row in episode_rows]
        final_xy = [row["final_goal_xy_err"] for row in episode_rows]
        colors = ["tab:green" if row["success"] else "tab:red" for row in episode_rows]
        success_flags = np.asarray([1.0 if row["success"] else 0.0 for row in episode_rows], dtype=np.float32)
        cumulative_success = np.cumsum(success_flags) / np.arange(1, len(success_flags) + 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(episodes, final_xy, c=colors, label="final goal xy err")
        ax.plot(episodes, final_xy, color="tab:blue", alpha=0.25)
        ax.set_xlabel("episode")
        ax.set_ylabel("final goal xy err (m)")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(episodes, cumulative_success, color="tab:green", label="cumulative success rate")
        ax2.set_ylabel("success rate")
        ax2.set_ylim(0.0, 1.05)
        fig.tight_layout()
        episodes_path = plots_dir / "episode_outcomes.png"
        fig.savefig(episodes_path, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(episodes, [row["final_z_err"] for row in episode_rows], label="final z err")
        axes[0].plot(episodes, [row["final_signed_z_err"] for row in episode_rows], label="final signed z err")
        axes[0].set_ylabel("meters")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(episodes, [row["final_goal_rel_x"] for row in episode_rows], label="goal rel x")
        axes[1].plot(episodes, [row["final_goal_rel_y"] for row in episode_rows], label="goal rel y")
        axes[1].set_ylabel("meters")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(episodes, [row["final_cmd_roll_rate"] for row in episode_rows], label="cmd roll rate")
        axes[2].plot(episodes, [row["final_cmd_pitch_rate"] for row in episode_rows], label="cmd pitch rate")
        axes[2].plot(episodes, [row["final_cmd_thrust"] for row in episode_rows], label="cmd thrust")
        axes[2].set_ylabel("command")
        axes[2].set_xlabel("episode")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        fig.tight_layout()
        diagnostics_path = plots_dir / "episode_diagnostics.png"
        fig.savefig(diagnostics_path, dpi=150)
        plt.close(fig)

    print(f"[PPO TRAIN] saved plots to {plots_dir}")


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
        hover_thrust=args.hover_thrust,
        thrust_delta=args.thrust_delta,
        thrust_min=args.thrust_min,
        thrust_max=args.thrust_max,
        residual_gain=args.residual_gain,
        goal_feedback_scale=args.goal_feedback_scale,
        attitude_feedback_scale=args.attitude_feedback_scale,
        z_pos_gain=args.z_pos_gain,
        z_vel_gain=args.z_vel_gain,
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
        recover_z_tolerance_m=args.recover_z_tolerance_m,
        start_logging=not args.no_pegasus_log,
        log_dir=args.pegasus_log_dir,
        reward_alive=args.reward_alive,
        reward_progress_scale=args.reward_progress_scale,
        reward_distance_scale=args.reward_distance_scale,
        reward_z_scale=args.reward_z_scale,
        reward_control_scale=0.05,
        reward_success=args.reward_success,
        reward_crash=-30.0,
        reward_timeout=args.reward_timeout,
        goal_xy_radius_min=args.goal_xy_radius_min,
        goal_xy_radius_max=args.goal_xy_radius_max,
        goal_z_delta_max=args.goal_z_delta_max,
        goal_tolerance_m=args.goal_tolerance_m,
        goal_z_tolerance_m=args.goal_z_tolerance_m,
        goal_speed_xy_tolerance_mps=args.goal_speed_xy_tolerance_mps,
        goal_speed_z_tolerance_mps=args.goal_speed_z_tolerance_mps,
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
    metrics_dir = run_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    terminal_log = None
    if not args.no_terminal_log:
        terminal_log = start_terminal_log(run_dir)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    update_metrics_path = metrics_dir / "update_metrics.csv"
    episode_metrics_path = metrics_dir / "episode_metrics.csv"

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
    print(f"goal_feedback_scale: {args.goal_feedback_scale}")
    print(f"attitude_feedback_scale: {args.attitude_feedback_scale}")
    print(
        f"thrust: hover={args.hover_thrust}, delta={args.thrust_delta}, "
        f"range=[{args.thrust_min}, {args.thrust_max}]"
    )
    print(f"z_pos_gain: {args.z_pos_gain}")
    print(f"z_vel_gain: {args.z_vel_gain}")
    print(f"recover_z_tolerance_m: {args.recover_z_tolerance_m}")
    print(
        f"goal_xy_radius: [{args.goal_xy_radius_min}, {args.goal_xy_radius_max}], "
        f"goal_z_delta_max: {args.goal_z_delta_max}, "
        f"goal_tolerance_m: {args.goal_tolerance_m}, "
        f"goal_z_tolerance_m: {args.goal_z_tolerance_m}"
    )
    print(
        f"success_speed_tolerance: xy={args.goal_speed_xy_tolerance_mps}, "
        f"z={args.goal_speed_z_tolerance_mps}"
    )
    print(
        f"reward: progress={args.reward_progress_scale}, distance={args.reward_distance_scale}, "
        f"z={args.reward_z_scale}, success={args.reward_success}, timeout={args.reward_timeout}"
    )
    print(f"init_action_std: {args.init_action_std}")
    print("=" * 80)

    obs, _ = env.reset()
    total_steps = 0
    update = 0
    update_rows = []
    episode_rows = []
    episode_rewards = []
    episode_xy_errs = []
    episode_z_errs = []
    episode_goal_distances = []
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
            stats_z = []
            stats_signed_z = []
            stats_progress = []
            stats_speed_xy = []
            stats_speed_z = []
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
                episode_rewards.append(float(reward))
                stats_reasons[info.get("done_reason", "running")] += 1
                if info.get("xy_err") is not None:
                    xy_err = float(info["xy_err"])
                    stats_xy.append(xy_err)
                    episode_xy_errs.append(xy_err)
                if info.get("z_err") is not None:
                    z_err = float(info["z_err"])
                    stats_z.append(z_err)
                    episode_z_errs.append(z_err)
                if info.get("goal_distance") is not None:
                    episode_goal_distances.append(float(info["goal_distance"]))
                if info.get("signed_z_err") is not None:
                    stats_signed_z.append(float(info["signed_z_err"]))
                if info.get("goal_xy_progress") is not None:
                    stats_progress.append(float(info["goal_xy_progress"]))
                if info.get("speed_xy") is not None:
                    stats_speed_xy.append(float(info["speed_xy"]))
                if info.get("speed_z") is not None:
                    stats_speed_z.append(float(info["speed_z"]))

                obs = next_obs
                total_steps += 1
                if done:
                    episode_row = {
                        "episode": int(info.get("episode_id", 0)),
                        "total_steps": int(total_steps),
                        "episode_steps": int(info.get("step_id", len(episode_rewards))),
                        "return": float(np.sum(episode_rewards)) if episode_rewards else 0.0,
                        "done_reason": str(info.get("done_reason", "unknown")),
                        "final_goal_xy_err": float(info["xy_err"]) if info.get("xy_err") is not None else 0.0,
                        "final_z_err": float(info["z_err"]) if info.get("z_err") is not None else 0.0,
                        "final_goal_distance": float(info["goal_distance"]) if info.get("goal_distance") is not None else 0.0,
                        "final_goal_rel_x": float(info["goal_rel_x"]) if info.get("goal_rel_x") is not None else 0.0,
                        "final_goal_rel_y": float(info["goal_rel_y"]) if info.get("goal_rel_y") is not None else 0.0,
                        "final_goal_rel_z": float(info["goal_rel_z"]) if info.get("goal_rel_z") is not None else 0.0,
                        "final_signed_z_err": float(info["signed_z_err"]) if info.get("signed_z_err") is not None else 0.0,
                        "final_speed_xy": float(info["speed_xy"]) if info.get("speed_xy") is not None else 0.0,
                        "final_speed_z": float(info["speed_z"]) if info.get("speed_z") is not None else 0.0,
                        "final_cmd_roll_rate": float(info["cmd_roll_rate"]) if info.get("cmd_roll_rate") is not None else 0.0,
                        "final_cmd_pitch_rate": float(info["cmd_pitch_rate"]) if info.get("cmd_pitch_rate") is not None else 0.0,
                        "final_cmd_yaw_rate": float(info["cmd_yaw_rate"]) if info.get("cmd_yaw_rate") is not None else 0.0,
                        "final_cmd_thrust": float(info["cmd_thrust"]) if info.get("cmd_thrust") is not None else 0.0,
                        "mean_goal_xy_err": float(np.mean(episode_xy_errs)) if episode_xy_errs else 0.0,
                        "max_goal_xy_err": float(np.max(episode_xy_errs)) if episode_xy_errs else 0.0,
                        "mean_z_err": float(np.mean(episode_z_errs)) if episode_z_errs else 0.0,
                        "max_z_err": float(np.max(episode_z_errs)) if episode_z_errs else 0.0,
                        "success": str(info.get("done_reason", "unknown")) == "success",
                    }
                    episode_rows.append(episode_row)
                    append_csv_row(episode_metrics_path, EPISODE_METRIC_FIELDS, episode_row)
                    episode_rewards = []
                    episode_xy_errs = []
                    episode_z_errs = []
                    episode_goal_distances = []
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
            mean_z = float(np.mean(stats_z)) if stats_z else 0.0
            max_z = float(np.max(stats_z)) if stats_z else 0.0
            mean_signed_z = float(np.mean(stats_signed_z)) if stats_signed_z else 0.0
            mean_progress = float(np.mean(stats_progress)) if stats_progress else 0.0
            mean_speed_xy = float(np.mean(stats_speed_xy)) if stats_speed_xy else 0.0
            max_speed_xy = float(np.max(stats_speed_xy)) if stats_speed_xy else 0.0
            mean_speed_z = float(np.mean(stats_speed_z)) if stats_speed_z else 0.0
            max_speed_z = float(np.max(stats_speed_z)) if stats_speed_z else 0.0
            policy_loss_mean = float(np.mean(pg_losses)) if pg_losses else 0.0
            value_loss_mean = float(np.mean(value_losses)) if value_losses else 0.0
            entropy_mean = float(np.mean(entropies)) if entropies else 0.0
            success_count = int(stats_reasons.get("success", 0))
            timeout_count = int(stats_reasons.get("timeout", 0))
            other_done_count = int(sum(
                count
                for reason, count in stats_reasons.items()
                if reason not in ("running", "success", "timeout")
            ))
            update_row = {
                "update": int(update),
                "total_steps": int(total_steps),
                "mean_rollout_reward": mean_reward,
                "mean_xy_err": mean_xy,
                "max_xy_err": max_xy,
                "mean_z_err": mean_z,
                "max_z_err": max_z,
                "mean_signed_z_err": mean_signed_z,
                "mean_goal_xy_progress": mean_progress,
                "mean_speed_xy": mean_speed_xy,
                "max_speed_xy": max_speed_xy,
                "mean_speed_z": mean_speed_z,
                "max_speed_z": max_speed_z,
                "success_count": success_count,
                "timeout_count": timeout_count,
                "other_done_count": other_done_count,
                "done_reasons": json.dumps(dict(stats_reasons), sort_keys=True),
                "policy_loss": policy_loss_mean,
                "value_loss": value_loss_mean,
                "entropy": entropy_mean,
            }
            update_rows.append(update_row)
            append_csv_row(update_metrics_path, UPDATE_METRIC_FIELDS, update_row)
            print("=" * 80)
            print(f"[PegasusPPO] update={update}, steps={total_steps}/{args.num_env_steps}")
            print(f"  mean_rollout_reward: {mean_reward:.4f}")
            print(f"  mean_xy_err: {mean_xy:.3f}")
            print(f"  max_xy_err: {max_xy:.3f}")
            print(f"  mean_z_err: {mean_z:.3f}")
            print(f"  mean_signed_z_err: {mean_signed_z:.3f}")
            print(f"  mean_goal_xy_progress: {mean_progress:.4f}")
            print(f"  mean_speed_xy: {mean_speed_xy:.3f}")
            print(f"  done_reasons: {dict(stats_reasons)}")
            print(f"  policy_loss: {policy_loss_mean:.6f}")
            print(f"  value_loss: {value_loss_mean:.6f}")
            print(f"  entropy: {entropy_mean:.6f}")
    except KeyboardInterrupt:
        print("\n[PPO TRAIN] interrupted, closing environment...")
    finally:
        env.close()
        print("[PPO TRAIN] environment closed")
        print(f"[PPO TRAIN] saved update metrics to {update_metrics_path}")
        print(f"[PPO TRAIN] saved episode metrics to {episode_metrics_path}")
        save_training_plots(run_dir, update_rows, episode_rows)
        if terminal_log is not None:
            log_file, original_stdout, original_stderr = terminal_log
            print(f"[PPO TRAIN] saved terminal output to {log_file.name}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
