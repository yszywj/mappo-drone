"""
Small vector-env adapter for the real Pegasus/PX4 environment.

The MAPPO implementation expects a vectorized environment interface with shape:
    obs:       (n_rollout_threads, num_agents, obs_dim)
    share_obs: (n_rollout_threads, num_agents, share_obs_dim)
    rewards:   (n_rollout_threads, num_agents, 1)
    dones:     (n_rollout_threads, num_agents)

For Pegasus/PX4, use n_rollout_threads = 1. Running multiple real-time
Pegasus/PX4 worlds in parallel is not supported by this adapter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class PegasusSingleVecEnv:
    """Vectorized wrapper around exactly one simulator-backed two-drone env."""

    def __init__(
        self,
        env_fn: Callable[[], object],
        *,
        auto_reset: bool = True,
        stop_on_unrecoverable: bool = True,
    ):
        self.env = env_fn()
        self.auto_reset = bool(auto_reset)
        self.stop_on_unrecoverable = bool(stop_on_unrecoverable)
        self.num_envs = 1

        # MAPPO runner expects these attributes.
        self.num_agents = self.env.num_agents
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space
        self.action_space = self.env.action_space

        self.last_infos = None

    def reset(self):
        obs, share_obs, infos = self.env.reset()
        return (
            obs[None, ...].astype(np.float32),
            share_obs[None, ...].astype(np.float32),
            [infos],
        )

    def step(self, actions):
        """
        actions: np.ndarray, shape (1, num_agents, action_dim)
        returns:
            obs:       (1, num_agents, obs_dim)
            share_obs: (1, num_agents, share_obs_dim)
            rewards:   (1, num_agents, 1)
            dones:     (1, num_agents)
            infos:     list length 1, each item is list[num_agents] of info dict
        """
        if actions.shape[0] != 1:
            raise ValueError("PegasusSingleVecEnv only supports n_rollout_threads=1")

        obs, share_obs, rewards, dones, infos = self.env.step(actions[0])

        terminal_infos = infos

        terminal_reason = "unknown"
        if terminal_infos:
            for info in terminal_infos:
                reason = info.get("done_reason", "unknown")
                if reason not in ["running", "reset", "unknown"]:
                    terminal_reason = reason
                    break

            if terminal_reason == "unknown":
                terminal_reason = terminal_infos[0].get("done_reason", "unknown")

        if bool(np.any(dones)):
            # 这些是真正危险/不可恢复风险，直接停止。
            hard_unrecoverable_keywords = [
                "near_ground",
                "disarmed",
                "failsafe",
                "stale_observation",
            ]

            # 这些可能只是保守阈值触发，允许先尝试 reset/recover。
            soft_recoverable_keywords = [
                "timeout",
                "too_high_alt",
                "xy_out_of_bounds",
                "z_out_of_bounds",
                "tilt_too_large",
                "body_rate_too_large",
                "falling_fast",
                "collision_distance",
            ]

            is_hard_unrecoverable = any(
                k in terminal_reason for k in hard_unrecoverable_keywords
            )

            is_soft_recoverable = any(
                k in terminal_reason for k in soft_recoverable_keywords
            )

            if is_hard_unrecoverable and self.stop_on_unrecoverable:
                self.close()
                raise RuntimeError(
                    "Unrecoverable Pegasus/PX4 episode termination: "
                    f"{terminal_reason}. Stop training and restart Pegasus/PX4 before continuing."
                )

            if self.auto_reset:
                try:
                    reset_obs, reset_share_obs, reset_infos = self.env.reset()

                    obs = reset_obs
                    share_obs = reset_share_obs

                    for i in range(len(terminal_infos)):
                        terminal_infos[i]["reset_after_done"] = True
                        terminal_infos[i]["reset_success"] = True
                        if i < len(reset_infos):
                            terminal_infos[i]["next_reset_home"] = reset_infos[i].get("home", None)

                except Exception as e:
                    # reset/recover 失败，说明当前仿真状态已经不适合继续训练。
                    for i in range(len(terminal_infos)):
                        terminal_infos[i]["reset_after_done"] = True
                        terminal_infos[i]["reset_success"] = False
                        terminal_infos[i]["reset_error"] = str(e)

                    self.close()
                    raise RuntimeError(
                        "Pegasus/PX4 episode ended with a recoverable reason, "
                        "but reset/recover_to_home failed.\n"
                        f"terminal_reason={terminal_reason}\n"
                        f"reset_error={e}\n"
                        "Stop training and restart Pegasus/PX4 before continuing."
                    ) from e

        return (
            obs[None, ...].astype(np.float32),
            share_obs[None, ...].astype(np.float32),
            rewards[None, ...].astype(np.float32),
            dones[None, ...].astype(bool),
            [terminal_infos],
        )

    def close(self):
        self.env.close()
