"""
Stage-1 Pegasus/PX4 task for MAPPO: safe hover.

This task intentionally does NOT train goal reaching yet. The objective is:
- stay armed and in OFFBOARD/CTBR without PX4 failsafe
- keep altitude near the post-takeoff home point
- keep XY drift bounded
- avoid collision
- finish the episode by timeout, then recover to home and start the next episode

It subclasses TwoDroneCTBREnv, so it preserves your existing workflow:
first takeoff once -> capture home -> every reset recovers to home -> next episode.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from envs.mavlink_ctbr_controller.rl_bridge import (
    GoalPoint,
    goal_distance,
    inter_drone_distance,
)
from envs.pegasus_mappo_env.two_drone_ctbr_env import TwoDroneCTBREnv


class SafeHoverTwoDroneEnv(TwoDroneCTBREnv):
    """
    Conservative first curriculum task.

    Goal points are set to each drone's current home point after reset/recover.
    The reward is dominated by safety and home-position stability, not navigation.
    """

    def _sample_or_set_goals(self) -> None:
        # For hover training, the "goal" is the recovered home point.
        # This keeps the existing observation layout unchanged:
        # goal_rel = home - current_position.
        for agent in self.agents:
            home = agent.state.home
            if home is None:
                raise RuntimeError("home must be set before safe-hover goal assignment")
            agent.set_goal(GoalPoint(home.x, home.y, home.z))

    def _compute_rewards_and_dones(self, actions: Sequence[Sequence[float]], ok_time: bool):
        raw_obs = [agent.get_observation() for agent in self.agents]
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        done = False
        done_reason = "running"

        if not ok_time:
            done = True
            done_reason = "sim_time_timeout"
            rewards[:, 0] += self.config.reward_crash

        # Single-drone safety checks reuse your existing CTBRDroneRLAdapter logic.
        safety_results = [agent.check_single_drone_safety() for agent in self.agents]
        for i, safety in enumerate(safety_results):
            if safety.abnormal:
                done = True
                done_reason = safety.reason
                rewards[i, 0] += self.config.reward_crash

        # Collision/proximity safety.
        d12 = inter_drone_distance(raw_obs[0], raw_obs[1])
        if d12 < self.config.collision_distance_m:
            done = True
            done_reason = f"collision_distance={d12:.2f}m"
            rewards[:, 0] += self.config.reward_collision
        elif d12 < self.config.warning_distance_m:
            proximity = (self.config.warning_distance_m - d12) / max(1e-6, self.config.warning_distance_m)
            rewards[:, 0] -= self.config.reward_close_penalty_scale * proximity

        # Dense safe-hover shaping.
        for i, agent in enumerate(self.agents):
            obs = raw_obs[i]
            home = agent.state.home
            if home is None:
                raise RuntimeError("home is not set during safe-hover reward computation")

            xy_err = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
            z_err = abs(float(obs.z) - home.z)
            speed = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2 + float(obs.vz) ** 2)
            tilt = math.sqrt(float(obs.roll) ** 2 + float(obs.pitch) ** 2)
            action_vec = np.asarray(actions[i], dtype=np.float32)
            control_penalty = float(np.mean(np.square(np.clip(action_vec, -1.0, 1.0))))

            # Update last_goal_distance for logging/info consistency.
            if agent.state.goal is not None:
                agent.state.last_goal_distance = goal_distance(obs, agent.state.goal)

            rewards[i, 0] += self.config.reward_alive      # alive bonus
            rewards[i, 0] -= 0.20 * xy_err                 # stay near home horizontally
            rewards[i, 0] -= 0.45 * z_err                  # stay near home altitude
            rewards[i, 0] -= 0.04 * speed                  # avoid drifting fast
            rewards[i, 0] -= 0.08 * tilt                   # avoid large attitude
            rewards[i, 0] -= self.config.reward_control_scale * control_penalty

        # Timeout is the normal successful ending for this curriculum stage.
        if self._step_id >= self.config.episode_length and not done:
            done = True
            done_reason = "timeout"
            rewards[:, 0] += self.config.reward_timeout

        dones = np.array([done] * self.num_agents, dtype=bool)
        return rewards, dones, done_reason
