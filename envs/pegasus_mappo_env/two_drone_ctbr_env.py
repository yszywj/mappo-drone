"""
Two-drone MAPPO-style environment for Pegasus/PX4 controlled through MAVLink CTBR.

This is intentionally a real-time/simulator-backed environment, not a vectorized
fast simulator.  Start with a smoke test before plugging it into MAPPO.

Returned interface:
    obs, share_obs, info = env.reset()
    obs, share_obs, rewards, dones, infos = env.step(actions)

Shapes:
    obs:       np.ndarray, shape (2, 25)
    share_obs: np.ndarray, shape (2, 50), same global state repeated for each agent
    actions:   np.ndarray/list, shape (2, 4), each in [-1, 1]
    rewards:   np.ndarray, shape (2, 1)
    dones:     np.ndarray, shape (2,), team done for now
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # MAPPO paper code often uses gym, not gymnasium
    try:
        import gym
        from gym import spaces
    except Exception:
        gym = object
        spaces = None

from envs.mavlink_ctbr_controller.ctbr_controller import CTBRController
from envs.mavlink_ctbr_controller.rl_bridge import (
    CTBRDroneRLAdapter,
    GoalPoint,
    HomePoint,
    goal_distance,
    inter_drone_distance,
    observation_vector,
)
from .config import TwoDroneEnvConfig


class TwoDroneCTBREnv(gym.Env if hasattr(gym, "Env") else object):
    metadata = {"name": "TwoDroneCTBREnv"}

    def __init__(self, config: Optional[TwoDroneEnvConfig] = None, seed: Optional[int] = None):
        self.config = config or TwoDroneEnvConfig()
        self.num_agents = 2
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._airborne = False
        self._episode_id = 0
        self._step_id = 0
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.controllers: List[CTBRController] = []
        self.agents: List[CTBRDroneRLAdapter] = []
        self.time_keeper = None

        self.obs_dim = 25
        self.share_obs_dim = self.obs_dim * self.num_agents
        self.action_dim = 4

        if spaces is not None:
            self.observation_space = [spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
                                      for _ in range(self.num_agents)]
            self.share_observation_space = [spaces.Box(-np.inf, np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
                                            for _ in range(self.num_agents)]
            self.action_space = [spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)
                                 for _ in range(self.num_agents)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def connect(self) -> None:
        if self._connected:
            return

        os.makedirs(self.config.log_dir, exist_ok=True)
        self.controllers = []
        self.agents = []
        for i in range(self.num_agents):
            drone_id = i + 1
            ctrl = CTBRController(
                connection_str=self.config.connection_strs[i],
                target_system=self.config.target_systems[i],
                log_dir=self.config.log_dir,
                log_subdir=f"drone_{drone_id}",
                log_filename=f"trajectory_{self._timestamp}_drone_{drone_id}",
                enable_logging=self.config.start_logging,
            )
            ctrl.configure_rl_sitl_params()
            agent = CTBRDroneRLAdapter(
                drone_id=drone_id,
                controller=ctrl,
                action_limits=self.config.action_limits,
                safety_limits=self.config.safety_limits,
            )
            agent.start_io(data_stream_hz=self.config.data_stream_hz, start_logging=self.config.start_logging)
            self.controllers.append(ctrl)
            self.agents.append(agent)

        self.time_keeper = self.controllers[0].get_sim_time_keeper()
        self._connected = True

    def close(self) -> None:
        for agent in self.agents:
            try:
                agent.cleanup()
            except Exception:
                pass
        self.controllers = []
        self.agents = []
        self._connected = False
        self._airborne = False

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self.connect()

        if not self._airborne and self.config.auto_takeoff_on_first_reset:
            self._takeoff_all_once()

        if not all(agent.state.home is not None for agent in self.agents):
            self._capture_homes()

        # Recover to stable home points before every RL episode.
        need_recover = not self._all_near_home(self.config.recover_tolerance_m)

        if need_recover:
            recover_ok = self._recover_all_to_home()
            if not recover_ok:
                raise RuntimeError("reset failed: at least one drone could not recover to home")
        else:
            for agent in self.agents:
                agent.set_safe_ctbr()
                agent.start_ctbr(self.config.ctbr_send_hz)

        self._sample_or_set_goals()
        self._step_id = 0
        self._episode_id += 1

        for agent in self.agents:
            agent.set_safe_ctbr()
            if hasattr(agent.controller, "set_episode"):
                agent.controller.set_episode(self._episode_id, phase="collect", step_id=0)
            agent.start_ctbr(self.config.ctbr_send_hz)

        obs, share_obs = self._build_obs_and_share_obs()
        info = self._build_info(done_reason="reset")
        return obs, share_obs, info

    def step(self, actions: Sequence[Sequence[float]]):
        if len(actions) != self.num_agents:
            raise ValueError(f"actions must have length {self.num_agents}, got {len(actions)}")

        for i, agent in enumerate(self.agents):
            agent.apply_policy_action(actions[i])
            if hasattr(agent.controller, "set_episode_step"):
                agent.controller.set_episode_step(self._step_id)

        ok_time = self.time_keeper.wait(self.config.step_dt_sim_sec, timeout=2.0)
        self._step_id += 1

        obs, share_obs = self._build_obs_and_share_obs()
        rewards, dones, done_reason = self._compute_rewards_and_dones(actions, ok_time=ok_time)
        infos = self._build_info(done_reason=done_reason)

        if np.any(dones):
            normal_episode_end = done_reason in ["success", "timeout"]

            for agent in self.agents:
                # 正常 episode 结束：不要停 CTBR，只把控制量置为安全悬停值
                if normal_episode_end:
                    agent.set_safe_ctbr()
                else:
                    # 真正异常才停，后续由 reset/recover 接管
                    agent.stop_ctbr()

                if hasattr(agent.controller, "mark_episode_done"):
                    agent.controller.mark_episode_done(
                        reason=done_reason,
                        crashed=not normal_episode_end,
                    )

        return obs, share_obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Core mechanics
    # ------------------------------------------------------------------

    def _takeoff_all_once(self) -> None:
        # Start with sequential takeoff for reliability.  You can parallelize later.
        for i, ctrl in enumerate(self.controllers):
            ok = ctrl.auto_takeoff(
                target_altitude=self.config.takeoff_altitudes[i],
                timeout=int(self.config.takeoff_timeout_sim_sec),
                use_sim_time=True,
            )
            if not ok:
                raise RuntimeError(f"takeoff failed for drone {i + 1}")

        self.time_keeper.wait(self.config.stabilize_after_takeoff_sim_sec, timeout=5.0)
        for i, ctrl in enumerate(self.controllers):
            obs = ctrl.data_sync.get_latest_observation()
            ctrl.change_control_mode(
                mode=6,
                is_maintain_offboard=False,
                default_x=obs.x,
                default_y=obs.y,
                default_z=obs.z,
            )
        self.time_keeper.wait(0.5, timeout=2.0)
        self._capture_homes()
        self._airborne = True

    def _capture_homes(self) -> None:
        for agent in self.agents:
            agent.capture_home()

    def _recover_all_to_home(self) -> bool:
        homes = []
        for agent in self.agents:
            if agent.state.home is None:
                raise RuntimeError(f"drone {agent.drone_id}: home is not set")
            homes.append(agent.state.home)

        # 停 CTBR 前先设安全值，避免最后一条策略动作继续保持
        for agent in self.agents:
            agent.set_safe_ctbr()
            agent.stop_ctbr()
            if hasattr(agent.controller, "set_episode_phase"):
                agent.controller.set_episode_phase("recover")

        # 如果已经 OFFBOARD，不重复切模式；否则切一次
        for agent, home in zip(self.agents, homes):
            ctrl = agent.controller
            if not (getattr(ctrl, "_armed", False) and ctrl._flight_mode_name() == "OFFBOARD"):
                ok = ctrl.change_control_mode(
                    mode=6,
                    is_maintain_offboard=False,
                    default_x=home.x,
                    default_y=home.y,
                    default_z=home.z,
                    wait_for_data_timeout=0.5,
                )
                if not ok:
                    return False

        start_ms = self.time_keeper.now_ms()
        timeout_ms = int(self.config.recover_timeout_sim_sec * 1000)

        while self.time_keeper.now_ms() - start_ms < timeout_ms:
            all_ok = True

            # 关键：每一轮都给所有无人机发 setpoint
            for agent, home in zip(self.agents, homes):
                agent.controller.send_hover_setpoint(home.x, home.y, home.z)

            for agent, home in zip(self.agents, homes):
                safety = agent.check_single_drone_safety()

                # 短暂 stale 不要立即失败；真正 disarmed/failsafe/near_ground 才失败
                if safety.abnormal and not safety.recoverable:
                    if safety.reason not in ["stale_observation"]:
                        return False

                obs = agent.get_observation()
                err = math.sqrt(
                    (float(obs.x) - home.x) ** 2
                    + (float(obs.y) - home.y) ** 2
                    + (float(obs.z) - home.z) ** 2
                )
                if err >= self.config.recover_tolerance_m:
                    all_ok = False

            if all_ok:
                for agent in self.agents:
                    agent.set_safe_ctbr()
                    agent.start_ctbr(self.config.ctbr_send_hz)
                return True

            self.time_keeper.wait(0.05, timeout=1.0)

        return False

    def _sample_or_set_goals(self) -> None:
        if self.config.fixed_goals is not None:
            for i, agent in enumerate(self.agents):
                gx, gy, gz = self.config.fixed_goals[i]
                agent.set_goal(GoalPoint(gx, gy, gz))
            return

        # Sample goals around each home point.  For the first stable version, keep z near home.
        for agent in self.agents:
            if agent.state.home is None:
                raise RuntimeError("home must be set before sampling goals")
            home = agent.state.home
            r = self._rng.uniform(self.config.goal_xy_radius_min, self.config.goal_xy_radius_max)
            theta = self._rng.uniform(-math.pi, math.pi)
            dz = self._rng.uniform(-self.config.goal_z_delta_max, self.config.goal_z_delta_max)
            goal = GoalPoint(
                x=home.x + r * math.cos(theta),
                y=home.y + r * math.sin(theta),
                z=home.z + dz,
            )
            agent.set_goal(goal)

    def _build_obs_and_share_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        raw_obs = [agent.get_observation() for agent in self.agents]
        obs_n = []
        for i, agent in enumerate(self.agents):
            other_i = 1 - i
            goal = agent.state.goal
            if goal is None:
                raise RuntimeError("goal is not set")
            obs_vec = observation_vector(
                own=raw_obs[i],
                other=raw_obs[other_i],
                goal=goal,
                prev_action=agent.state.prev_action,
            )
            obs_n.append(obs_vec)

        obs = np.stack(obs_n, axis=0).astype(np.float32)
        global_state = obs.reshape(-1).astype(np.float32)
        share_obs = np.stack([global_state.copy() for _ in range(self.num_agents)], axis=0)
        return obs, share_obs

    def _compute_rewards_and_dones(self, actions: Sequence[Sequence[float]], ok_time: bool):
        raw_obs = [agent.get_observation() for agent in self.agents]
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        done = False
        done_reason = "running"

        if not ok_time:
            done = True
            done_reason = "sim_time_timeout"

        # Single-drone safety.
        safety_results = [agent.check_single_drone_safety() for agent in self.agents]
        for i, safety in enumerate(safety_results):
            if safety.abnormal:
                done = True
                done_reason = safety.reason
                rewards[i, 0] += self.config.reward_crash

        # Multi-drone collision / proximity.
        d12 = inter_drone_distance(raw_obs[0], raw_obs[1])
        if d12 < self.config.collision_distance_m:
            done = True
            done_reason = f"collision_distance={d12:.2f}m"
            rewards[:, 0] += self.config.reward_collision
        elif d12 < self.config.warning_distance_m:
            # Smooth penalty when too close.
            proximity = (self.config.warning_distance_m - d12) / max(1e-6, self.config.warning_distance_m)
            rewards[:, 0] -= self.config.reward_close_penalty_scale * proximity

        # Goal progress and distance shaping.
        all_reached = True
        for i, agent in enumerate(self.agents):
            goal = agent.state.goal
            assert goal is not None
            dist = goal_distance(raw_obs[i], goal)
            prev = agent.state.last_goal_distance if agent.state.last_goal_distance is not None else dist
            progress = prev - dist
            agent.state.last_goal_distance = dist

            action_vec = np.asarray(actions[i], dtype=np.float32)
            control_penalty = float(np.mean(np.square(np.clip(action_vec, -1.0, 1.0))))

            rewards[i, 0] += self.config.reward_alive
            rewards[i, 0] += self.config.reward_progress_scale * progress
            rewards[i, 0] -= self.config.reward_distance_scale * dist
            rewards[i, 0] -= self.config.reward_control_scale * control_penalty

            if dist > self.config.goal_tolerance_m:
                all_reached = False

        if all_reached and not done:
            done = True
            done_reason = "success"
            rewards[:, 0] += self.config.reward_success

        if self._step_id >= self.config.episode_length and not done:
            done = True
            done_reason = "timeout"
            rewards[:, 0] += self.config.reward_timeout

        dones = np.array([done] * self.num_agents, dtype=bool)
        return rewards, dones, done_reason

    def _build_info(self, done_reason: str):
        raw_obs = [agent.get_observation() for agent in self.agents]
        goals = [agent.state.goal for agent in self.agents]
        dists_to_goal = [goal_distance(raw_obs[i], goals[i]) if goals[i] else None for i in range(self.num_agents)]
        d12 = inter_drone_distance(raw_obs[0], raw_obs[1])
        infos = []
        for i, agent in enumerate(self.agents):
            snap = agent.snapshot()
            infos.append({
                "episode_id": self._episode_id,
                "step_id": self._step_id,
                "drone_id": agent.drone_id,
                "done_reason": done_reason,
                "goal_distance": dists_to_goal[i],
                "inter_drone_distance": d12,
                "is_fresh": snap.is_fresh,
                "armed": snap.armed,
                "flight_mode": snap.flight_mode,
                "last_status_text": snap.last_status_text,
                "goal": None if goals[i] is None else (goals[i].x, goals[i].y, goals[i].z),
                "home": None if agent.state.home is None else (agent.state.home.x, agent.state.home.y, agent.state.home.z),
            })
        return infos

    def _all_near_home(self, tolerance_m: float) -> bool:
        for agent in self.agents:
            if agent.state.home is None:
                return False

            obs = agent.get_observation()
            home = agent.state.home
            err = math.sqrt(
                (float(obs.x) - home.x) ** 2
                + (float(obs.y) - home.y) ** 2
                + (float(obs.z) - home.z) ** 2
            )
            if err > tolerance_m:
                return False

        return True

    # ------------------------------------------------------------------
    # Convenience methods for MAPPO integrations
    # ------------------------------------------------------------------

    def get_env_info(self) -> Dict[str, int]:
        return {
            "num_agents": self.num_agents,
            "obs_shape": self.obs_dim,
            "share_obs_shape": self.share_obs_dim,
            "action_shape": self.action_dim,
            "episode_limit": self.config.episode_length,
        }
