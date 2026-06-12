from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
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
    goal_distance,
    observation_vector,
)
from .config import SingleDroneEnvConfig


class SingleDroneHoverEnv(gym.Env if hasattr(gym, "Env") else object):
    """
    Single-drone safe-hover task for PPO residual learning.

    Observation layout, length 27:
      own pos(3), own vel(3), own attitude(3), own body rates(3),
      goal relative pos(3), other relative pos(3), other relative vel(3),
      prev CTBR action(4), inside goal zone(1), goal dwell fraction(1).

    For single-drone PPO, "other" is a virtual copy at the same pose/velocity,
    so other relative pos/vel are zeros.
    Action layout, length 4:
      normalized residual CTBR action in [-1, 1]; CTBRDroneRLAdapter combines it
      with the PD stabilizer according to action_limits.residual_gain.
    """

    metadata = {"name": "SingleDroneHoverEnv"}

    def __init__(self, config: Optional[SingleDroneEnvConfig] = None, seed: Optional[int] = None):
        self.config = config or SingleDroneEnvConfig()
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._airborne = False
        self._episode_id = 0
        self._step_id = 0
        self._last_goal_xy_err: Optional[float] = None
        self._last_goal_xy_progress = 0.0
        self._last_reward_terms: Dict[str, float] = {}
        self._inside_goal_zone = False
        self._goal_dwell_steps = 0
        self._goal_dwell_fraction = 0.0
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.controller: Optional[CTBRController] = None
        self.agent: Optional[CTBRDroneRLAdapter] = None
        self.time_keeper = None

        self.obs_dim = 27
        self.action_dim = 4

        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def connect(self) -> None:
        if self._connected:
            return

        os.makedirs(self.config.log_dir, exist_ok=True)
        self.controller = CTBRController(
            connection_str=self.config.connection_str,
            target_system=self.config.target_system,
            log_dir=self.config.log_dir,
            log_subdir="single_drone",
            log_filename=f"trajectory_{self._timestamp}_single_drone",
            enable_logging=self.config.start_logging,
        )
        self.controller.configure_rl_sitl_params()
        self.agent = CTBRDroneRLAdapter(
            drone_id=1,
            controller=self.controller,
            action_limits=self.config.action_limits,
            safety_limits=self.config.safety_limits,
        )
        self.agent.start_io(
            data_stream_hz=self.config.data_stream_hz,
            start_logging=self.config.start_logging,
        )
        self.time_keeper = self.controller.get_sim_time_keeper()
        self._connected = True

    def close(self) -> None:
        if self.agent is not None:
            try:
                self.agent.cleanup()
            except Exception:
                pass
        self.controller = None
        self.agent = None
        self.time_keeper = None
        self._connected = False
        self._airborne = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self.connect()
        assert self.agent is not None
        assert self.controller is not None

        if not self._airborne and self.config.auto_takeoff_on_first_reset:
            self._takeoff_once()

        if self.agent.state.home is None:
            self.agent.capture_home()

        if not self._near_home_and_slow():
            ok = self._recover_to_home()
            if not ok:
                raise RuntimeError("reset failed: drone could not recover to home")
        else:
            self.agent.set_safe_ctbr()
            self.agent.start_ctbr(self.config.ctbr_send_hz)

        self._step_id = 0
        self._episode_id += 1
        self._last_reward_terms = {}
        self._inside_goal_zone = False
        self._goal_dwell_steps = 0
        self._goal_dwell_fraction = 0.0
        home = self.agent.state.home
        assert home is not None
        self._sample_or_set_goal()
        self._refresh_goal_zone_state(reset_dwell=True)
        self.agent.set_safe_ctbr()
        if hasattr(self.controller, "set_episode"):
            self.controller.set_episode(self._episode_id, phase="collect", step_id=0)
        self.agent.start_ctbr(self.config.ctbr_send_hz)

        self._print_reset_ready_state()
        obs = self._build_obs()
        info = self._build_info("reset")
        return obs, info

    def step(self, action):
        assert self.agent is not None
        if np.asarray(action).shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {np.asarray(action).shape}")

        self.agent.apply_policy_action(np.clip(action, -1.0, 1.0))
        if hasattr(self.agent.controller, "set_episode_step"):
            self.agent.controller.set_episode_step(self._step_id)

        ok_time = self.time_keeper.wait(self.config.step_dt_sim_sec, timeout=2.0)
        self._step_id += 1

        reward, done, done_reason = self._compute_reward_done(action, ok_time=ok_time)
        obs = self._build_obs()
        info = self._build_info(done_reason)

        if done:
            self._print_done_state(done_reason)
            normal_episode_end = done_reason in ("success", "timeout")
            if normal_episode_end:
                self.agent.set_safe_ctbr()
            else:
                self.agent.stop_ctbr()
            if hasattr(self.agent.controller, "mark_episode_done"):
                self.agent.controller.mark_episode_done(
                    reason=done_reason,
                    crashed=not normal_episode_end,
                )

        return obs, float(reward), bool(done), info

    def _takeoff_once(self) -> None:
        assert self.controller is not None
        ok = self.controller.auto_takeoff(
            target_altitude=self.config.takeoff_altitude,
            timeout=int(self.config.takeoff_timeout_sim_sec),
            use_sim_time=True,
        )
        if not ok:
            raise RuntimeError("takeoff failed")

        self.time_keeper.wait(self.config.stabilize_after_takeoff_sim_sec, timeout=8.0)
        obs = self.controller.data_sync.get_latest_observation()
        self.controller.change_control_mode(
            mode=6,
            is_maintain_offboard=False,
            default_x=obs.x,
            default_y=obs.y,
            default_z=obs.z,
        )
        self.time_keeper.wait(0.5, timeout=2.0)
        assert self.agent is not None
        self.agent.capture_home()
        self._airborne = True

    def _recover_to_home(self) -> bool:
        assert self.agent is not None
        assert self.controller is not None
        if self.agent.state.home is None:
            raise RuntimeError("home is not set")

        home = self.agent.state.home
        self.agent.set_safe_ctbr()
        self.agent.stop_ctbr()
        if hasattr(self.controller, "set_episode_phase"):
            self.controller.set_episode_phase("recover")

        if not self.controller.is_probably_offboard():
            ok = self.controller.change_control_mode(
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
            safety = self.agent.check_single_drone_safety()
            if safety.abnormal and not safety.recoverable:
                if safety.reason not in ["stale_observation"]:
                    return False

            self.controller.send_hover_setpoint(home.x, home.y, home.z)

            if self._near_home_and_slow():
                hold_start_ms = self.time_keeper.now_ms()
                hold_ms = 2000
                stable = True

                while self.time_keeper.now_ms() - hold_start_ms < hold_ms:
                    self.controller.send_hover_setpoint(home.x, home.y, home.z)
                    self.time_keeper.wait(0.05, timeout=1.0)
                    if not self._near_home_and_slow():
                        stable = False
                        break

                if stable:
                    self.agent.set_safe_ctbr()
                    self.agent.start_ctbr(self.config.ctbr_send_hz)
                    return True

            self.time_keeper.wait(0.05, timeout=1.0)

        return False

    def _sample_or_set_goal(self) -> None:
        assert self.agent is not None
        if self.agent.state.home is None:
            raise RuntimeError("home must be set before sampling goal")

        home = self.agent.state.home
        min_r = max(0.0, self.config.goal_xy_radius_min)
        max_r = max(min_r, self.config.goal_xy_radius_max)
        r = self._rng.uniform(min_r, max_r)
        theta = self._rng.uniform(-math.pi, math.pi)
        z_delta_max = max(0.0, self.config.goal_z_delta_max)
        dz = self._rng.uniform(-z_delta_max, z_delta_max)
        goal = GoalPoint(
            x=home.x + r * math.cos(theta),
            y=home.y + r * math.sin(theta),
            z=home.z + dz,
        )
        self.agent.set_goal(goal)
        obs = self.agent.get_observation()
        self._last_goal_xy_err = math.sqrt((float(obs.x) - goal.x) ** 2 + (float(obs.y) - goal.y) ** 2)
        self._last_goal_xy_progress = 0.0

    def _build_obs(self) -> np.ndarray:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        if home is None:
            raise RuntimeError("home must be set before building observation")
        goal = self.agent.state.goal or GoalPoint(home.x, home.y, home.z)
        base_obs = observation_vector(
            own=obs,
            other=obs,
            goal=goal,
            prev_action=self.agent.state.prev_action,
        )
        dwell_obs = np.array([
            1.0 if self._inside_goal_zone else 0.0,
            self._goal_dwell_fraction,
        ], dtype=np.float32)
        return np.concatenate([base_obs, dwell_obs]).astype(np.float32)

    @property
    def _required_goal_dwell_steps(self) -> int:
        if self.config.success_dwell_sec <= 0.0:
            return 1
        return max(1, int(math.ceil(self.config.success_dwell_sec / self.config.step_dt_sim_sec)))

    def _goal_zone_status(self) -> Tuple[bool, float, float, float, float]:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        if home is None:
            raise RuntimeError("home is not set during goal-zone check")
        goal = self.agent.state.goal or GoalPoint(home.x, home.y, home.z)
        xy_err = math.sqrt((float(obs.x) - goal.x) ** 2 + (float(obs.y) - goal.y) ** 2)
        z_err = abs(float(obs.z) - goal.z)
        speed_xy = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2)
        speed_z = abs(float(obs.vz))
        inside_goal_zone = (
            xy_err <= self.config.goal_tolerance_m
            and z_err <= self.config.goal_z_tolerance_m
            and speed_xy <= self.config.goal_speed_xy_tolerance_mps
            and speed_z <= self.config.goal_speed_z_tolerance_mps
        )
        return inside_goal_zone, xy_err, z_err, speed_xy, speed_z

    def _refresh_goal_zone_state(self, reset_dwell: bool = False) -> None:
        inside_goal_zone, _, _, _, _ = self._goal_zone_status()
        self._inside_goal_zone = inside_goal_zone
        if reset_dwell:
            self._goal_dwell_steps = 0
        elif inside_goal_zone:
            self._goal_dwell_steps += 1
        else:
            self._goal_dwell_steps = 0
        self._goal_dwell_fraction = min(
            1.0,
            float(self._goal_dwell_steps) / float(self._required_goal_dwell_steps),
        )

    def _compute_reward_done(self, action, ok_time: bool) -> Tuple[float, bool, str]:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        if home is None:
            raise RuntimeError("home is not set during reward computation")

        done = False
        done_reason = "running"
        reward_crash = 0.0

        if not ok_time:
            done = True
            done_reason = "sim_time_timeout"
            reward_crash += self.config.reward_crash

        safety = self.agent.check_single_drone_safety()
        if safety.abnormal:
            done = True
            done_reason = safety.reason
            reward_crash += self.config.reward_crash

        goal = self.agent.state.goal or GoalPoint(home.x, home.y, home.z)
        xy_err = math.sqrt((float(obs.x) - goal.x) ** 2 + (float(obs.y) - goal.y) ** 2)
        z_err = abs(float(obs.z) - goal.z)
        prev_xy_err = self._last_goal_xy_err if self._last_goal_xy_err is not None else xy_err
        goal_xy_progress = prev_xy_err - xy_err
        self._last_goal_xy_err = xy_err
        self._last_goal_xy_progress = goal_xy_progress
        speed = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2 + float(obs.vz) ** 2)
        tilt = math.sqrt(float(obs.roll) ** 2 + float(obs.pitch) ** 2)
        control_penalty = float(np.mean(np.square(np.clip(action, -1.0, 1.0))))

        reward_alive = self.config.reward_alive
        reward_progress = self.config.reward_progress_scale * goal_xy_progress
        reward_distance = -self.config.reward_distance_scale * xy_err
        reward_z = -self.config.reward_z_scale * z_err
        reward_speed = -0.04 * speed
        reward_tilt = -0.08 * tilt
        reward_control = -self.config.reward_control_scale * control_penalty
        reward_goal_zone = 0.0
        reward_dwell = 0.0
        reward_success = 0.0
        reward_timeout = 0.0

        speed_xy = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2)
        speed_z = abs(float(obs.vz))
        inside_goal_zone = (
            xy_err <= self.config.goal_tolerance_m
            and z_err <= self.config.goal_z_tolerance_m
            and speed_xy <= self.config.goal_speed_xy_tolerance_mps
            and speed_z <= self.config.goal_speed_z_tolerance_mps
        )
        self._inside_goal_zone = inside_goal_zone
        if inside_goal_zone:
            self._goal_dwell_steps += 1
            self._goal_dwell_fraction = min(
                1.0,
                float(self._goal_dwell_steps) / float(self._required_goal_dwell_steps),
            )
            reward_goal_zone = self.config.reward_goal_zone
            reward_dwell = self.config.reward_dwell_scale * self._goal_dwell_fraction
        else:
            self._goal_dwell_steps = 0
            self._goal_dwell_fraction = 0.0

        if not done and self._goal_dwell_steps >= self._required_goal_dwell_steps:
            done = True
            done_reason = "success"
            reward_success = self.config.reward_success

        if self._step_id >= self.config.episode_length and not done:
            done = True
            done_reason = "timeout"
            reward_timeout = self.config.reward_timeout

        reward = (
            reward_alive
            + reward_progress
            + reward_distance
            + reward_z
            + reward_speed
            + reward_tilt
            + reward_control
            + reward_goal_zone
            + reward_dwell
            + reward_success
            + reward_crash
            + reward_timeout
        )
        self._last_reward_terms = {
            "reward_alive": float(reward_alive),
            "reward_progress": float(reward_progress),
            "reward_distance": float(reward_distance),
            "reward_z": float(reward_z),
            "reward_speed": float(reward_speed),
            "reward_tilt": float(reward_tilt),
            "reward_control": float(reward_control),
            "reward_goal_zone": float(reward_goal_zone),
            "reward_dwell": float(reward_dwell),
            "reward_success": float(reward_success),
            "reward_crash": float(reward_crash),
            "reward_timeout": float(reward_timeout),
            "reward_total": float(reward),
        }

        return reward, done, done_reason

    def _build_info(self, done_reason: str) -> Dict[str, Any]:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        goal = self.agent.state.goal
        xy_err = None
        z_err = None
        goal_dist = None
        home_xy_err = None
        goal_rel_x = None
        goal_rel_y = None
        goal_rel_z = None
        signed_z_err = None
        speed_xy = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2)
        speed_z = abs(float(obs.vz))
        cmd = self.agent.state.prev_action
        if goal is not None:
            goal_rel_x = goal.x - float(obs.x)
            goal_rel_y = goal.y - float(obs.y)
            goal_rel_z = goal.z - float(obs.z)
            signed_z_err = float(obs.z) - goal.z
            xy_err = math.sqrt((float(obs.x) - goal.x) ** 2 + (float(obs.y) - goal.y) ** 2)
            z_err = abs(float(obs.z) - goal.z)
            goal_dist = goal_distance(obs, goal)
        if home is not None:
            home_xy_err = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
        return {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "done_reason": done_reason,
            "xy_err": xy_err,
            "z_err": z_err,
            "goal_distance": goal_dist,
            "goal_xy_progress": self._last_goal_xy_progress,
            "inside_goal_zone": self._inside_goal_zone,
            "goal_dwell_steps": self._goal_dwell_steps,
            "required_goal_dwell_steps": self._required_goal_dwell_steps,
            "goal_dwell_fraction": self._goal_dwell_fraction,
            "goal_dwell_time_sec": self._goal_dwell_steps * self.config.step_dt_sim_sec,
            "required_goal_dwell_time_sec": self._required_goal_dwell_steps * self.config.step_dt_sim_sec,
            "goal_rel_x": goal_rel_x,
            "goal_rel_y": goal_rel_y,
            "goal_rel_z": goal_rel_z,
            "signed_z_err": signed_z_err,
            "home_xy_err": home_xy_err,
            "speed_xy": speed_xy,
            "speed_z": speed_z,
            "cmd_roll_rate": float(cmd[0]),
            "cmd_pitch_rate": float(cmd[1]),
            "cmd_yaw_rate": float(cmd[2]),
            "cmd_thrust": float(cmd[3]),
            "goal": None if goal is None else (goal.x, goal.y, goal.z),
            "home": None if home is None else (home.x, home.y, home.z),
            **self._last_reward_terms,
        }

    def _near_home_and_slow(self) -> bool:
        assert self.agent is not None
        if self.agent.state.home is None:
            return False
        obs = self.agent.get_observation()
        home = self.agent.state.home
        xy_err = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
        z_err = abs(float(obs.z) - home.z)
        speed_xy = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2)
        speed_z = abs(float(obs.vz))
        return (
            xy_err <= self.config.recover_tolerance_m
            and z_err <= self.config.recover_z_tolerance_m
            and speed_xy <= 0.20
            and speed_z <= 0.20
        )

    def _print_reset_ready_state(self) -> None:
        info = self._build_info("reset")
        print(
            f"[PPO RESET READY] episode={self._episode_id}, "
            f"goal_xy_err={info['xy_err']:.2f}, z_err={info['z_err']:.2f}, "
            f"signed_z_err={info['signed_z_err']:.2f}, "
            f"speed_xy={info['speed_xy']:.2f}, speed_z={info['speed_z']:.2f}, "
            f"inside_goal_zone={info['inside_goal_zone']}, "
            f"dwell={info['goal_dwell_steps']}/{info['required_goal_dwell_steps']}"
        )

    def _print_done_state(self, reason: str) -> None:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        goal = self.agent.state.goal
        if home is None or goal is None:
            print(f"[PPO ENV DONE] episode={self._episode_id}, step={self._step_id}, reason={reason}")
            return
        xy_err = math.sqrt((float(obs.x) - goal.x) ** 2 + (float(obs.y) - goal.y) ** 2)
        z_err = abs(float(obs.z) - goal.z)
        signed_z_err = float(obs.z) - goal.z
        goal_rel_x = goal.x - float(obs.x)
        goal_rel_y = goal.y - float(obs.y)
        cmd = self.agent.state.prev_action
        print(
            f"[PPO ENV DONE] episode={self._episode_id}, step={self._step_id}, reason={reason}, "
            f"goal_xy_err={xy_err:.2f}, z_err={z_err:.2f}, signed_z_err={signed_z_err:.2f}, "
            f"goal_rel=({goal_rel_x:.2f},{goal_rel_y:.2f}), "
            f"vx={obs.vx:.2f}, vy={obs.vy:.2f}, vz={obs.vz:.2f}, "
            f"roll={obs.roll:.3f}, pitch={obs.pitch:.3f}, yaw={obs.yaw:.3f}, "
            f"inside_goal_zone={self._inside_goal_zone}, "
            f"dwell={self._goal_dwell_steps}/{self._required_goal_dwell_steps}, "
            f"cmd_rate=({cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}), cmd_thrust={cmd[3]:.3f}"
        )
