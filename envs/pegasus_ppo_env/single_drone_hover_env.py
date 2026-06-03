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
    observation_vector,
)
from .config import SingleDroneEnvConfig


class SingleDroneHoverEnv(gym.Env if hasattr(gym, "Env") else object):
    """
    Single-drone safe-hover task for PPO residual learning.

    Observation layout, length 25, matching one MAPPO agent:
      own pos(3), own vel(3), own attitude(3), own body rates(3),
      goal relative pos(3), other relative pos(3), other relative vel(3),
      prev CTBR action(4).

    For single-drone PPO, "other" is a virtual copy at the same pose/velocity,
    so other relative pos/vel are zeros. This keeps the actor input compatible
    with MAPPO while excluding multi-drone effects.
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
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.controller: Optional[CTBRController] = None
        self.agent: Optional[CTBRDroneRLAdapter] = None
        self.time_keeper = None

        self.obs_dim = 25
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
        home = self.agent.state.home
        assert home is not None
        self.agent.set_goal(GoalPoint(home.x, home.y, home.z))
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

        obs = self._build_obs()
        reward, done, done_reason = self._compute_reward_done(action, ok_time=ok_time)
        info = self._build_info(done_reason)

        if done:
            self._print_done_state(done_reason)
            normal_episode_end = done_reason == "timeout"
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

    def _build_obs(self) -> np.ndarray:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        if home is None:
            raise RuntimeError("home must be set before building observation")
        goal = self.agent.state.goal or GoalPoint(home.x, home.y, home.z)
        return observation_vector(
            own=obs,
            other=obs,
            goal=goal,
            prev_action=self.agent.state.prev_action,
        )

    def _compute_reward_done(self, action, ok_time: bool) -> Tuple[float, bool, str]:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        if home is None:
            raise RuntimeError("home is not set during reward computation")

        reward = 0.0
        done = False
        done_reason = "running"

        if not ok_time:
            done = True
            done_reason = "sim_time_timeout"
            reward += self.config.reward_crash

        safety = self.agent.check_single_drone_safety()
        if safety.abnormal:
            done = True
            done_reason = safety.reason
            reward += self.config.reward_crash

        xy_err = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
        z_err = abs(float(obs.z) - home.z)
        speed = math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2 + float(obs.vz) ** 2)
        tilt = math.sqrt(float(obs.roll) ** 2 + float(obs.pitch) ** 2)
        control_penalty = float(np.mean(np.square(np.clip(action, -1.0, 1.0))))

        reward += self.config.reward_alive
        reward -= 0.20 * xy_err
        reward -= 0.45 * z_err
        reward -= 0.04 * speed
        reward -= 0.08 * tilt
        reward -= self.config.reward_control_scale * control_penalty

        if self._step_id >= self.config.episode_length and not done:
            done = True
            done_reason = "timeout"
            reward += self.config.reward_timeout

        return reward, done, done_reason

    def _build_info(self, done_reason: str) -> Dict[str, Any]:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        xy_err = None
        z_err = None
        if home is not None:
            xy_err = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
            z_err = abs(float(obs.z) - home.z)
        return {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "done_reason": done_reason,
            "xy_err": xy_err,
            "z_err": z_err,
            "speed_xy": math.sqrt(float(obs.vx) ** 2 + float(obs.vy) ** 2),
            "home": None if home is None else (home.x, home.y, home.z),
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
            and z_err <= 0.5
            and speed_xy <= 0.20
            and speed_z <= 0.20
        )

    def _print_reset_ready_state(self) -> None:
        info = self._build_info("reset")
        print(
            f"[PPO RESET READY] episode={self._episode_id}, "
            f"xy_err={info['xy_err']:.2f}, z_err={info['z_err']:.2f}, "
            f"speed_xy={info['speed_xy']:.2f}"
        )

    def _print_done_state(self, reason: str) -> None:
        assert self.agent is not None
        obs = self.agent.get_observation()
        home = self.agent.state.home
        if home is None:
            print(f"[PPO ENV DONE] episode={self._episode_id}, step={self._step_id}, reason={reason}")
            return
        xy_err = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
        z_err = abs(float(obs.z) - home.z)
        cmd = self.agent.state.prev_action
        print(
            f"[PPO ENV DONE] episode={self._episode_id}, step={self._step_id}, reason={reason}, "
            f"xy_err={xy_err:.2f}, z_err={z_err:.2f}, "
            f"vx={obs.vx:.2f}, vy={obs.vy:.2f}, vz={obs.vz:.2f}, "
            f"roll={obs.roll:.3f}, pitch={obs.pitch:.3f}, yaw={obs.yaw:.3f}, "
            f"cmd_rate=({cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}), cmd_thrust={cmd[3]:.3f}"
        )
