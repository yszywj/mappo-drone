from __future__ import annotations

from dataclasses import dataclass, field

from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits


@dataclass
class SingleDroneEnvConfig:
    connection_str: str = "udp:0.0.0.0:14540"
    target_system: int = 1

    data_stream_hz: int = 20
    ctbr_send_hz: int = 20
    step_dt_sim_sec: float = 0.5
    episode_length: int = 40

    takeoff_altitude: float = 5.0
    takeoff_timeout_sim_sec: float = 40.0
    stabilize_after_takeoff_sim_sec: float = 5.0
    recover_timeout_sim_sec: float = 25.0
    recover_tolerance_m: float = 0.5
    auto_takeoff_on_first_reset: bool = True

    start_logging: bool = True
    log_dir: str = "./log_folder"

    # Goal sampling around the drone's home point, in local NED coordinates.
    goal_xy_radius_min: float = 0.0
    goal_xy_radius_max: float = 0.0
    goal_z_delta_max: float = 0.0
    goal_tolerance_m: float = 0.25
    goal_z_tolerance_m: float = 0.35
    goal_speed_xy_tolerance_mps: float = 0.25
    goal_speed_z_tolerance_mps: float = 0.25

    reward_alive: float = 0.05
    reward_progress_scale: float = 3.0
    reward_distance_scale: float = 0.10
    reward_z_scale: float = 0.20
    reward_control_scale: float = 0.05
    reward_success: float = 8.0
    reward_crash: float = -30.0
    reward_timeout: float = 2.0

    action_limits: CTBRActionLimits = field(default_factory=CTBRActionLimits)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
