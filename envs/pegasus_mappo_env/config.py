from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits


@dataclass
class TwoDroneEnvConfig:
    # MAVLink endpoints created by Pegasus/PX4. Adjust to your launch setup.
    connection_strs: Tuple[str, str] = ("udp:0.0.0.0:14540", "udp:0.0.0.0:14541")
    target_systems: Tuple[int, int] = (1, 2)

    # Data/control frequencies.
    data_stream_hz: int = 30
    ctbr_send_hz: int = 30
    step_dt_sim_sec: float = 0.05
    episode_length: int = 300

    # Initial takeoff / reset.
    takeoff_altitudes: Tuple[float, float] = (5.0, 9.0)
    takeoff_timeout_sim_sec: float = 40.0
    stabilize_after_takeoff_sim_sec: float = 1.5
    recover_timeout_sim_sec: float = 12.0
    recover_tolerance_m: float = 0.60
    auto_takeoff_on_first_reset: bool = True
    start_logging: bool = True
    log_dir: str = "./log_folder"

    # Goal sampling around each drone's home point, in local NED coordinates.
    goal_scenario: str = "random"
    world_xy_offsets: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
    goal_xy_radius_min: float = 1.5
    goal_xy_radius_max: float = 5.0
    goal_z_delta_max: float = 0.5
    cross_goal_distance_m: float = 2.0
    fixed_goals: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    # Success/collision thresholds.
    goal_tolerance_m: float = 0.6
    collision_distance_m: float = 1.0
    warning_distance_m: float = 2.5

    # Reward weights. Keep conservative first; tune after smoke tests.
    reward_progress_scale: float = 2.0
    reward_distance_scale: float = 0.05
    reward_alive: float = 0.05
    reward_control_scale: float = 0.05
    reward_close_penalty_scale: float = 0.25
    reward_success: float = 10.0
    reward_collision: float = -30.0
    reward_crash: float = -30.0
    reward_timeout: float = 2.0

    action_limits: CTBRActionLimits = field(default_factory=CTBRActionLimits)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
