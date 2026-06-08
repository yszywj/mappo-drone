#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from typing import Tuple

import numpy as np
import torch

from envs.mavlink_ctbr_controller.rl_bridge import CTBRActionLimits, SafetyLimits
from envs.pegasus_ppo_env.config import SingleDroneEnvConfig
from envs.pegasus_ppo_env.single_drone_hover_env import SingleDroneHoverEnv


def parse_args():
    parser = argparse.ArgumentParser("Probe CTBR roll/pitch rate axis responses")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--pulse_rate", type=float, default=0.025)
    parser.add_argument("--pulse_duration", type=float, default=0.8)
    parser.add_argument("--settle_duration", type=float, default=1.0)
    parser.add_argument("--ctbr_send_hz", type=int, default=20)
    parser.add_argument("--data_stream_hz", type=int, default=20)
    parser.add_argument("--takeoff_altitude", type=float, default=5.0)
    parser.add_argument("--connection_str", type=str, default="udp:0.0.0.0:14540")
    parser.add_argument("--target_system", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
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
        residual_gain=0.0,
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
        step_dt_sim_sec=0.5,
        episode_length=80,
        takeoff_altitude=args.takeoff_altitude,
        stabilize_after_takeoff_sim_sec=5.0,
        recover_timeout_sim_sec=25.0,
        recover_tolerance_m=0.5,
        start_logging=not args.no_pegasus_log,
        log_dir=args.pegasus_log_dir,
        action_limits=action_limits,
        safety_limits=safety_limits,
    )
    return SingleDroneHoverEnv(cfg, seed=args.seed)


def wait(env: SingleDroneHoverEnv, seconds: float) -> None:
    assert env.time_keeper is not None
    env.time_keeper.wait(seconds, timeout=max(2.0, seconds + 2.0))


def send_raw_ctbr(env: SingleDroneHoverEnv, roll_rate: float, pitch_rate: float, yaw_rate: float = 0.0) -> None:
    assert env.agent is not None
    env.agent.controller.update_ctbr_send_params(
        body_roll_rate=roll_rate,
        body_pitch_rate=pitch_rate,
        body_yaw_rate=yaw_rate,
        thrust=env.config.action_limits.hover_thrust,
    )
    env.agent.state.prev_action = np.array(
        [roll_rate, pitch_rate, yaw_rate, env.config.action_limits.hover_thrust],
        dtype=np.float32,
    )


def obs_tuple(env: SingleDroneHoverEnv) -> Tuple[float, float, float, float, float, float, float, float, float]:
    assert env.agent is not None
    obs = env.agent.get_observation()
    return (
        float(obs.x),
        float(obs.y),
        float(obs.z),
        float(obs.vx),
        float(obs.vy),
        float(obs.vz),
        float(obs.roll),
        float(obs.pitch),
        float(obs.yaw),
    )


def recover_and_settle(env: SingleDroneHoverEnv, args) -> None:
    assert env.agent is not None
    if not env._near_home_and_slow():
        ok = env._recover_to_home()
        if not ok:
            raise RuntimeError("axis probe failed: recover_to_home failed")
    env.agent.start_ctbr(args.ctbr_send_hz)
    send_raw_ctbr(env, 0.0, 0.0, 0.0)
    wait(env, args.settle_duration)


def run_probe(env: SingleDroneHoverEnv, args, name: str, roll_rate: float, pitch_rate: float) -> None:
    recover_and_settle(env, args)
    x0, y0, z0, vx0, vy0, vz0, roll0, pitch0, yaw0 = obs_tuple(env)

    send_raw_ctbr(env, roll_rate, pitch_rate, 0.0)
    wait(env, args.pulse_duration)
    x1, y1, z1, vx1, vy1, vz1, roll1, pitch1, yaw1 = obs_tuple(env)

    send_raw_ctbr(env, 0.0, 0.0, 0.0)
    wait(env, args.settle_duration)

    print(
        "[CTBR AXIS PROBE] "
        f"{name}: cmd_roll_rate={roll_rate:+.3f}, cmd_pitch_rate={pitch_rate:+.3f}, "
        f"yaw0={yaw0:+.3f}, yaw1={yaw1:+.3f}, "
        f"dx={x1 - x0:+.3f}, dy={y1 - y0:+.3f}, dz={z1 - z0:+.3f}, "
        f"dvx={vx1 - vx0:+.3f}, dvy={vy1 - vy0:+.3f}, dvz={vz1 - vz0:+.3f}, "
        f"roll={roll0:+.3f}->{roll1:+.3f}, pitch={pitch0:+.3f}->{pitch1:+.3f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(1)

    env = make_env(args)
    try:
        print("=" * 80)
        print("Pegasus CTBR axis probe")
        print(f"pulse_rate: {args.pulse_rate}")
        print(f"pulse_duration: {args.pulse_duration}")
        print(f"settle_duration: {args.settle_duration}")
        print("=" * 80)

        env.reset()
        rate = abs(float(args.pulse_rate))
        for episode_idx in range(1, args.episodes + 1):
            print(f"[CTBR AXIS PROBE] batch={episode_idx}")
            run_probe(env, args, "+roll", +rate, 0.0)
            run_probe(env, args, "-roll", -rate, 0.0)
            run_probe(env, args, "+pitch", 0.0, +rate)
            run_probe(env, args, "-pitch", 0.0, -rate)
    except KeyboardInterrupt:
        print("\n[CTBR AXIS PROBE] interrupted, closing environment...")
    finally:
        env.close()
        print("[CTBR AXIS PROBE] environment closed")


if __name__ == "__main__":
    main()
