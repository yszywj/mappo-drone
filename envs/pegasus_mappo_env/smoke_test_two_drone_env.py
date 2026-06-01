"""
Smoke test before connecting the environment to MAPPO.

Run from ~/mappo-drone:
    python -m envs.pegasus_mappo_env.smoke_test_two_drone_env

It uses a tiny hand-written policy and prints env-level signals.  Do this before
running MAPPO so you know reset/step/reward/done are valid.
"""

import math
import signal
import numpy as np

from .config import TwoDroneEnvConfig
from .two_drone_ctbr_env import TwoDroneCTBREnv


env = None


def _sig_handler(sig, frame):
    global env
    print("\n收到退出信号，正在关闭环境...")
    if env is not None:
        env.close()
    raise SystemExit(0)

def simple_policy(step_id: int):
    actions = []

    kp_xy = 0.04
    kd_xy = 0.03
    kp_z = 0.8

    max_xy_action = 0.05
    max_thrust_action = 0.3

    for agent in env.agents:
        obs = agent.get_observation()
        home = agent.state.home

        x_err = obs.x - home.x
        y_err = obs.y - home.y
        z_err = obs.z - home.z

        vx = obs.vx
        vy = obs.vy

        # 与上一版相反
        pitch_action = kp_xy * x_err + kd_xy * vx
        roll_action = -kp_xy * y_err - kd_xy * vy

        thrust_action = kp_z * z_err

        pitch_action = np.clip(pitch_action, -max_xy_action, max_xy_action)
        roll_action = np.clip(roll_action, -max_xy_action, max_xy_action)
        thrust_action = np.clip(thrust_action, -max_thrust_action, max_thrust_action)

        actions.append(
            np.array(
                [roll_action, pitch_action, 0.0, thrust_action],
                dtype=np.float32,
            )
        )

    return np.stack(actions, axis=0)

# def simple_policy(step_id: int):
#     obs = env.agents

#     actions = []
#     for agent in env.agents:
#         o = agent.get_observation()
#         home_z = agent.state.home.z

#         # NED: z 越负，高度越高
#         z_err = o.z - home_z

#         # z_err > 0 表示比 home 低，需要加推力
#         thrust_action = 0.8 * z_err
#         thrust_action = np.clip(thrust_action, -0.3, 0.3)

#         actions.append(np.array([0.0, 0.0, 0.0, thrust_action], dtype=np.float32))

#     return np.stack(actions, axis=0)

# def simple_policy(step_id: int):
#     # Policy actions in [-1, 1], NOT physical CTBR values.
#     t = 0.05 * step_id
#     a1 = np.array([0.15 * math.sin(t), -0.12 * math.cos(t), 0.0, 0.05], dtype=np.float32)
#     a2 = np.array([-0.15 * math.sin(t), 0.12 * math.cos(t), 0.0, 0.05], dtype=np.float32)
#     return np.stack([a1, a2], axis=0)

# def simple_policy(step_id: int):
#     a1 = np.array(
#         [0.0, 0.0, 0.0, 0.0],
#         dtype=np.float32
#     )

#     a2 = np.array(
#         [0.0, 0.0, 0.0, 0.0],
#         dtype=np.float32
#     )

#     return np.stack([a1, a2], axis=0)

def main():
    global env
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    cfg = TwoDroneEnvConfig(
        step_dt_sim_sec=0.5,
        episode_length=40,
        ctbr_send_hz=20,
        data_stream_hz=20,
        # goal_xy_radius_min=1.0,
        # goal_xy_radius_max=2.5,
        fixed_goals=(
            (2.0, 0.0, -6.0),   # drone1
            (-2.0, 0.0, -8.0),  # drone2
        ),
    )
    env = TwoDroneCTBREnv(cfg, seed=1)

    try:
        obs, share_obs, info = env.reset()
        print("reset ok")
        print("obs shape:", obs.shape, "share_obs shape:", share_obs.shape)
        print("info:", info)

        for step in range(cfg.episode_length):
            actions = simple_policy(step)
            obs, share_obs, rewards, dones, infos = env.step(actions)
            if step % 10 == 0 or dones.any():
                print(
                    f"step={step:04d}, reward={rewards.reshape(-1)}, "
                    f"done={dones}, reason={infos[0]['done_reason']}, "
                    f"goal_dist={[round(x['goal_distance'], 2) for x in infos]}, "
                    f"d12={infos[0]['inter_drone_distance']:.2f}"
                )
            if dones.any():
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
