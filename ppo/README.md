# Single-drone Pegasus PPO

This folder contains a small single-agent PPO trainer for Pegasus/PX4 hover
experiments. It is intentionally independent from the existing MAPPO runner.

Example:

```bash
python -m ppo.train_pegasus_hover_ppo \
  --num_env_steps 800 \
  --rollout_steps 80 \
  --episode_length 40 \
  --step_dt_sim_sec 0.5 \
  --residual_gain 0.02 \
  --init_action_std 0.05
```

After the 800-step smoke test is stable, run a longer check:

```bash
python -m ppo.train_pegasus_hover_ppo \
  --num_env_steps 4000 \
  --rollout_steps 80 \
  --episode_length 40 \
  --step_dt_sim_sec 0.5 \
  --residual_gain 0.02 \
  --init_action_std 0.05
```

The policy is trained as a residual on top of the PD stabilizer implemented in
`envs/mavlink_ctbr_controller/rl_bridge.py`.

The single-drone observation is 25-dimensional and matches one MAPPO agent's
observation layout. The missing second drone is represented as a virtual copy at
the same pose/velocity, so `other_rel_pos` and `other_rel_vel` are zeros.

Deterministic evaluation uses the actor mean action instead of sampling from the
Gaussian policy. Use the same `residual_gain`, `episode_length`, and
`hidden_size` as the training run:

```bash
python -m ppo.eval_pegasus_hover_ppo \
  --checkpoint ./results/PegasusSingleDroneHover/ppo/seed1_YYYYMMDD_HHMMSS/models/actor_critic.pt \
  --episodes 5 \
  --episode_length 80 \
  --step_dt_sim_sec 0.5 \
  --residual_gain 0.05 \
  --init_action_std 0.05
```

To diagnose the raw CTBR roll/pitch axis signs, run:

```bash
python -m ppo.probe_ctbr_axes \
  --pulse_rate 0.025 \
  --pulse_duration 0.8 \
  --settle_duration 1.0
```
