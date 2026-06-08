"""
Minimal MAPPO runner for the real-time Pegasus/PX4 safe-hover task.

This avoids the StarCraft-specific runner assumptions and avoids wandb.
It uses the existing R_MAPPOPolicy, R_MAPPO trainer, and SharedReplayBuffer.
"""

from __future__ import annotations
from collections import Counter
import time
from pathlib import Path

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class PegasusHoverRunner:
    def __init__(self, all_args, envs, device, run_dir: Path):
        self.all_args = all_args
        self.envs = envs
        self.device = device
        self.run_dir = Path(run_dir)
        self.num_agents = envs.num_agents
        self.n_rollout_threads = all_args.n_rollout_threads
        self.episode_length = all_args.episode_length
        self.num_env_steps = int(all_args.num_env_steps)
        self.recurrent_N = all_args.recurrent_N
        self.hidden_size = all_args.hidden_size

        self.save_dir = self.run_dir / "models"
        self.log_dir = self.run_dir / "logs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        share_obs_space = (
            envs.share_observation_space[0]
            if all_args.use_centralized_V
            else envs.observation_space[0]
        )

        print("obs_space:", envs.observation_space)
        print("share_obs_space:", envs.share_observation_space)
        print("action_space:", envs.action_space)

        self.policy = R_MAPPOPolicy(
            all_args,
            envs.observation_space[0],
            share_obs_space,
            envs.action_space[0],
            device=device,
        )

        # Critical safety setting for real drones/SITL:
        # default Gaussian std is 1.0, which is much too aggressive.
        # 0.10~0.15 keeps early exploration close to hover.
        init_std = float(getattr(all_args, "init_action_std", 0.12))
        try:
            self.policy.actor.act.action_out.logstd._bias.data.fill_(np.log(init_std))
            print(f"[PegasusHoverRunner] initial Gaussian action std set to {init_std}")
        except Exception as e:
            print(f"[PegasusHoverRunner] warning: failed to set initial action std: {e}")

        self.trainer = R_MAPPO(all_args, self.policy, device=device)

        self.buffer = SharedReplayBuffer(
            all_args,
            self.num_agents,
            envs.observation_space[0],
            share_obs_space,
            envs.action_space[0],
        )

    def run(self):
        obs, share_obs, reset_infos = self.envs.reset()
        if not self.all_args.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

        total_updates = max(
            1,
            self.num_env_steps // self.episode_length // self.n_rollout_threads,
        )
        start_time = time.time()

        for update in range(total_updates):
            rollout_stats = self._new_rollout_stats()

            if self.all_args.use_linear_lr_decay:
                self.policy.lr_decay(update, total_updates)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # MAPPO Gaussian can produce values outside [-1, 1].
                # Env also clips, but clipping here keeps logs and safety behavior more interpretable.
                env_actions = np.clip(actions, -1.0, 1.0)

                next_obs, next_share_obs, rewards, dones, infos = self.envs.step(env_actions)

                if not self.all_args.use_centralized_V:
                    next_share_obs = next_obs

                # New logging/statistics path.
                # This records true step/terminal info returned by the vec env.
                self._update_rollout_stats(
                    rollout_stats,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                )

                self.insert(
                    next_obs,
                    next_share_obs,
                    rewards,
                    dones,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

            self.compute()
            train_info = self.train()

            total_steps = (update + 1) * self.episode_length * self.n_rollout_threads

            if update % self.all_args.save_interval == 0 or update == total_updates - 1:
                self.save()

            if update % self.all_args.log_interval == 0 or update == total_updates - 1:
                elapsed = max(time.time() - start_time, 1e-6)
                fps = int(total_steps / elapsed)

                self._print_rollout_stats(
                    rollout_stats,
                    episode=update,
                    episodes=total_updates,
                    total_num_steps=total_steps,
                    train_infos=train_info,
                    fps=fps,
                )

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            available_actions=None,
            deterministic=False,
        )

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(
        self,
        obs,
        share_obs,
        rewards,
        dones,
        values,
        actions,
        action_log_probs,
        rnn_states,
        rnn_states_critic,
    ):
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones] = 0.0
        active_masks[dones_env] = 1.0

        # We handle timeouts as normal curriculum success, not bad transitions.
        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        if not self.all_args.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions=None,
        )

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_info

    def save(self):
        torch.save(self.policy.actor.state_dict(), str(self.save_dir / "actor.pt"))
        torch.save(self.policy.critic.state_dict(), str(self.save_dir / "critic.pt"))

    def _new_rollout_stats(self):
        return {
            "terminal_done_reasons": Counter(),
            "step_done_reasons": Counter(),
            "goal_dists": [],
            "goal_xy_errs": [],
            "z_errs": [],
            "inter_dists": [],
            "min_inter_drone_distance": float("inf"),
            "max_goal_dist": 0.0,
            "max_goal_xy_err": 0.0,
            "max_z_err": 0.0,
            "episode_rewards": [],
            "terminal_count": 0,
        }


    def _update_rollout_stats(self, stats, rewards, dones, infos):
        """
        infos shape:
            list length n_rollout_threads
            each item: list[num_agents] of info dict
        """
        stats["episode_rewards"].append(float(np.mean(rewards)))

        for env_i, env_infos in enumerate(infos):
            env_done = bool(np.all(dones[env_i]))

            if not isinstance(env_infos, (list, tuple)):
                continue

            env_reasons = []

            for agent_info in env_infos:
                if not isinstance(agent_info, dict):
                    continue

                reason = agent_info.get("done_reason", "unknown")
                stats["step_done_reasons"][reason] += 1
                env_reasons.append(reason)

                goal_dist = agent_info.get("goal_distance", None)
                if goal_dist is not None:
                    goal_dist = float(goal_dist)
                    stats["goal_dists"].append(goal_dist)
                    stats["max_goal_dist"] = max(stats["max_goal_dist"], goal_dist)

                goal_xy_err = agent_info.get("goal_xy_err", None)
                if goal_xy_err is not None:
                    goal_xy_err = float(goal_xy_err)
                    stats["goal_xy_errs"].append(goal_xy_err)
                    stats["max_goal_xy_err"] = max(stats["max_goal_xy_err"], goal_xy_err)

                z_err = agent_info.get("z_err", None)
                if z_err is not None:
                    z_err = float(z_err)
                    stats["z_errs"].append(z_err)
                    stats["max_z_err"] = max(stats["max_z_err"], z_err)

                d12 = agent_info.get("inter_drone_distance", None)
                if d12 is not None:
                    d12 = float(d12)
                    stats["inter_dists"].append(d12)
                    stats["min_inter_drone_distance"] = min(stats["min_inter_drone_distance"], d12)

            if env_done:
                stats["terminal_count"] += 1

                # Prefer non-running/non-reset reason.
                terminal_reason = "unknown"
                for r in env_reasons:
                    if r not in ["running", "reset"]:
                        terminal_reason = r
                        break

                if terminal_reason == "unknown" and len(env_reasons) > 0:
                    terminal_reason = env_reasons[0]

                stats["terminal_done_reasons"][terminal_reason] += 1


    def _print_rollout_stats(self, stats, episode, episodes, total_num_steps, train_infos, fps):
        mean_reward = float(np.mean(stats["episode_rewards"])) if stats["episode_rewards"] else 0.0
        mean_goal_dist = float(np.mean(stats["goal_dists"])) if stats["goal_dists"] else 0.0
        max_goal_dist = float(stats["max_goal_dist"]) if stats["goal_dists"] else 0.0
        mean_goal_xy_err = float(np.mean(stats["goal_xy_errs"])) if stats["goal_xy_errs"] else 0.0
        max_goal_xy_err = float(stats["max_goal_xy_err"]) if stats["goal_xy_errs"] else 0.0
        mean_z_err = float(np.mean(stats["z_errs"])) if stats["z_errs"] else 0.0
        max_z_err = float(stats["max_z_err"]) if stats["z_errs"] else 0.0
        mean_d12 = float(np.mean(stats["inter_dists"])) if stats["inter_dists"] else 0.0
        min_d12 = (
            float(stats["min_inter_drone_distance"])
            if stats["min_inter_drone_distance"] < float("inf")
            else 0.0
        )

        print("=" * 80)
        print(
            f"[PegasusHover] update {episode + 1}/{episodes}, "
            f"steps {total_num_steps}/{self.num_env_steps}, FPS={fps}"
        )
        print(f"  mean_rollout_reward: {mean_reward:.4f}")
        print(f"  mean_goal_dist: {mean_goal_dist:.3f}")
        print(f"  max_goal_dist: {max_goal_dist:.3f}")
        print(f"  mean_goal_xy_err: {mean_goal_xy_err:.3f}")
        print(f"  max_goal_xy_err: {max_goal_xy_err:.3f}")
        print(f"  mean_z_err: {mean_z_err:.3f}")
        print(f"  max_z_err: {max_z_err:.3f}")
        print(f"  mean_inter_drone_distance: {mean_d12:.3f}")
        print(f"  min_inter_drone_distance: {min_d12:.3f}")
        print(f"  terminal_done_reasons: {dict(stats['terminal_done_reasons'])}")
        print(f"  step_done_reasons: {dict(stats['step_done_reasons'])}")

        for k, v in train_infos.items():
            try:
                print(f"  {k}: {float(v):.6f}")
            except Exception:
                print(f"  {k}: {v}")
