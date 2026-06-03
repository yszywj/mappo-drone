from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64, init_std: float = 0.05):
        super().__init__()
        self.actor_body = mlp(obs_dim, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.critic_body = mlp(obs_dim, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(init_std)), dtype=torch.float32))

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_body(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return log_prob, entropy, value
