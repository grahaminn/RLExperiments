import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent, device


class BaseBstrapDQN(nn.Module):

    def __init__(self, env, huber_loss, n_heads, n_actions, gamma, learning_rate,
                 kernel_size, conv_out_channels, fc_out_features, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.huber_loss = huber_loss
        self.n_heads = n_heads
        self.gamma = gamma

        self.conv_layer = nn.Conv2d(in_channels=env.state_shape[0], out_channels=conv_out_channels,
                                    kernel_size=kernel_size, stride=1)

        h = env.h - kernel_size + 1
        w = env.w - kernel_size + 1

        # All heads are evaluated in one batched matmul per layer instead of a Python
        # loop over separate nn.Linear modules. The weights are initialised by building
        # the per-head nn.Linear layers in the same order as before and stacking them,
        # so a given seed gives exactly the same starting network.
        fc_in = h * w * conv_out_channels
        heads = [(nn.Linear(fc_in, fc_out_features), nn.Linear(fc_out_features, n_actions))
                 for _ in range(n_heads)]
        self.head_w1 = nn.Parameter(torch.stack([l1.weight.T for l1, _ in heads]).detach().contiguous())  # [H, F, fc]
        self.head_b1 = nn.Parameter(torch.stack([l1.bias for l1, _ in heads]).detach().unsqueeze(1))     # [H, 1, fc]
        self.head_w2 = nn.Parameter(torch.stack([l2.weight.T for _, l2 in heads]).detach().contiguous())  # [H, fc, A]
        self.head_b2 = nn.Parameter(torch.stack([l2.bias for _, l2 in heads]).detach().unsqueeze(1))     # [H, 1, A]

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, foreach=True)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = torch.relu(self.conv_layer(x)).flatten(1)
        x = x.unsqueeze(0).expand(self.n_heads, -1, -1)                        # [H, B, F]
        x = torch.relu(torch.baddbmm(self.head_b1, x, self.head_w1))            # [H, B, fc]
        return torch.baddbmm(self.head_b2, x, self.head_w2).transpose(0, 1)     # [B, H, A]

    def compute_q_from_heads(self, q, actions):
        # q: [B, H, A]; actions: [B] (same action for every head) or [B, H] (one per head)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1).expand(-1, self.n_heads)
        return q.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, H]

    def select_action(self, q):
        return self.select_action_vote(q)

    def select_action_vote(self, q):
        votes = q.argmax(dim=-1)                                # [B, H]
        counts = F.one_hot(votes, self.n_actions).sum(dim=1)    # [B, A]
        return counts.argmax(dim=-1)

    def select_action_greedy(self, q):
        return q.mean(dim=1).argmax(dim=-1)

    def train_step(self, transitions, gamma, tdqn):
        # Batched arrays from ReplayBuffer.draw, already float32 / int64, so
        # torch.from_numpy wraps them without copying or converting.
        states, actions, rewards, next_states, terminateds, truncateds = transitions

        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device).unsqueeze(-1)
        not_done = torch.from_numpy((1 - terminateds) * (1 - truncateds)).to(device).unsqueeze(-1)

        q_values = self.compute_q_from_heads(self(states), actions)  # [B, H]

        with torch.no_grad():
            # Double DQN, per head: each online head picks its own greedy action
            # and the matching target head evaluates it.
            next_actions = self(next_states).argmax(dim=-1)                       # [B, H]
            next_q = tdqn.compute_q_from_heads(tdqn(next_states), next_actions)   # [B, H]
            target = rewards + gamma * next_q * not_done

        loss_fn = F.smooth_l1_loss if self.huber_loss else F.mse_loss
        loss = loss_fn(q_values, target)  # mean over batch and heads

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class IDSDQN(BaseBstrapDQN):

    def __init__(self, env, huber_loss, n_heads, n_actions, gamma, learning_rate,
                 kernel_size, conv_out_channels, fc_out_features, seed, rho, n_stds):
        super().__init__(env, huber_loss, n_heads, n_actions, gamma, learning_rate,
                         kernel_size, conv_out_channels, fc_out_features, seed)
        self.rho = rho
        self.n_stds = n_stds

    def select_action(self, q):
        return self.select_action_ids(q)

    def select_action_ids(self, q):
        mean = q.mean(dim=1)                    # [B, A]
        var = q.var(dim=1, correction=0)        # [B, A], spread across heads
        std = var.sqrt()
        regret = (mean + self.n_stds * std).max(dim=-1, keepdim=True)[0] - (mean - self.n_stds * std)
        info_gain = torch.log1p(var / self.rho) + 1e-5
        return (regret.square() / info_gain).argmin(dim=-1)


class IDSDQNAgent(Agent):
    """Bootstrapped DQN heads with information-directed sampling (Nikolov et al., 2019, DQN-IDS)."""

    name = 'IDS-DQN'

    def __init__(self, env, seed, n_stds, rho, n_heads=10, **kwargs):
        self.n_stds = n_stds
        self.rho = rho
        self.n_heads = n_heads  # number of bootstrap heads
        super().__init__(env, seed, **kwargs)

    def build_network(self):
        return IDSDQN(self.env, True, self.n_heads, self.n_actions, self.gamma, self.learning_rate,
                      self.kernel_size, self.conv_out_channels, self.fc_out_features, self.seed,
                      rho=self.rho, n_stds=self.n_stds)

    def select_action(self, encoded_state, episode):
        with torch.inference_mode():
            q = self.network(np.asarray(encoded_state)[None])   # [1, H, A]
            return self.network.select_action(q).item()
