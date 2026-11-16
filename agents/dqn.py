import numpy as np
import torch

from .base import Agent, device


class DeepQNetwork(torch.nn.Module):
    def __init__(self, env, learning_rate, kernel_size, conv_out_channels,
                 fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)
        self.random_state = np.random.RandomState(seed)
        self.n_actions = env.n_actions

        self.conv_layer = torch.nn.Conv2d(in_channels=env.state_shape[0], out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        # Get the height and width
        h = env.h
        w = env.w

        h = h - kernel_size + 1
        w = w - kernel_size + 1

        self.fc_layer = torch.nn.Linear(in_features=h * w * conv_out_channels, out_features=fc_out_features)
        self.output_layer = torch.nn.Linear(in_features=fc_out_features, out_features=self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, foreach=True)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = torch.relu(self.conv_layer(x)).flatten(1)
        x = torch.relu(self.fc_layer(x))
        return self.output_layer(x)

    def train_step(self, transitions, gamma, tdqn):
        # Batched arrays from ReplayBuffer.draw, already float32 / int64, so
        # torch.from_numpy wraps them without copying or converting.
        states, actions, rewards, next_states, terminateds, truncateds = transitions

        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        not_done = torch.from_numpy((1 - terminateds) * (1 - truncateds)).to(device)

        q = self(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * not_done

        target = rewards + gamma * next_q
        loss = torch.nn.functional.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_greedy(self, agent_net):
        best = np.flatnonzero(np.isclose(agent_net, agent_net.max()))
        return self.random_state.choice(best)


class DQNAgent(Agent):
    """DQN with epsilon-greedy exploration; epsilon decays linearly from `epsilon` to 0
    over the run."""

    name = 'DQN'

    def __init__(self, env, seed, epsilon, **kwargs):
        self.epsilon = epsilon
        super().__init__(env, seed, **kwargs)

    def build_network(self):
        return DeepQNetwork(self.env, self.learning_rate, self.kernel_size, self.conv_out_channels,
                            self.fc_out_features, seed=self.seed)

    def begin_training(self, max_episodes):
        self.epsilons = np.linspace(self.epsilon, 0, max_episodes)

    def select_action(self, encoded_state, episode):
        if self.random_state.rand() < self.epsilons[episode]:
            return self.random_state.choice(self.n_actions)
        with torch.inference_mode():
            q = self.network(np.asarray(encoded_state)[None])[0].cpu().numpy()
        return self.network.eval_greedy(q)
