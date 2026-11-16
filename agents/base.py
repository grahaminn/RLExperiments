from abc import ABC, abstractmethod

import numpy as np
import torch

from .replay_buffer import ReplayBuffer

# use GPU if available, or MPS, otherwise, use CPU
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')


class Agent(ABC):
    """Base class for every deep Q-learning method in this project.

    All agents are used the same way:

        agent = SomeAgent(env, seed=..., **hyperparameters)
        rewards = agent.learn(max_episodes)   # tensor of discounted reward per episode

    The training loop (replay buffer, optional random warm-up, target-network updates,
    reward bookkeeping) lives here and is shared. A new method only has to provide:

        build_network()                       -> the online / target network (an nn.Module
                                                 with a train_step(transitions, gamma, target)
                                                 method)
        select_action(encoded_state, episode) -> the action to take in that state

    and may override begin_training(max_episodes) to set up per-run schedules.
    """

    def __init__(self, env, seed, gamma, learning_rate, batch_size, target_update_frequency,
                 buffer_size, kernel_size, conv_out_channels, fc_out_features, learning_starts=0):
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.kernel_size = kernel_size
        self.conv_out_channels = conv_out_channels
        self.fc_out_features = fc_out_features
        # learning_starts: number of initial agent steps that take uniformly random actions,
        # with no training until then (Nikolov et al., 2019, use 50,000 on Atari). Without it,
        # actions the agent never tries keep their random initial values, and IDS can get stuck
        # repeating one move (e.g. walking into a wall) for the whole run.
        self.learning_starts = learning_starts
        self.n_actions = env.n_actions

        self.random_state = np.random.RandomState(seed)
        self.replay_buffer = ReplayBuffer(buffer_size, self.random_state)
        self.warmup_rng = np.random.RandomState(10_000 + seed)  # separate RNG: other sampling is unchanged
        self.total_steps = 0

        self.network = self.build_network().to(device)
        self.target_network = self.build_network().to(device)
        self.target_network.requires_grad_(False)

    @abstractmethod
    def build_network(self):
        """Return a new network. Called twice: once for the online and once for the target network."""

    @abstractmethod
    def select_action(self, encoded_state, episode):
        """Return the action to take in encoded_state during the given episode (after any warm-up)."""

    def begin_training(self, max_episodes):
        """Hook called at the start of learn(), e.g. to build an exploration schedule."""

    def learn(self, max_episodes):
        self.begin_training(max_episodes)
        env = self.env
        sums_of_rewards = []

        for i in range(max_episodes):
            state, encoded_state = env.reset()
            sum_of_discounted_rewards = 0.0
            terminated = False
            truncated = False
            t = 0

            while not (terminated or truncated):
                if self.total_steps < self.learning_starts:
                    action = self.warmup_rng.randint(self.n_actions)
                else:
                    action = self.select_action(encoded_state, i)

                next_state, encoded_next_state, reward, terminated, truncated, info = env.step(action)

                sum_of_discounted_rewards += self.gamma ** t * reward
                self.replay_buffer.append((encoded_state, action, reward, encoded_next_state, terminated, truncated))
                state = next_state
                encoded_state = encoded_next_state
                t += 1
                self.total_steps += 1

                if len(self.replay_buffer) >= self.batch_size and self.total_steps >= self.learning_starts:
                    transitions = self.replay_buffer.draw(self.batch_size)
                    self.network.train_step(transitions, self.gamma, self.target_network)

            sums_of_rewards.append(sum_of_discounted_rewards)

            if (i % self.target_update_frequency) == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        return torch.tensor(sums_of_rewards)
