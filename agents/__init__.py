"""Deep Q-learning agents, one method per file.

Every agent is a subclass of Agent and is used the same way:

    agent = DQNAgent(env, seed=0, **hyperparameters)
    rewards = agent.learn(max_episodes)

To add a method, create a new file here with an Agent subclass that implements
build_network() and select_action(encoded_state, episode), and import it below.
"""
from .base import Agent, device
from .replay_buffer import ReplayBuffer
from .dqn import DeepQNetwork, DQNAgent
from .ids_dqn import BaseBstrapDQN, IDSDQN, IDSDQNAgent
