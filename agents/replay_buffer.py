import numpy as np


class ReplayBuffer:
    """Circular replay buffer stored in preallocated numpy arrays.

    draw() returns a tuple of batched arrays (states, actions, rewards,
    next_states, terminateds, truncateds), ready for torch.from_numpy.
    Index i counts from the oldest stored transition; batches are drawn with
    replacement from the global numpy RNG (seeded once per run).
    """

    def __init__(self, buffer_size, random_state):
        self.buffer_size = buffer_size
        self.random_state = random_state
        self.size = 0   # number of transitions stored
        self.pos = 0    # next slot to write; once full, also the oldest transition
        self.states = None  # allocated on the first append, once the state shape is known

    def __len__(self):
        return self.size

    def append(self, transition):
        state, action, reward, next_state, terminated, truncated = transition
        if self.states is None:
            shape = (self.buffer_size,) + np.shape(state)
            self.states = np.empty(shape, dtype=np.float32)
            self.next_states = np.empty(shape, dtype=np.float32)
            self.actions = np.empty(self.buffer_size, dtype=np.int64)
            self.rewards = np.empty(self.buffer_size, dtype=np.float32)
            self.terminateds = np.empty(self.buffer_size, dtype=np.float32)
            self.truncateds = np.empty(self.buffer_size, dtype=np.float32)
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminateds[i] = terminated
        self.truncateds[i] = truncated
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def draw(self, batch_size):
        # Ensure the batch size does not exceed the buffer size
        batch_size = min(batch_size, self.size)
        # Random sample (with replacement), indexed oldest-first
        indices = np.random.choice(self.size, size=batch_size)
        if self.size == self.buffer_size:
            indices = (indices + self.pos) % self.buffer_size
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.terminateds[indices], self.truncateds[indices])
