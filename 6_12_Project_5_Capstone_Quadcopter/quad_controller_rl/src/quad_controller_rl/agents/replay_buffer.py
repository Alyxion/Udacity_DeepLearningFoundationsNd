import random

from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def test_run():
    """Test run ReplayBuffer implementation."""
    buf = ReplayBuffer(10)  # small buffer to test with
    
    # Add some sample data with a known pattern:
    #     state: i, action: 0/1, reward: -1/0/1, next_state: i+1, done: 0/1
    for i in range(15):  # more than maximum size to force overwriting
        buf.add(i, i % 2, i % 3 - 1, i + 1, i % 4)
    
    # Print buffer size and contents
    print("Replay buffer: size =", len(buf))  # maximum size if full
    for i, e in enumerate(buf.memory):
        print(i, e)  # should show circular overwriting
    
    # Randomly sample a batch
    batch = buf.sample(5)
    print("\nRandom batch: size =", len(batch))  # maximum size if full
    for e in batch:
        print(e)
