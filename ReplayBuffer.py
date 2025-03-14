from collections import deque, namedtuple
import random as rand

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'nextState', 'reward'))

class ReplayBuffer(object):
    def __init__(self, seed, capacity):
        # Seeding
        rand.seed(seed)
        
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batchSize):
        return rand.sample(self.buffer, batchSize)

    def __len__(self):
        return len(self.buffer)