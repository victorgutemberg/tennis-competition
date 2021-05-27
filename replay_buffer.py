from abc import abstractmethod
from collections import namedtuple, deque

import numpy as np
import random
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        '''Sample a batch of experiences from memory.'''

        indexes, experiences, is_weights = self._get_experices_sample()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        is_weights = torch.from_numpy(np.vstack(is_weights)).float().to(device)

        return indexes, (states, actions, rewards, next_states, dones), is_weights

    @abstractmethod
    def _get_experices_sample(self):
        raise NotImplementedError()

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, batch_size, buffer_size, seed):
        '''Initialize a UniformReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        '''
        super().__init__(action_size, batch_size, seed)
        self.memory = deque(maxlen=buffer_size)

    def batch_update(self, indexes, priorities):
        pass

    def _get_experices_sample(self):
        indexes = np.random.choice(range(len(self.memory)), size=self.batch_size)
        experiences = [self.memory[i] for i in indexes]
        is_weights = np.ones(self.batch_size)

        return (indexes, experiences, is_weights)

    def __len__(self):
        return super().__len__()
