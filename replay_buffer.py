import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Stores experiences as tuples: (state, action, reward, next_state, done)
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the memory."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, device="cpu"):
        """
        Randomly samples a batch of experiences.
        Returns them as PyTorch tensors ready for the DQN.
        """
        # Get random sample
        experiences = random.sample(self.buffer, k=batch_size)
        
        # Unpack the list of tuples into separate lists
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert lists to numpy arrays first for efficiency
        states_np = np.array(states)
        next_states_np = np.array(next_states)
        
        # Convert to PyTorch Tensors and send to device
        states_t = torch.FloatTensor(states_np).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states_np).to(device)
        dones_t = torch.FloatTensor(dones).to(device)
        
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)
