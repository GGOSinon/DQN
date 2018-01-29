import numpy as np
import random 

class DataMemory():
    
    def __init__(self, capacity = 5000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, x):
        if len(self.memory)<self.capacity:
            self.memory.append(x)
        else:
            self.memory[self.position] = x
            self.position = (self.position+1) % self.capacity

    def sample(self, mb_size):
        if len(self.memory)<mb_size: return self.memory
        return random.sample(self.memory, mb_size)

    def __len__(self):
        return len(self.memory)
