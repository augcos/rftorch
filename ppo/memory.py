import numpy as np

# Memory is the class for a memory buffer
class PPO_Memory():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.value_memory = []
        self.reward_memory = []
        self.done_memory = []
    
    # save_memory receives the information from the enviroment and the agent and saves it
    # in the lists
    def save_memory(self, state, action, prob, value, reward, done):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.prob_memory.append(prob)
        self.value_memory.append(value)
        self.reward_memory.append(reward)
        self.done_memory.append(done)

    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.value_memory = []
        self.reward_memory = []
        self.done_memory = []

    # sample_memory returns a batch of memories with size batch_size
    def get_batches(self):
        batch_start = np.arange(0, len(self.state_memory), self.batch_size)
        indices = np.arange(len(self.state_memory), dtype=np.int32)
        np.random.shuffle(indices)
        batch_indices = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batch_indices, \
                np.array(self.state_memory), \
                np.array(self.action_memory), \
                np.array(self.prob_memory), \
                np.array(self.value_memory), \
                np.array(self.reward_memory), \
                np.array(self.done_memory)
