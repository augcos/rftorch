import torch as T
import numpy as np

# DDPGMemory is the class for a memory buffer
class DDPGMemory():
    # the memory is initialized
    def __init__(self, mem_size, input_dims, action_dims):
        self.mem_size = mem_size
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, action_dims), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.mem_cntr = 0       # memory counter is initialized to 0
    
    # save_memory receives the information from the enviroment and the agent and saves it
    # in the numpy arrays
    def save_memory(self, state, action, reward, new_state, done):
        # we create and index that is reseted after reaching mem_size
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.done_memory[index] = 1-done

        # memory counter is increased by one
        self.mem_cntr += 1

    # sample_memory returns a batch of memories with size batch_size
    def sample_memory(self, batch_size, device):
        batch_index = np.random.choice(min(self.mem_cntr, self.mem_size),
                                        batch_size, replace=False)

        # batches are created as tensors in the provided device
        states = T.tensor(self.state_memory[batch_index], dtype=T.float).to(device)
        actions = T.tensor(self.action_memory[batch_index], dtype=T.float).to(device)
        rewards = T.tensor(self.reward_memory[batch_index], dtype=T.float).to(device)
        new_states = T.tensor(self.new_state_memory[batch_index], dtype=T.float).to(device)
        dones = T.tensor(self.done_memory[batch_index], dtype=T.bool).to(device)
        
        return states, actions, rewards, new_states, dones
