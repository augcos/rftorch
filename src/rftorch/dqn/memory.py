import numpy as np

# Memory is the class for a memory buffer
class DQNMemory():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)

    # save_memory receives the information from the enviroment and the agent and saves it
    # in the numpy arrays
    def save_memory(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.done_memory[index] = done
        self.mem_cntr += 1

    # sample_memory returns a batch of memories with size batch_size
    def sample_memory(self, batch_size):
        batch_index = np.random.choice(min(self.mem_cntr, self.mem_size),
                                        batch_size, replace=False)

        states = self.state_memory[batch_index]
        actions = self.action_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        new_states = self.new_state_memory[batch_index]
        dones = self.done_memory[batch_index]
        
        return states, actions, rewards, new_states, dones