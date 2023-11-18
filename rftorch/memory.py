import numpy as np

# MemoryBuffer is the class for the replay buffers
class MemoryBuffer():
    def __init__(self, 
                 mem_size: int, 
                 input_shape: tuple, 
                 discrete: bool) -> None:
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.discrete = discrete
        self.counter = 0
        self.full = False

        self.states = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.new_states = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        self.dones = np.zeros(self.mem_size, dtype=bool)

        if self.discrete:
            self.actions = np.zeros(self.mem_size, dtype=np.int8)
        else:
            self.actions = np.zeros((self.mem_size, self.action_dim), dtype=np.float32)

    # save_memory receives saves the enviroment information in the buffers
    def save_memory(self, 
                    state: np.ndarray, 
                    action: np.ndarray, 
                    reward: np.ndarray, 
                    new_state: np.ndarray, 
                    done: bool) -> None:
        self._update_counter()
        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.new_states[self.counter] = new_state
        self.dones[self.counter] = done


    # sample_memory returns a batch of memories
    def sample_memory(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        batch_indices = np.random.choice(self.mem_size if self.full else self.counter, min(batch_size, self.mem_size), 
                                         replace=False)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        new_states = self.new_states[batch_indices]
        dones = self.dones[batch_indices]
        
        return states, actions, rewards, new_states, dones

    # clean_memory resets the buffers
    def clean_memory(self) -> None:
        self.counter = 0
        self.full = False
        self.states = np.zeros((self.mem_size, *self.input_dim), dtype=np.float32)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.new_states = np.zeros((self.mem_size, *self.input_dim), dtype=np.float32)
        self.done_buffer = np.zeros(self.mem_size, dtype=bool)

        if self.discrete:
            self.actions = np.zeros(self.mem_size, dtype=np.int8)
        else:
            self.actions = np.zeros((self.mem_size, self.action_dim), dtype=np.float32)
    
    def _update_counter(self) -> None:
        self.counter += 1
        if self.counter==self.mem_size:
            self.counter = 0
            self.full = True