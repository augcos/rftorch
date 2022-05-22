import torch as T
import numpy as np
from memory import Memory
import copy

class DQN_agent():
    def __init__(self, DQN_network, loss=T.nn.MSELoss(), tau=100, epsilon=1.0, min_epsilon=0.01, 
                step_epsilon=5e-5, gamma=0.99, mem_size=100000, batch_size=64):
        self.DQN_network = DQN_network
        self.target_network = copy.deepcopy(DQN_network)
        self.loss = loss
        self.tau = tau

        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.step_epsilon = step_epsilon
        self.batch_size = batch_size

        self.action_space = [i for i in range(DQN_network.n_actions)]
        self.memory = Memory(mem_size=mem_size, input_shape=DQN_network.input_shape)

    def eval(self, state):
        state = T.tensor([state]).to(self.DQN_network.device)
        q_values = self.DQN_network.forward(state)
        action = T.argmax(q_values).item()

        return action

    def epsilon_greedy(self, state):
        if np.random.random() > self.epsilon:
            action = self.eval(state)
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.DQN_network.optimizer.zero_grad()
        self.update_target_network()

        aux_index = np.arange(self.batch_size, dtype=np.int32)
        states, actions, rewards, new_states, dones = self.memory.sample_memory(batch_size=self.batch_size)
        states = T.tensor(states).to(self.DQN_network.device)
        rewards = T.tensor(rewards).to(self.DQN_network.device)
        new_states = T.tensor(new_states).to(self.DQN_network.device)
        dones = T.tensor(dones).to(self.DQN_network.device)

        q_pre = self.DQN_network.forward(states)[aux_index, actions]
        q_post = self.DQN_network.forward(new_states)
        q_target = self.target_network.forward(new_states)

        q_target[dones] = 0.0

        q_updated = rewards + self.gamma * q_target[aux_index, T.argmax(q_post, dim=1)]

        loss = self.loss(q_updated, q_pre).to(self.DQN_network.device)
        loss.backward()
        self.DQN_network.optimizer.step()
        self.epsilon = self.epsilon - self.step_epsilon if self.epsilon > self.min_epsilon \
                                                            else self.min_epsilon    

    def update_target_network(self):
        if self.memory.mem_cntr % self.tau == 0:
            self.target_network.load_state_dict(self.DQN_network.state_dict())