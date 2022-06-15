import torch as T
import numpy as np
import copy
from rftorch.dqn.memory import DQNMemory
from rftorch.dqn.example_networks import ExampleDeepQNetwork, ExampleDuelingDeepQNetwork

# DQN_agent is the class for the agent implementing a Double Deep-Q-Network agent
class DQNAgent():
    def __init__(self, DQN_network=ExampleDeepQNetwork(), loss=T.nn.MSELoss(), tau=100, epsilon=1.0, 
                min_epsilon=0.01, step_epsilon=0.001, gamma=0.99, mem_size=100000, batch_size=64):
        self.DQN_network = DQN_network                      # neural network (default: ExampleDeepQNetwork)
        self.target_network = copy.deepcopy(DQN_network)    # target network is a copy of the main network
        self.loss = loss                                    # loss function (default: MSE)
        self.tau = tau                                      # target network update period

        self.gamma = gamma                                  # reward discount rate
        self.epsilon = epsilon                              # exploration constant
        self.min_epsilon = min_epsilon                      # min value for epsilon
        self.step_epsilon = step_epsilon                    # epsilon deacrease step value
        self.batch_size = batch_size                        # training batch size

        self.action_space = [i for i in range(DQN_network.n_actions)]                   # action space
        self.memory = DQNMemory(mem_size=mem_size, input_dims=DQN_network.input_dims)    # memory object

    
    # save_memory is an interface for the save_memory method of the memory object
    def save_memory(self, state, action, reward, new_state, done):
        self.memory.save_memory(state, action, reward, new_state, done)


    # get_eval_action returns the action for a given state with the highest q-value
    def get_eval_action(self, state):
        self.DQN_network.eval()
        state = T.tensor(state, dtype=T.float).to(self.DQN_network.device)
        q_values = self.DQN_network.forward(state)
        action = T.argmax(q_values).item()

        return action


    # get_train_action returns the action for a given state using an epsilon greedy policy
    def get_train_action(self, state):
        if np.random.random() > self.epsilon:
            action = self.get_eval_action(state)
        else:
            action = np.random.choice(self.action_space)
        
        return action


    # learn executes a learning step for the neural network of the agent
    def learn(self):
        # if not enought memories (smaller than batch size) then return
        if self.memory.mem_cntr < self.batch_size:
            return

        self.update_target_network()                # target network update funcion is called
        self.DQN_network.eval()                     # network is set to eval mode

        # memory is sampled and the outputs turned into pytorch tensors
        states, actions, rewards, new_states, dones = self.memory.sample_memory(batch_size=self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.DQN_network.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.DQN_network.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.DQN_network.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.DQN_network.device)

        # q-values are computed
        q_pre = self.DQN_network.forward(states)[range(len(actions)), actions]
        q_post = self.DQN_network.forward(new_states)
        q_target = self.target_network.forward(new_states)
        q_target[dones] = 0.0

        # updated q-value is computed 
        q_updated = rewards + self.gamma * q_target[range(len(q_post)), T.argmax(q_post, dim=1)]

        # loss is computed and backpropagated
        self.DQN_network.train()                    # network is set to train mode
        self.DQN_network.optimizer.zero_grad()      # network gradients are zeroed out
        loss = self.loss(q_updated, q_pre).to(self.DQN_network.device)
        loss.backward()
        
        # network performs a learning step and epsilon is updated
        self.DQN_network.optimizer.step()
        self.epsilon = self.epsilon - self.step_epsilon if self.epsilon > self.min_epsilon \
                                                            else self.min_epsilon    


    # update_target_network updates the target network parameters with the parameters of the 
    # main network (every tau learning steps)
    def update_target_network(self):
        if self.memory.mem_cntr % self.tau == 0:
            self.target_network.load_state_dict(self.DQN_network.state_dict())






# Dueling_DQN_agent is the class for the agent implementing a DQN agent with dueling improvement
class DuelingDQNAgent():
    def __init__(self, DQN_network=ExampleDuelingDeepQNetwork(), loss=T.nn.MSELoss(), tau=250, epsilon=1.0, 
                min_epsilon=0.01, step_epsilon=5e-5, gamma=0.99, mem_size=100000, batch_size=64):
        self.DQN_network = DQN_network                      # neural network
        self.target_network = copy.deepcopy(DQN_network)    # target network is a copy of the main network
        self.loss = loss                                    # loss function (default: MSE)
        self.tau = tau                                      # target network update period

        self.gamma = gamma                                  # reward discount rate
        self.epsilon = epsilon                              # exploration constant
        self.min_epsilon = min_epsilon                      # min value for epsilon
        self.step_epsilon = step_epsilon                    # epsilon deacrease step value
        self.batch_size = batch_size                        # training batch size

        self.action_space = [i for i in range(DQN_network.n_actions)]                       # action space
        self.memory = DQNMemory(mem_size=mem_size, input_dims=DQN_network.input_dims)      # memory object 


    # save_memory is an interface for the save_memory method of the memory object
    def save_memory(self, state, action, reward, new_state, done):
        self.memory.save_memory(state, action, reward, new_state, done)


    # get_action returns the action for a given state with the highest q-value
    def get_eval_action(self, state):
        self.DQN_network.eval()
        state = T.tensor(state, dtype=T.float).to(self.DQN_network.device)
        _, advantage = self.DQN_network.forward(state)
        action = T.argmax(advantage).item()

        return action


    # epsilon_greedy returns the action for a given state using an epsilon greedy training policy
    def get_train_action(self, state):
        if np.random.random() > self.epsilon:
            action = self.get_eval_action(state)
        else:
            action = np.random.choice(self.action_space)
        
        return action


    # learn executes a learning step for the neural network of the agent
    def learn(self):
        # if not enought memories (smaller than batch size) then return
        if self.memory.mem_cntr < self.batch_size:
            return
            
        self.update_target_network()                # target network update funcion is called
        self.DQN_network.eval()                     # network is set to eval mode

        # memory is sampled and the outputs turned into pytorch tensors
        states, actions, rewards, new_states, dones = self.memory.sample_memory(batch_size=self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.DQN_network.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.DQN_network.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.DQN_network.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.DQN_network.device)

        # state values and advantages are computed
        value_pre, advantage_pre = self.DQN_network.forward(states)
        value_pos, advantage_post = self.DQN_network.forward(new_states)
        value_target, advantage_target = self.target_network.forward(new_states)

        # q-values are computed
        q_pre = T.add(value_pre, (advantage_pre - advantage_pre.mean(dim=1, keepdim=True)))[range(len(actions)), actions]
        q_post = T.add(value_pos, (advantage_post - advantage_post.mean(dim=1, keepdim=True)))
        q_target = T.add(value_target, (advantage_target - advantage_target.mean(dim=1, keepdim=True)))
        q_target[dones] = 0.0


        # updated q-value is computed 
        q_updated = rewards + self.gamma * q_target[range(len(q_post)), T.argmax(q_post, dim=1)]

        # loss is computed and backpropagated
        self.DQN_network.train()                    # network is set to train mode
        self.DQN_network.optimizer.zero_grad()      # network gradients are zeroed out
        loss = self.loss(q_updated, q_pre).to(self.DQN_network.device)
        loss.backward()
        
        # network performs a learning step and epsilon is updated
        self.DQN_network.optimizer.step()
        self.epsilon = self.epsilon - self.step_epsilon if self.epsilon > self.min_epsilon \
                                                            else self.min_epsilon    


    # update_target_network updates the target network parameters with the parameters of the 
    # main network (every tau learning steps)
    def update_target_network(self):
        if self.memory.mem_cntr % self.tau == 0:
            self.target_network.load_state_dict(self.DQN_network.state_dict())