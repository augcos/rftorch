import numpy as np
import copy
import pickle
import torch as T
import torch.nn.functional as F
from rftorch.memory import MemoryBuffer


# DQNAgent is the parent class for the Deep-Q learning agents
class DQNAgent():
    ################################################ Constructor method ################################################
    def __init__(self, 
                 critic: T.nn.Module,   
                 input_shape: tuple, 
                 n_actions: int, 
                 tau: int = 1, 
                 eps: float = 1.0, 
                 min_eps: float = 0.01, 
                 d_eps: float = 1e-4, 
                 gamma: float = 0.99, 
                 mem_size: int = 1000000, 
                 batch_size: int = 256) -> None:
        
        # Enviroment parameters
        self.input_shape = input_shape
        self.action_space = [i for i in range(n_actions)]    

        # Critic network parameters
        self.critic = critic
        self.device = critic.device   

        # Target network parameters
        self.tau = tau
        self.target_steps = 0
        if self.tau>1:
            self.double = True
            self.target = copy.deepcopy(self.critic)
        else:
            self.double = False

        # Learning parameters
        self.gamma = gamma
        self.eps = eps
        self.min_eps = min_eps
        self.d_eps = d_eps
        self.batch_size = batch_size

        # Memory parameters
        self.mem_size = mem_size
        self.buffer = MemoryBuffer(mem_size=self.mem_size, input_shape=self.input_shape, discrete=True)




    ################################################## Public methods ##################################################
    # predict returns the action taken by the agent, given an input state. The train flag indicates if the agent follows
    # either a epsilon-greedy or a greedy policy
    def predict(self, state: np.ndarray, train: bool = False) -> np.int8:
        if train:
            if np.random.random() > self.eps:
                self._set_train()
                q_values = self.critic.forward(T.tensor(state, dtype=T.float).to(self.device))
            else:
                return np.random.choice(self.action_space).astype(np.int8)
        else:
            self._set_eval()
            q_values = self.critic.forward(T.tensor(state, dtype=T.float).to(self.device))

        return T.argmax(q_values).detach().cpu().numpy().astype(np.int8)


    # step takes a new memory as input, saves it to the replay buffer and performs a learning step
    def step(self, 
             state: np.ndarray, 
             action: np.int8, 
             reward: float, 
             new_state: np.ndarray, 
             done: bool) -> None:
        # the new memory is saved to the replay buffer
        self.buffer.save_memory(state, action, reward, new_state, done)

        # we check if we have at least batch_size memories to learn from. If not, we exit the function without learning    
        if not (self.buffer.full or self.buffer.counter > self.batch_size):
            return

        # if appropiate,  the target network is updated
        self._update_target()

        # a batch of memories is sampled from the replay buffer
        states, actions, rewards, new_states, dones = self._sample_memory()

        # the networks are set to train mode, and the loss is computed         
        self._set_train()
        loss = self._get_loss(states, actions, rewards, new_states, dones)  

        # we zero out the network gradients, and loss is backpropagated
        self.critic.optimizer.zero_grad()      
        loss.backward()                         
        self.critic.optimizer.step()       
        
        # epsilon is updated 
        self._update_epsilon()

    
    def save(self, path: str) -> None:
        with open(path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)



    ################################################# Private methods ##################################################
    # _update_target updates the target network, if appropiate
    def _update_target(self) -> None:
        if self.double:
            self.target_steps += 1
            if self.target_steps==self.tau:
                self.target_steps = 0
                self.target.load_state_dict(self.critic.state_dict())


    # _get_loss returns the MSE loss from a given batch of memories
    def _get_loss(self, 
                  states: np.ndarray, 
                  actions: np.ndarray, 
                  rewards: np.ndarray, 
                  new_states: np.ndarray, 
                  dones: np.ndarray) -> T.tensor:
        # q_values are computed
        q_pre = self.critic.forward(states)[range(len(actions)), actions]
        q_post = self.critic.forward(new_states)
        if self.double:
            q_target = self.target.forward(new_states)
        else:
            q_target = q_post
        q_target[dones] = 0.0

        # updated q-value and loss are computed
        q_updated = rewards + self.gamma * q_target[range(len(q_post)), T.argmax(q_post, dim=1)]
        loss = F.mse_loss(q_updated, q_pre)
        
        return loss


    # _sample_memory returns a batch of memories from the replay buffer. States, rewards, new_states and dones are
    # returned as tensors
    def _sample_memory(self) -> tuple[T.tensor, T.tensor, T.tensor, T.tensor, T.tensor]:
        states, actions, rewards, new_states, dones = self.buffer.sample_memory(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.device)

        return states, actions, rewards, new_states, dones


    # update_epsilon updates the epsilon parameter
    def _update_epsilon(self) -> None:
        self.eps = self.eps - self.d_eps if self.eps > self.min_eps else self.min_eps  


    # _set_eval sets the critic (and target) network to eval mode
    def _set_eval(self) -> None:
        self.critic.eval()
        if self.double:
            self.target.eval()


    # _set_train sets the critic (and target) network to train mode
    def _set_train(self):
        self.critic.train()
        if self.double:
            self.target.train()


    

   








# DuelingDQNAgent implements a DQN agent with dueling improvement
class DuelingDQNAgent(DQNAgent):
    ################################################ Constructor method ################################################
    def __init__(self, critic, input_shape, n_actions, tau=1, eps=1.0, min_eps=0.01, d_eps=1e-4, gamma=0.99, 
                mem_size=100000, batch_size=256):
        # we call the constructor method of the parent class
        super().__init__(critic, input_shape, n_actions, tau, eps, min_eps, d_eps, gamma, mem_size, batch_size)


    ################################################## Public methods ##################################################
    # predict returns the action taken by the agent, given an input state. The train flag indicates if the agent follows
    # either a epsilon-greedy or a greedy policy
    def predict(self, state: np.ndarray, train: bool = False) -> np.int8:
        if train:
            if np.random.random() > self.eps:
                self._set_train()
                _, advantage = self.critic.forward(T.tensor(np.expand_dims(state, 0), dtype=T.float).to(self.device))
            else:
                return np.random.choice(self.action_space).astype(np.int8)
        else:
            self._set_eval()
            _, advantage = self.critic.forward(T.tensor(np.expand_dims(state, 0), dtype=T.float).to(self.device))
        
        return T.argmax(advantage).detach().cpu().numpy().astype(np.int8)
        
        

    ################################################# Private methods ##################################################
    # _get loss returns the MSE loss from a batch of memories
    def _get_loss(self, 
                  states: np.ndarray, 
                  actions: np.ndarray, 
                  rewards: np.ndarray, 
                  new_states: np.ndarray, 
                  dones: np.ndarray) -> T.tensor:
        # state values and advantages are computed
        value_pre, advantage_pre = self.critic.forward(states)
        value_pos, advantage_post = self.critic.forward(new_states)

        # q-values are computed
        q_pre = T.add(value_pre, 
                        (advantage_pre - advantage_pre.mean(dim=1, keepdim=True)))[range(len(actions)), actions]
        q_post = T.add(value_pos, 
                        (advantage_post - advantage_post.mean(dim=1, keepdim=True)))
        if self.double:
            value_target, advantage_target = self.target.forward(new_states)
            q_target = T.add(value_target, 
                            (advantage_target - advantage_target.mean(dim=1, keepdim=True)))
        else:
            q_target = q_post
        q_target[dones] = 0.0

        # updated q-value and loss are computed 
        q_updated = rewards + self.gamma * q_target[range(len(q_post)), T.argmax(q_post, dim=1)]
        loss = F.mse_loss(q_updated, q_pre)

        return loss