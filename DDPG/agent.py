import torch as T
import numpy as np
from noise import OUActionNoise
from memory import Memory
import copy

# DQN_agent is the class for the agent implementing a Deep-Q-Network agent
class DQN_agent():
    def __init__(self, actor_network, critic_network, tau=250, gamma=0.99, mem_size=100000, 
                batch_size=64, action_limit=1):        
        self.actor_network = actor_network                      # actor neural network
        self.critic_network = critic_network                    # critic neural network        
        self.target_actor = copy.deepcopy(actor_network)        # target network is a copy of the main network
        self.target_critic = copy.deepcopy(critic_network)      # target network is a copy of the main network
                          
        self.tau = tau                              # target networks update period
        self.gamma = gamma                          # reward discount rate
        self.batch_size = batch_size                # training batch size
        self.action_limit = action_limit            # value limit of the action space

        self.noise = OUActionNoise(mu=np.zeros(self.actor_network.n_actions))               # noise generator
        self.memory = Memory(mem_size=mem_size, input_shape=actor_network.input_shape)      # memory object of the agent


    # get_action returns the action for a given state with the highest q-value
    def get_action(self, state, train=False):
        self.actor_network.eval()
        state = T.tensor(state).to(self.actor_network.device)
        mu = self.actor_network.forward(state).to(self.actor_network.device)
        if train:
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor_network.device)
        else:
            mu_prime = mu
        
        return mu_prime.cpu().detach().numpy()


    # learn executes a learning step for the neural network of the agent
    def learn(self):
        # if not enought memories (smaller than batch size) then return
        if self.memory.mem_cntr < self.batch_size:
            return

        self.actor_network.optimizer.zero_grad()        # network gradients are zeroed out
        self.critic_network.optimizer.zero_grad()
        self.update_target_network()                    # target networks update function is called

        # memory is sampled and the outputs turned into pytorch tensors
        states, actions, rewards, new_states, dones = self.memory.sample_memory(batch_size=self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor_network.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor_network.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor_network.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor_network.device)
        dones = T.tensor(dones, dtype=T.int).to(self.actor_network.device)
        
        # networks used for loss calculations are turned to eval mode
        self.target_actor.eval() 
        self.target_critic.eval()  
        self.critic_network.eval() 

        # values are computed
        target_actions = self.target_actor.forward(new_states)
        target_new_critic_values = self.target_critic.forward(new_states, target_actions)
        critic_values = self.critic_network.forward(states, actions)

        # target values are computed
        target = []
        for i in range(self.batch_size):
            target.append(rewards[i] + self.gamma * target_new_critic_values[i] * dones[i])
        target = T.tensor(target).to(self.actor_network.device)


        self.critic_network.train()
        self.critic_network.optimizer.zero_grad()
        critic_loss = T.nn.Functional.mse_loss(target, critic_values)
        critic_loss.backward()
        self.critic_network.optimizer.step()
        self.critic_network.eval()

        mu = self.actor_network.forward(states)
        self.actor_network.train()
        actor_loss = -self.critic_network.forward(states, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor_network.optimizer.step()


    # update_target_network updates the target network parameters with the parameters of the 
    # main network (every tau learning steps)
    def update_target_network(self):
        if self.memory.mem_cntr % self.tau == 0:
            self.target_actor.load_state_dict(self.actor_network.state_dict())
            self.target_critic.load_state_dict(self.critic_network.state_dict())


