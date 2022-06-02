import torch as T
import numpy as np
from noise import OUActionNoise
from memory import DDPG_Memory
import copy

# DDPG_Agent is the class for the agent implementing a Deep Deterministic Policy Gradient agent
class DDPG_Agent():
    def __init__(self, actor_network, critic_network, tau=0.001, gamma=0.99, mem_size=1000000, batch_size=64):        
        self.actor_network = actor_network                      # actor neural network
        self.critic_network = critic_network                    # critic neural network        
        self.target_actor = copy.deepcopy(actor_network)        # target networks are copies of the main networks
        self.target_critic = copy.deepcopy(critic_network) 
                          
        self.tau = tau                                          # target networks update constant
        self.gamma = gamma                                      # reward discount rate
        self.batch_size = batch_size                            # training batch size
        self.device = self.actor_network.device                 # global device is copied from the actor network

        self.noise = OUActionNoise(mu=np.zeros(self.actor_network.n_actions))               # OUA noise generator
        self.memory = DDPG_Memory(mem_size=mem_size, input_dim=actor_network.input_dims,    # memory object
                                action_dim=actor_network.n_actions)      


    # save_memory provides an interface for the same name method of the memory object
    def save_memory(self, state, action, reward, new_state, done):
        self.memory.save_memory(state, action, reward, new_state, done)


    # get_eval_action returns the continuous action for a given state as a numpy array
    def get_eval_action(self, state):
        self.actor_network.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor_network.forward(state).to(self.device)
        
        return mu.cpu().detach().numpy()


    # get_train_action returns the continuous action with added noise for a given state as a pytorch tensor
    def get_train_action(self, state):
        self.actor_network.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor_network.forward(state).to(self.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.device)   # noise generator is called
        
        return mu_prime


    # learn executes a learning step for the neural networks of the agent
    def learn(self):
        # if not enought memories (smaller than batch size) then return
        if self.memory.mem_cntr < self.batch_size:
            return

        # memory is sampled
        states, actions, rewards, new_states, dones = self.memory.sample_memory(
                                                        batch_size=self.batch_size, device=self.device)
      
        # networks are turned to eval mode
        self.actor_network.eval() 
        self.critic_network.eval() 
        self.target_actor.eval() 
        self.target_critic.eval()  

        # action and q-values are computed
        mu = self.actor_network.forward(states)
        q = self.critic_network.forward(states, actions)
        mu_prime = self.target_actor.forward(new_states)
        q_prime = self.target_critic.forward(new_states, mu_prime)

        # target q-value
        q_target = []
        for i in range(self.batch_size):
            q_target.append(rewards[i] + self.gamma * q_prime[i] * dones[i])
        q_target = T.tensor(q_target, dtype=T.float).to(self.device)
        q_target = q_target.view(self.batch_size, 1)
        
        # Critic network is updated
        self.critic_network.train()
        self.critic_network.optimizer.zero_grad()
        critic_loss = T.nn.functional.mse_loss(q_target, q)
        critic_loss.backward()
        self.critic_network.optimizer.step()
        self.critic_network.eval()

        # Actor network is updated
        self.actor_network.train() 
        self.actor_network.optimizer.zero_grad()            # network gradients are zeroed out
        actor_loss = -self.critic_network.forward(states, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor_network.optimizer.step()

        # target networks weights are updated
        self.update_target_networks()


    # update_target_network updates the target network parameters with the parameters of the main 
    # network
    def update_target_networks(self):
        # current network parameters are loaded
        actor_params = dict(self.actor_network.named_parameters())
        critic_params = dict(self.critic_network.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        # target network parameters are updated using the constant tau
        for name in actor_params:
            target_actor_params[name] = self.tau*actor_params[name].clone() + \
                (1-self.tau)*target_actor_params[name].clone()

        for name in critic_params:
            target_critic_params[name] = self.tau*critic_params[name].clone() + \
                (1-self.tau)*target_critic_params[name].clone()

        # updated parameters are loaded to the target network
        self.target_actor.load_state_dict(target_actor_params)
        self.target_critic.load_state_dict(target_critic_params)
