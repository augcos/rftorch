import torch as T
import numpy as np
from noise import OUActionNoise
from memory import Memory
import copy

# DQN_agent is the class for the agent implementing a Deep-Q-Network agent
class DDPG_Agent():
    def __init__(self, actor_network, critic_network, tau=0.001, gamma=0.99, mem_size=1000000, batch_size=64):        
        self.actor_network = actor_network                      # actor neural network
        self.critic_network = critic_network                    # critic neural network        
        self.target_actor = copy.deepcopy(actor_network)        # target network is a copy of the main network
        self.target_critic = copy.deepcopy(critic_network)      # target network is a copy of the main network
                          
        self.tau = tau                              # target networks update constant
        self.gamma = gamma                          # reward discount rate
        self.batch_size = batch_size                # training batch size
        self.device = self.actor_network.device

        self.noise = OUActionNoise(mu=np.zeros(self.actor_network.n_actions))               # noise generator
        self.memory = Memory(mem_size=mem_size, input_dim=actor_network.input_dims,         # memory object of the agent
                                action_dim=actor_network.n_actions)      

    # get_action returns the action for a given state with the highest q-value
    def get_eval_action(self, state):
        self.actor_network.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor_network.forward(state).to(self.device)
        
        return mu.cpu().detach().numpy()

    # get_action returns the action for a given state with the highest q-value
    def get_train_action(self, state):
        self.actor_network.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor_network.forward(state).to(self.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.device)
        
        return mu_prime


    # learn executes a learning step for the neural network of the agent
    def learn(self):
        # if not enought memories (smaller than batch size) then return
        if self.memory.mem_cntr < self.batch_size:
            return

        # memory is sampled and the outputs turned into pytorch tensors
        states, actions, rewards, new_states, dones = self.memory.sample_memory(
                                                        batch_size=self.batch_size, device=self.device)
      
        # networks used for loss calculations are turned to eval mode
        self.actor_network.eval() 
        self.critic_network.eval() 
        self.target_actor.eval() 
        self.target_critic.eval()  

        # q
        mu = self.actor_network.forward(states)
        q = self.critic_network.forward(states, actions)
        mu_prime = self.target_actor.forward(new_states)
        q_prime = self.target_critic.forward(new_states, mu_prime)

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

        self.update_target_networks()


    # update_target_network updates the target network parameters with the parameters of the 
    # main network (every tau learning steps)
    def update_target_networks(self):
        actor_params = dict(self.actor_network.named_parameters())
        critic_params = dict(self.critic_network.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        for name in actor_params:
            target_actor_params[name] = self.tau*actor_params[name].clone() + \
                (1-self.tau)*target_actor_params[name].clone()

        for name in critic_params:
            target_critic_params[name] = self.tau*critic_params[name].clone() + \
                (1-self.tau)*target_critic_params[name].clone()


        self.target_actor.load_state_dict(target_actor_params)
        self.target_critic.load_state_dict(target_critic_params)
