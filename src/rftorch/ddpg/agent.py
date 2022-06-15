import torch as T
import numpy as np
from rftorch.ddpg.noise import OUActionNoise
import copy
from rftorch.ddpg.memory import DDPGMemory
from rftorch.ddpg.example_networks import ExampleActor, ExampleCritic

# DDPG_Agent is the class for the agent implementing a Deep Deterministic Policy Gradient agent
class DDPGAgent():
    def __init__(self, actor=ExampleActor(), critic=ExampleCritic(), tau=0.001, gamma=0.99, 
                mem_size=1000000, batch_size=64):        
        self.actor = actor                              # actor neural network
        self.critic = critic                            # critic neural network        
        self.target_actor = copy.deepcopy(actor)        # target actor neural network
        self.target_critic = copy.deepcopy(critic)      # target critic neural network
                          
        self.tau = tau                                  # target networks update constant
        self.gamma = gamma                              # reward discount rate
        self.batch_size = batch_size                    # training batch size
        self.device = self.actor.device

        self.noise = OUActionNoise(mu=np.zeros(self.actor.n_actions))               # noise generator
        self.memory = DDPGMemory(mem_size=mem_size, input_dims=actor.input_dims,   # memory object
                                action_dims=actor.n_actions)      


    # save_memory is an interface for the save_memory method of the memory object
    def save_memory(self, state, action, reward, new_state, done):
        self.memory.save_memory(state, action, reward, new_state, done)


    # get_action returns the action for a given state with the highest q-value
    def get_eval_action(self, state):
        # actor network is set to eval
        self.actor.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor.forward(state).to(self.device)
        
        # action value is returned as numpy array
        return mu.cpu().detach().numpy()


    # get_action returns the action for a given state with the highest q-value
    def get_train_action(self, state):
        # actor network is set to eval
        self.actor.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor.forward(state).to(self.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.device)
        
        # action value with noise is returned as a torch tensor
        return mu_prime


    # learn executes a learning step for the neural network of the agent
    def learn(self):
        # if not enought memories (smaller than batch size) then don't learn
        if self.memory.mem_cntr < self.batch_size:
            return

        # memory is sampled
        states, actions, rewards, new_states, dones = self.memory.sample_memory(
                                                    batch_size=self.batch_size, device=self.device)
      
        # networks used are set to eval mode
        self.actor.eval() 
        self.critic.eval() 
        self.target_actor.eval() 
        self.target_critic.eval()  

        # action and q-values are computed
        mu = self.actor.forward(states)
        q = self.critic.forward(states, actions)
        mu_prime = self.target_actor.forward(new_states)
        q_prime = self.target_critic.forward(new_states, mu_prime)

        # q-target values are computed
        q_target = []
        for i in range(self.batch_size):
            q_target.append(rewards[i] + self.gamma * q_prime[i] * dones[i])
        q_target = T.tensor(q_target, dtype=T.float).to(self.device)
        q_target = q_target.view(self.batch_size, 1)
        
        # Critic network is updated
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = T.nn.functional.mse_loss(q_target, q)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        # Actor network is updated
        self.actor.train() 
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # Target networks are updated
        self.update_target_networks()


    # update_target_network updates the target networks parameters
    def update_target_networks(self):
        # we get the network parameters
        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        # target actor parameters are updated
        for name in actor_params:
            target_actor_params[name] = self.tau*actor_params[name].clone() + \
                (1-self.tau)*target_actor_params[name].clone()

        # target critic parameters are updated
        for name in critic_params:
            target_critic_params[name] = self.tau*critic_params[name].clone() + \
                (1-self.tau)*target_critic_params[name].clone()

        # updated parameters are loaded to the target networks
        self.target_actor.load_state_dict(target_actor_params)
        self.target_critic.load_state_dict(target_critic_params)
