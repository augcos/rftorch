import torch as T
from rftorch.a2c.example_networks import ExampleActor, ExampleCritic

# ContinuousA2CAgent is the class for implementing an A2C agent with a continuous action space
class ContinuousA2CAgent():
    def __init__(self, actor=ExampleActor(), critic=ExampleCritic(), gamma=0.99, action_limit=1):
        self.actor = actor          # actor neural network
        self.critic = critic        # critic neural network
        self.gamma = gamma                          # reward discount rate
        self.action_limit = action_limit            # value limit of the action space

    # get_action returns the sample action for a given state and saves the log of the probability into the 
    # action_log_prob variable
    def get_action(self, state):
        state = T.tensor(state).to(self.actor.device)
        mu, std = self.actor.forward(state)
        action_probs = T.distributions.Normal(mu, T.exp(std))
        sampled = action_probs.sample()
        self.action_log_prob = action_probs.log_prob(sampled).to(self.actor.device)
        action = T.tanh(sampled) * self.action_limit
        
        return action.cpu().detach().numpy()
    
    # learn executes a learning step for the actor and critic network
    def learn(self, state, reward, new_state, done):
        # the enviroment data is turned into pytorch tensors
        state = T.tensor(state).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        new_state = T.tensor(new_state).to(self.critic.device)

        # network gradients are zeroed out
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        # state values are obtained from the critic network
        critic_value = self.critic.forward(state)
        new_critic_value = self.critic.forward(new_state)

        # advantage value and losses are computed
        advantage = reward + self.gamma * new_critic_value * (1-int(done)) - critic_value
        actor_loss = -self.action_log_prob.mean() * advantage.detach()
        critic_loss = advantage**2
        total_loss = actor_loss + critic_loss

        # loss is backpropagated and a learning step is performed for both networks
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()






# DiscreteA2CAgent is the class for implementing an A2C agent with a discrete action space
class DiscreteA2CAgent():
    def __init__(self, actor=ExampleActor(), critic=ExampleCritic(), gamma=0.99):
        self.actor = actor          # actor neural network
        self.critic = critic        # critic neural network
        self.gamma = gamma                          # reward discount rate
    
    # get_action returns the sample action for a given state and saves the log of the probability into the 
    # action_log_prob variable
    def get_action(self, state):
        state = T.tensor(state).to(self.actor.device)
        action_values = self.actor.forward(state)
        action_probs = T.distributions.Categorical(action_values)
        self.action = action_probs.sample()
        self.action_log_prob = action_probs.log_prob(self.action)

        return self.action.item()

    # learn executes a learning step for the actor and critic network
    def learn(self, state, reward, new_state, done):
        # the enviroment data is turned into pytorch tensors
        state = T.tensor(state).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        new_state = T.tensor(new_state).to(self.critic.device)

        # network gradients are zeroed out
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # state values are obtained from the critic network
        critic_value = self.critic.forward(state)
        new_critic_value = self.critic.forward(new_state)

        # advantage value and losses are computed
        advantage = reward + self.gamma * new_critic_value * (1-int(done)) - critic_value
        actor_loss = -self.action_log_prob * advantage
        critic_loss = advantage**2
        total_loss = actor_loss + critic_loss

        # loss is backpropagated and a learning step is performed for both networks
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
