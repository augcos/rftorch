import torch as T

# Continuos_A2C_agent is the class for implementing an A2C agent with a continuous
# action space
class Continuous_A2C_agent():
    def __init__(self, actor_network, critic_network, gamma=0.99, action_limit=1):
        self.actor_network = actor_network          # actor neural network
        self.critic_network = critic_network        # critic neural network
        self.gamma = gamma                          # reward discount rate
        self.action_limit = action_limit            # value limit of the action space

    # get_action returns the sample action for a given state and saves the log of the
    # probability into the action_log_prob variable
    def get_action(self, state):
        state = T.tensor(state).to(self.actor_network.device)
        mu, std = self.actor_network.forward(state)
        action_probs = T.distributions.Normal(mu, T.exp(std))
        sampled = action_probs.sample()
        self.action_log_prob = action_probs.log_prob(sampled).to(self.actor_network.device)
        action = T.tanh(sampled) * self.action_limit
        
        return action.cpu().detach().numpy()
    
    # learn executes a learning step for the actor and critic network
    def learn(self, state, reward, new_state, done):
        # the enviroment data is turned into pytorch tensors
        state = T.tensor(state).to(self.critic_network.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_network.device)
        new_state = T.tensor(new_state).to(self.critic_network.device)

        # network gradients are zeroed out
        self.actor_network.optimizer.zero_grad()
        self.critic_network.optimizer.zero_grad()
        
        # state values are obtained from the critic network
        critic_value = self.critic_network.forward(state)
        new_critic_value = self.critic_network.forward(new_state)

        # advantage value and losses are computed
        advantage = reward + self.gamma * new_critic_value * (1-int(done)) - critic_value
        actor_loss = -self.action_log_prob.mean() * advantage.detach()
        critic_loss = advantage**2
        total_loss = actor_loss + critic_loss

        # loss is backpropagated and a learning step is performed for both networks
        total_loss.backward()
        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()






# Discrete_A2C_Agent is the class for implementing an A2C agent with a discrete
# action space
class Discrete_A2C_Agent():
    def __init__(self, actor_network, critic_network, gamma=0.99):
        self.actor_network = actor_network          # actor neural network
        self.critic_network = critic_network        # critic neural network
        self.gamma = gamma                          # reward discount rate
    
    # get_action returns the sample action for a given state and saves the log of the
    # probability into the action_log_prob variable
    def get_action(self, state):
        state = T.tensor(state).to(self.actor_network.device)
        action_values = self.actor_network.forward(state)
        action_probs = T.distributions.Categorical(action_values)
        self.action = action_probs.sample()
        self.action_log_prob = action_probs.log_prob(self.action)

        return self.action.item()

    def learn(self, state, reward, new_state, done):
        # the enviroment data is turned into pytorch tensors
        state = T.tensor(state).to(self.critic_network.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_network.device)
        new_state = T.tensor(new_state).to(self.critic_network.device)

        # network gradients are zeroed out
        self.actor_network.optimizer.zero_grad()
        self.critic_network.optimizer.zero_grad()

        # state values are obtained from the critic network
        critic_value = self.critic_network.forward(state)
        new_critic_value = self.critic_network.forward(new_state)

        # advantage value and losses are computed
        advantage = reward + self.gamma * new_critic_value * (1-int(done)) - critic_value
        actor_loss = -self.action_log_prob * advantage
        critic_loss = advantage**2
        total_loss = actor_loss + critic_loss

        # loss is backpropagated and a learning step is performed for both networks
        total_loss.backward()
        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()
