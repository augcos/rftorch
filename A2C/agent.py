import torch as T

class Continuous_A2C_Agent():
    def __init__(self, actor_network, critic_network, gamma=0.99):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma

    def eval(self, state):
        state = T.tensor([state]).to(self.actor_network.device)
        mu, std = self.actor_network.forward(state)
        action_probs = T.distributions.Normal(mu, std)
        sampled = action_probs.sample(sample_shape=T.Size([1]))
        action = T.tanh(sampled)

        self.action_log_prob = action_probs.log_probs(sampled).to(self.actor_network.device)
        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_network.zero_grad()
        self.critic_optimizer.zero_grad()

        critic_value = self.critic.forward(state)
        new_critic_value = self.critic.forward(new_state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_network.device)

        delta = reward + self.gamma * new_critic_value * (1-int(done)) - critic_value
        actor_loss = -self.action_log_prob * delta
        critic_loss = delta**2
        total_loss = actor_loss + critic_loss

        total_loss.backward()
        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()




class Discrete_A2C_Agent():
    def __init__(self, actor_network, critic_network, gamma=0.99, n_outputs=1):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma
        self.n_outputs = n_outputs

    def eval(self, state):
        state = T.tensor([state]).to(self.actor_network.device)
        action_values = self.actor_network.forward(state)
        action_probs = T.distributions.Categorical(action_values)
        self.action = action_probs.sample()
        self.action_log_prob = action_probs.log_prob(self.action)

        return self.action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_network.zero_grad()
        self.critic_optimizer.zero_grad()

        critic_value = self.critic.forward(state)
        new_critic_value = self.critic.forward(new_state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_network.device)

        delta = reward + self.gamma * new_critic_value * (1-int(done)) - critic_value
        actor_loss = -self.action_log_prob * delta
        critic_loss = delta**2
        total_loss = actor_loss + critic_loss

        total_loss.backward()
        self.actor_network.optimizer.step()
        self.critic_network.optimizer.step()