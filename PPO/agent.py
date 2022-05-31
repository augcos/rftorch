import torch as T
import numpy as np
from memory import Memory

class Discrete_PPO_Agent():
    def __init__(self, actor_network, critic_network, gamma=0.99, gae=0.95, clipping=0.2, 
                    n_epochs=10, batch_size=64):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma
        self.gae = gae
        self.clipping = clipping
        self.n_epochs = n_epochs

        self.memory = Memory(batch_size)
        
    
    def get_action(self, state):
        self.actor_network.eval()
        self.critic_network.eval()

        state = T.tensor(state, dtype=T.float).to(self.actor_network.device)
        action_probs = self.actor_network.forward(state)
        value = self.critic_network.forward(state)

        action_sampled = action_probs.sample()
        prob = action_probs.log_prob(action_sampled).item()
        action = action_sampled.item()
        value = value.item()
        
        return action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            batch_indices, states, actions, probs, values, rewards, dones = \
                                                            self.memory.get_batches()

            advantage = np.zeros(len(rewards), dtype=np.float32)            
            for t in range(len(rewards)-1):
                discount = 1
                for k in range(t, len(rewards)-1):
                    advantage[t] += discount * (rewards[k] + self.gamma * values[k+1] * (1-int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae
            advantage = T.tensor(advantage, dtype=T.float).to(self.actor_network.device)

            values = T.tensor(values).to(self.actor_network.device)
            for batch in batch_indices:
                state_batch = T.tensor(states[batch], dtype=T.float).to(self.actor_network.device)
                old_prob_batch = T.tensor(probs[batch], dtype=T.float).to(self.actor_network.device)
                action_batch = T.tensor(actions[batch], dtype=T.float).to(self.actor_network.device)

                self.actor_network.eval()
                self.critic_network.eval()
                actor_probs = self.actor_network.forward(state_batch)
                critic_value = self.critic_network.forward(state_batch)
                critic_value = T.squeeze(critic_value)

                new_prob_batch = actor_probs.log_prob(action_batch)
                prob_ratio = new_prob_batch.exp() / old_prob_batch.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.clipping, 1+self.clipping)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor_network.train()
                self.critic_network.train()
                self.actor_network.optimizer.zero_grad()
                self.critic_network.optimizer.zero_grad()

                total_loss.backward()
                self.actor_network.optimizer.step()
                self.critic_network.optimizer.step()
        
        self.memory.clear_memory()

