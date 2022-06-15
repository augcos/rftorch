import torch as T
import numpy as np
from rftorch.ppo.memory import PPO_Memory
from rftorch.ppo.example_networks import ExampleActor, ExampleCritic

# DiscretePPOAgent is the class for the agent implementing a PPO agent with a discree action space
class DiscretePPOAgent():
    def __init__(self, actor=ExampleActor(), critic=ExampleCritic(), gamma=0.99, gae=0.95, 
                    clipping=0.2, n_epochs=4, batch_size=5):
        self.actor = actor                      # actor network
        self.critic = critic                    # critic network
        self.gamma = gamma                      # reward discount rate
        self.gae = gae                          # GAE lambda constant
        self.clipping = clipping                # policy clipping constant
        self.n_epochs = n_epochs                # number of epochcs
        
        self.memory = PPO_Memory(batch_size)    # memory object
        
        
    # save_memory is an interface for the save_memory method of the memory object
    def save_memory(self, state, action, prob, value, reward, done):
        self.memory.save_memory(state, action, prob, value, reward, done)


    # get_eval_action returns the action for a given state, its probability and the
    # value of the state
    def get_eval_action(self, state):
        # networks are set to eval mode
        self.actor.eval()
        self.critic.eval()

        # action probabilities and state value are computed
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action_probs = self.actor.forward(state)
        value = self.critic.forward(state)

        # action is sampled, and outputs are returned as python
        # numbers
        action = action_probs.sample()
        prob = action_probs.log_prob(action).item()
        action = action.item()
        value = value.item()
        
        return action, prob, value


    # learn executes a learning step for the neural network of the agent
    def learn(self):
        # for loop to iterate over each epoch
        for _ in range(self.n_epochs):
            # memory is sampled
            batch_indices, states, actions, probs, values, rewards, dones = \
                                                            self.memory.get_batches()

            # advantage values arec computed over each time step
            advantage = np.zeros(len(rewards), dtype=np.float32)            
            for t in range(len(rewards)-1):
                discount = 1
                for k in range(t, len(rewards)-1):
                    advantage[t] += discount * (rewards[k] + self.gamma * values[k+1] * (1-int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae
            advantage = T.tensor(advantage, dtype=T.float).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # we iterate over each batch
            for batch in batch_indices:
                # we turned the state, probs and action batched into torch tensors
                state_batch = T.tensor(states[batch], dtype=T.float).to(self.actor.device)
                old_prob_batch = T.tensor(probs[batch], dtype=T.float).to(self.actor.device)
                action_batch = T.tensor(actions[batch], dtype=T.float).to(self.actor.device)

                # we set the networks to eval mode
                self.actor.eval()
                self.critic.eval()
                # we compute the new action probabilities and state value
                actor_probs = self.actor.forward(state_batch)
                critic_value = self.critic.forward(state_batch)
                critic_value = T.squeeze(critic_value)

                # we compute the actor loss using the weighted clipped probability ratio
                new_prob_batch = actor_probs.log_prob(action_batch)
                prob_ratio = new_prob_batch.exp() / old_prob_batch.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.clipping, 1+self.clipping)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # we compute the critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                # we compute the total loss
                total_loss = actor_loss + 0.5*critic_loss
                
                # networks are set to train mode and gradients zero-ed out
                self.actor.train()
                self.critic.train()
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # loss is backpropagated and networks perform a learning step
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        # we clear the memory
        self.memory.clear_memory()