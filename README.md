# __rftorch__
## Introduction
__rftorch__ is a minimalistic __deep reinforcement learning (DRL)__ library written for __PyTorch__. While most well-known DRL libraries are deeply integrated with the gym enviroment, the goal of __rftorch__ is to provide a generic, minimalistic framework for working with DRL. __rftorch__ implements the main DRL algorithms, while giving complete freedom to the user on how to design the enviroment and the deep neural netowrk models.

Each DRL algorithm includes an example_main.py file implementing an test case of how to train and use the agent. Default models are imported from the example_networks.py files (hyperparameters are already somewhat optimized, training is ready to go without any further modifications).

This library is still in very early development, expect things to break. Future plans include improved testing and expanding the number of DRL agorithms implemented.

## Implemented agents
* __[Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783)__
* __[Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)__
* __[Deep-Q-Network (DQN)](https://arxiv.org/abs/1312.5602)__, including the main DQN improvements ([double DQN](https://arxiv.org/abs/1509.06461) and [dueling DQN](https://arxiv.org/abs/1511.06581). [Prioritized experience replay](https://arxiv.org/abs/1511.05952) coming soon...).
* __[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)__. For now, it is only for a deterministic action space. PPO for a continuous action space coming soon...

## Example
This snippet is a modified version of the code included in this [example file](https://github.com/augcos/rftorch/blob/master/src/rftorch/dqn/example_main.py). 
```python
# we import the gym enviroment and our Deep Q-Network agent
import gym
import numpy as np
from rftorch.dqn.agent import DQNAgent

# The lunar lander enviroment is loaded
env = gym.make('LunarLander-v2')

# we create the Deep Q-Network agent with the default settings
agent = DQNAgent()

# we define some auxiliary variables
score_history = []
eps_history = []
n_episodes = 1000

# agent is trained in the enviroment
for i in range(n_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        # we get the action using a epsilon greedy policy (see DQN implementation)
        action = agent.get_train_action(state)
        # we get the info from the enviroment
        new_state, reward, done, info = env.step(action)
        # the agent saves this interaction in the memory buffer
        agent.save_memory(state, action, reward, new_state, done)
        # the agent learns from the memory buffer
        agent.learn()
        # state is modified, and reward is added to the total score
        state = new_state
        score += reward
    score_history.append(score)
    # we print the training output
    print("Episode %d - Score: %.2f - Average score: %.2f - Epsilon: %.2f" 
        % (i, score, np.mean(score_history[-100:]), agent.epsilon))
```

## References
Code in this library is heavily inspired by the work of [Phil Tabor](https://github.com/philtabor). Please check his youtube channel [here](https://www.youtube.com/c/MachineLearningwithPhil).