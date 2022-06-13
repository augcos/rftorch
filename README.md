# __rftorch__
## Introduction
__rftorch__ is a minimalistic __deep reinforcement learning (DRL)__ library written for __PyTorch__. Most well-known DRL libraries are well integrated with the gym enviroment. The goal of __rftorch__ is to implement the main DRL agents leaving the enviroment and neural network implementations completely to the user to the user.

Each DRL algorithm includes an example_main.py file implementing 

## Implemented agents
* __[Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783)__
* __[Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)__
* __[Deep-Q-Network (DQN)](https://arxiv.org/abs/1312.5602)__, including the main DQN improvements ([double DQN](https://arxiv.org/abs/1509.06461) and [dueling DQN](https://arxiv.org/abs/1511.06581). [Prioritized experience replay](https://arxiv.org/abs/1511.05952) coming soon...).
* __[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)__, only for a deterministic action space. PPO for a continuous action space coming soon...