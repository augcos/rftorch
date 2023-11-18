# __rftorch__
## Introduction
__rftorch__ is a minimalistic deep reinforcement learning (DRL) library written for PyTorch. 

Most well-known DRL libraries enforce the [Gymnasium]() API for custom environments. The goal of __rftorch__ is to provide a generic framework that does not require but supports Gymnasium-style environments.

## Implemented algorithms
For now, the following DRL algorithms have been implemented in rftorch:
* [Deep-Q Learning](https://arxiv.org/pdf/1312.5602.pdf) (DQN), extended with the Double and Dueling DQN architectures.
* More coming soon...

## How to use
Run the following commands to download and locally install this library to your python environment:
```
git clone github.com/augcos/rftorch
pip install -e .
```

You can try one of the [examples](https://github.com/augcos/rftorch/examples) included in the library. For instance, you can train a DQN agent on the [LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander) environment running the command:
```
python3 ./rftorch/examples/dqn/lunar_lander_ddqn_train.py
```

Afterwards, you can see the trained agent in action with the command:
```
python3 ./rftorch/examples/dqn/lunar_lander_ddqn_run.py
```

## Acknowledgements
Code for the implementation of the reinforcement learning agents in this library is inspired by the work of [Phil Tabor](https://github.com/philtabor). Please check his youtube channel [here](https://www.youtube.com/c/MachineLearningwithPhil).