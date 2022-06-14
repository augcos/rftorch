# This is an example script to run a A2C agent that learns how to play the CartPole-v0 enviroment. The agent uses the 
# models found in example_networks.py
import gym
import numpy as np
from agent import DiscreteA2CAgent

# The continuous lunar lander enviroment is loaded
env = gym.make('CartPole-v0')
agent = DiscreteA2CAgent()
score_history = []
n_episodes = 2500
chkpt_update = 10

# agent is trained in the enviroment
for i in range(n_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)
        agent.learn(state, reward, new_state, done)
        state = new_state
        score += reward
    score_history.append(score)
    print("Episode %d - Score: %.2f - Average score: %.2f" % (i, score, np.mean(score_history[-100:])))

    # actor and critic models are saved every chkpt_update episodes
    if i%chkpt_update==0:
        agent.actor.save_checkpoint()
        agent.critic.save_checkpoint()