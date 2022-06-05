import gym
import numpy as np
from agent import Discrete_PPO_Agent
from example_networks import ExampleActor, ExampleCritic

# The lunar lander enviroment is loaded
env = gym.make('CartPole-v0')
agent = Discrete_PPO_Agent()
N=20
score_history = []
n_episodes = 1000
chkpt_update = 10
n_steps = 0
learn_iters = 0

# agent is trained in the enviroment
for i in range(n_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action, prob, val = agent.get_eval_action(state)
        new_state, reward, done, info = env.step(action)
        agent.save_memory(state, action, prob, val, reward, done)
        n_steps += 1
        score += reward
        if n_steps%N==0:
            agent.learn()
            learn_iters += 1
        state = new_state
    score_history.append(score)
    print("Episode %d - Score: %.2f - Average score: %.2f - Learning steps: %d" 
            % (i, score, np.mean(score_history[-100:]), learn_iters))
