import gym
import numpy as np
from agent import DQN_agent
from example_networks import ExampleDeepQNetwork

# The lunar lander enviroment is loaded
env = gym.make('LunarLander-v2')
agent = DQN_agent()
score_history = []
eps_history = []
n_episodes = 1000
chkpt_update = 10

# agent is trained in the enviroment
for i in range(n_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = agent.get_train_action(state)
        new_state, reward, done, info = env.step(action)
        agent.save_memory(state, action, reward, new_state, done)
        agent.learn()
        state = new_state
        score += reward
    score_history.append(score)
    print("Episode %d - Score: %.2f - Average score: %.2f - Epsilon: %.2f" 
            % (i, score, np.mean(score_history[-100:]), agent.epsilon))

    # actor and critic models are saved every chkpt_update episodes
    if i%chkpt_update==0:
        agent.DQN_network.save_checkpoint()