import gym
import numpy as np
from agent import DDPG_Agent

# The continuous lunar lander enviroment is loaded
env = gym.make('LunarLanderContinuous-v2')
agent = DDPG_Agent(tau=1)
score_history = []
n_episodes = 3000
chkpt_update = 10

# agent is trained in the enviroment
for i in range(n_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = agent.get_train_action(state).cpu().detach().numpy()
        new_state, reward, done, info = env.step(action)
        agent.save_memory(state, action, reward, new_state, done)
        agent.learn()
        state = new_state
        score += reward
    score_history.append(score)
    print("Episode %d - Score: %.2f - Average score: %.2f" % (i, score, np.mean(score_history[-100:])))

    # actor and critic models are saved every chkpt_update episodes
    if i%chkpt_update==0:
        agent.actor.save_checkpoint()
        agent.critic.save_checkpoint()