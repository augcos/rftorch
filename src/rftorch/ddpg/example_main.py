# This is an example script to run a DDPG agent that learns how to play the LunarLanderContinuous-v2 enviroment. The 
# agent uses the models found in example_networks.py
import gym
import numpy as np
from rftorch.ddpg.agent import DDPGAgent

if __name__=="__main__":
    # The continuous lunar lander enviroment is loaded
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPGAgent(tau=1)
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