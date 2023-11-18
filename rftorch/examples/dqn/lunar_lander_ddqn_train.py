# This is an example script to run a DQN agent that learns how to play the LunarLander-v2 enviroment. The agent uses the
# models found in example_networks.py
import numpy as np
import gymnasium as gym

from rftorch.agents.dqn import DuelingDQNAgent
from rftorch.examples.dqn.networks import DuelingDeepQNetwork

if __name__=="__main__":
    # the lunar lander enviroment is loaded
    env = gym.make("LunarLander-v2")

    # we create a Dueling DQN agent
    critic = DuelingDeepQNetwork(fc1_dims=1024, 
                                 fc2_dims=512,
                                 lr=5e-4, 
                                 input_shape=env.observation_space.shape, 
                                 n_actions=env.action_space.n)
    
    agent = DuelingDQNAgent(critic, 
                            input_shape=env.observation_space.shape, 
                            n_actions=env.action_space.n, 
                            d_eps=1e-5, 
                            tau=500, 
                            batch_size=64)

    # auxiliar variables
    score_history = []
    eps_history = []
    n_episodes = 3000
    max_steps = 1000
    chkpt_update = 100

    # agent is trained in the enviroment
    for i in range(n_episodes):
        done = False
        score = 0
        steps=0
        state, _ = env.reset()

        # while episode is not finished, learning is performed
        while not done:
            action = agent.predict(state, train=True)
            new_state, reward, done, _, info = env.step(action)
            agent.step(state, action, reward, new_state, done)
            
            state = new_state
            score += reward
            steps += 1
            if steps>=max_steps:
                done = True
        
        # episode score is appended to score history and average of the last 100 episodes is printed
        score_history.append(score)
        print("Episode %d - Step %d - Score: %.2f - Average score: %.2f - Epsilon: %.2f" 
              % (i, steps, score, np.mean(score_history[-100:]), agent.eps))

        # model is saved every chkpt_update episodes
        if i % chkpt_update==0:
            agent.save('lunar_lander_ddqn.pkl')
