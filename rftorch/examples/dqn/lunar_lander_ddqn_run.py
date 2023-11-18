# This is an example script to run a DQN agent that learns how to play the LunarLander-v2 enviroment. The agent uses the
# models found in example_networks.py
import numpy as np
import gymnasium as gym
import pickle

if __name__=="__main__":
    # the lunar lander enviroment is loaded
    env = gym.make("LunarLander-v2", render_mode="human")
    n_episodes = 1000
    max_steps = 1000
    score_history = []
    eps_history = []
    
    with open('lunar_lander_ddqn.pkl', 'rb') as inp:
        agent = pickle.load(inp)
    
    # agent is trained in the enviroment
    for i in range(n_episodes):
        done = False
        score = 0
        step = 0
        state, _ = env.reset()

        # while episode is not finished, learning is performed
        while not done:
            action = agent.predict(state, train=False)
            new_state, reward, done, _, info = env.step(action)

            state = new_state
            score += reward
            step += 1
            if step>=max_steps:
                done = True
        
        # episode score is appended to score history and average of the last 100 episodes is printed
        score_history.append(score)
        print("Episode %d - Step %d - Score: %.2f - Average score: %.2f - Epsilon: %.2f" 
              % (i, step, score, np.mean(score_history[-100:]), agent.eps))
