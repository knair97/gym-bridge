import numpy as np
import gym
import gym_bridge
import time

# A simple test script to see if the environment is working
# Actions are taken at random

env = gym.make('bridge-v0')

env.reset()
done = False

print("Succesfully setup environment. \nRunning and rendering sample game...\n")
env.render()
while not done:

    action = env.sample_action()
    _, _, done, _ = env.step(action)
    
    env.render()
    
n_games = 1000
T_tot = 0

print("Timing %d games..." % n_games)
for g in range(n_games):
    
    t = time.time()
    env.reset()
    done = False
    
    while not done:
        
        action = env.sample_action()
        _, _, done, _ = env.step(action)
        
    T_tot += time.time() - t
    
env.close()

print("Game rate: %.0f games/s" % (n_games / T_tot))