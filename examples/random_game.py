import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../gym_icy_gridworld/envs'))
from icy_gridworld_env import IcyGridWorldEnv

env = IcyGridWorldEnv()

# Play a few episodes of a random game, and render.
for i in range(3):
    observation = env.reset()
    done = False
    env.render()
    while not done:
        (observation, reward, done) = env.step(np.random.choice(range(4)))
        env.render()
