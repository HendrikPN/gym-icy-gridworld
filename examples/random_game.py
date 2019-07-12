import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../gym_icy_gridworld/envs'))
from icy_gridworld_env import IcyGridWorldEnv

ENV_PARAMS = {"grid_size": 20, "agent_id": 0, "max_steps": 100}

env = IcyGridWorldEnv(**ENV_PARAMS)

# Play a few episodes of a random game, and render.
for i in range(3000):
    observation = env.reset()
    done = False
    env.render()
    while not done:
        (observation, reward, done) = env.step(np.random.choice(range(2)))
        # print(f'Idealized observation: {env.idealize_observation()} as (d, v_1, v_2 x)')
        env.render()
