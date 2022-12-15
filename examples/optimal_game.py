import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../gym_icy_gridworld/envs'))
from icy_gridworld_env import IcyGridWorldEnv

ENV_PARAMS = {"grid_size": 20, "agent_id": 0, "max_steps": 100}

env = IcyGridWorldEnv(**ENV_PARAMS)

# Play a few episodes of a perfect game, and render.
for i in range(100):
    observation = env.reset()
    done = False
    env.render()
    action_seq = env.get_optimal_sequence()
    print(action_seq)
    while not done:
        action = action_seq.popleft()
        (observation, reward, done) = env.step(action)
        # print(f'Idealized observation: {env.idealize_observation()} as (d, v_1, v_2, x)')
        # print(f'Simplified observation: {env.simplify_observation()} as (x_1, x_2, x_3, x_4)')
        env.render()
