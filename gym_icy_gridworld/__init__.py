from gym.envs.registration import register

register(
    id='icy-gridworld-v0',
    entry_point='gym_icy_gridworld.envs:IcyGridWorldEnv',
)
