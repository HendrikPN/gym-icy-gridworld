# gym-icy-gridworld

An openAI gym environment for representation learning within a reinforcement learning (RL) setting.
This environment is build in accordance with the [openAI gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md#how-to-create-new-environments-for-gym)
policy for standardized RL environments.

## GymIcyGridworldEnv

This environment is an empty N x M gridworld on a torus without walls. 
At each reset, the agent reset randomly, and the reward is placed randomly at a minimum distance from the agent.
The observation at each time step consists of two images of the whole environment for the last two sequential time steps.
The images are (N * 7 x M * 7) pixels and include the position of both agent and reward.
Unlike the common gridworld environment this environment can be icy, which means that each action accelerates the agent in one direction. The velocity is limited by the grid size.
The goal of the agent is to quickly gather the reward at each episode. The reward is obtained only if the agent lands exactly on the rewarded position and is lower for higher speeds.

* Notes: 
    * Simplification: By choosing N = 1 *or* M = 1 the environment becomes effectively one-dimensional. Still, 4 actions are possible.
    * Complication: The environment can be made partially observable if the acceleration is set to >=1 and the first image of the observation is ignored.

# Installation

```
git clone -b master https://github.com/HendrikPN/gym-icy-gridworld.git
cd gym-icy-gridworld
pip install --user -e .
``` 
