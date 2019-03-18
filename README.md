# gym-icy-gridworld

An openAI gym environment for representation learning within a reinforcement learning (RL) setting.
This environment is build in accrodance with the [openAI gym](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym)
policy for standardized RL environments.

## GymIcyGridworldEnv

This environment is an empty N x M gridworld with outer walls. 
At each reset, the agent is moved to its initial position in the center, and the reward is placed randomly, close to the border.
The observation at each time step consists of two images of the whole environment for the last two sequential time steps.
The images are (N * 7 x M * 7) pixels and include the position of both agent and reward.
Unlike the common gridworld environment this environment can be icy, which means that each action accelerates the agent in one direction.
The goal of the agent is to quickly gather the reward at each episode.

# Installation

```
cd gym-icy-gridworld
pip install --user -e .
```
