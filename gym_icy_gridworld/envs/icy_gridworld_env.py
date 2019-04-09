import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from operator import add
import typing
from typing import List, Tuple
import cv2

class IcyGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        """
        This environment is a common N x M gridworld where the agent is accelerated by each action. 
        The agent observes the whole world and has to find a way to quickly gather the reward = +1. 
        A reward is obtained once the agent lands exactly on the rewarded position. By default the reward is 0.
        An episode ends if the reward has been obtained or the maximum number of steps is exceeded. 
        By default there is no restriction ot the number of steps.
        At each reset, the agent is moved to its initial position and the reward is placed randomly
        close to the border.

        Args:
            **kwargs:
                grid_size (:obj:`list` of :obj:`int`): The size of the grid. Defaults to [10, 10].
                                                       If an element is 1, the gridworld becomes one-dimensional.
                acceleration (int): The acceleration of the environment. Defaults to 0.
                max_steps (int): The maximum number of allowed time steps. Defaults to 0, i.e. no restriction.
        """
        if 'grid_size' in kwargs and type(kwargs['grid_size']) is list:
            setattr(self, '_grid_size', kwargs['grid_size'])
        else:
            setattr(self, '_grid_size', [10, 10])
        if 'acceleration' in kwargs and type(kwargs['acceleration']) is int:
            setattr(self, '_acceleration', kwargs['acceleration'])
        else:
            setattr(self, '_acceleration', 0)
        if 'max_steps' in kwargs and type(kwargs['max_steps']) is int:
            setattr(self, '_max_steps', kwargs['max_steps'])
        else:
            setattr(self, '_max_steps', 0)
        self._img_size = (np.array(self._grid_size) + np.array([2, 2])) * 7 #: image size is [(grid_size + walls) * 7]^2 pixels

        #:class:`gym.Box`: The specifications of the image to be used as observation.
        self.observation_space=gym.spaces.Box(low=0, high=1, shape=(self._img_size[0],self._img_size[1]), dtype=np.float32)
        #:class:`gym.Discrete`: The space of actions available to the agent.
        self.action_space=gym.spaces.Discrete(4)
        
        #array of int: The initial position of the agent.
        self._agent_init = [int(self._grid_size[0]/2), int(self._grid_size[1]/2)]
        #array of int: The current position of the agent.
        self._agent_pos = self._agent_init
        #array of int: The current velocity of the agent in all directions.
        self._agent_velocity = [0, 0]
        #array of int: The current position of the reward.
        self._reward_pos = [0, 0]
        
        #function: Sets the static part of the observed image, i.e. walls.
        self._get_static_image()
        #np.array of float: The currently observed image.
        self._img_current = np.zeros(self.observation_space.shape)
        #np.array of float: The previously observed image.
        self._img_previous = np.zeros(self.observation_space.shape)
        #int: Number of time steps since last reset.
        self._time_steps = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        An action increases the velocity of an agent in one direction. and forces a move in that direction according to its speed.
        A move is forced according to the agent's velocity.
        If the agent encounters a wall along one direction, the velocity is set to zero in that direction and the agent stops.
        If the maximum number of steps is exceeded the agent receives a negative reward.

        Args:
            action (int): The index of the action to be taken.

        Returns:
            observation (numpy.ndarray): An array representing the current and the previous image of the environment.
            reward (float): The reward given after this time step.
            done (bool): The information whether or not the episode is finished.

        """
        # Action increases velocity according to acceleration. 
        # If acceleration is 0, velocity is reset at each step.
        if not self._acceleration:
            self._agent_velocity = [0, 0]
        if action == 0:
            self._agent_velocity[0] += 1 * self._acceleration if self._acceleration else 1
        elif action == 1:
            self._agent_velocity[1] += 1 * self._acceleration if self._acceleration else 1
        elif action == 2:
            self._agent_velocity[0] += -1 * self._acceleration if self._acceleration else -1
        elif action == 3:
            self._agent_velocity[1] += -1 * self._acceleration if self._acceleration else -1
        else:
            raise TypeError(f'The action is not valid. The action should be an integer 0 <= action <= {self.action_space.n}')

        # Move according to velocity.
        self._agent_pos = list(map(add, self._agent_pos, self._agent_velocity))

        # If agent would hit a wall, set velocity to zero and stop agent.
        for index, pos in enumerate(self._agent_pos):
            if pos >= self._grid_size[index]:
                self._agent_velocity[index] = 0
                self._agent_pos[index] = self._grid_size[index] - 1
            elif pos < 0:
                self._agent_velocity[index] = 0
                self._agent_pos[index] = 0

        # Check whether reward was found. Last step may get rewarded.
        if self._agent_pos == self._reward_pos:
            reward = 1.
            done = True
        # Check whether maximum number of time steps has been reached.
        elif self._max_steps and self._time_steps >= self._max_steps:
            reward = -1.
            done = True
        # Continue otherwise.
        else:
            reward = 0.
            done = False

        # Set previous image
        self._img_previous = self._img_current
        # Create a new image and observation.
        self._img_current = self._get_image()
        observation = self._get_observation()

        return (observation, reward, done)

    def reset(self) -> np.ndarray:
        """
        Agent is reset to the initial position with velocity 0. 
        Reward is placed randomly at a position <= grid_size/3 from the outer wall if the size permits it.

        Returns:
            observation (numpy.ndarray): An array representing the current and the previous image of the environment.
        """
        # Reset internal timer.
        self._time_steps = 0

        # Place the agent.
        self._agent_pos = self._agent_init

        # Place reward at random at distance <=grid_size/3 if dimensionality permits.
        if self._grid_size[0] < 3:
            dist_x = 1
        else:
            dist_x = 3
        if self._grid_size[1] < 3:
            dist_y = 1
        else:
            dist_y = 3
        distance = [np.random.choice(range(int(self._grid_size[0]/dist_x))), np.random.choice(range(int(self._grid_size[1]/dist_y)))]
        if np.random.choice([0,1]):
            self._reward_pos[0] = distance[0]
        else:
            self._reward_pos[0] = self._grid_size[0] - distance[0] - 1
        if np.random.choice([0,1]):
            self._reward_pos[1] = distance[1]
        else:
            self._reward_pos[1] = self._grid_size[1] - distance[1] - 1

        # Create initial image.
        self._img_current = self._get_image()
        self._img_previous = self._img_current

        return self._get_observation()

    def render(self, mode: str ='human') -> None:
        """
        Renders the current state of the environment as an image in a popup window.

        Args:
            mode (str): The mode in which the image is rendered. Defaults to 'human' for human-friendly. 
                        Currently, only 'human' is supported.
        """
        if mode == 'human':
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600,600)
            cv2.imshow('image',np.uint8(self._img_current * 255))
            cv2.waitKey(50)
        else:
            raise NotImplementedError('We only support `human` render mode.')
 
    # ----------------- helper methods ---------------------------------------------------------------------

    def _get_static_image(self) -> None:
        """
        Generate the static part of the gridworld image, i.e. walls, image of the agent and reward.
        """
        # Empty world.
        gridworld = np.zeros(self.observation_space.shape)

        # Draw outer walls.
        walls_coord = []
        for i in range(self.observation_space.shape[0]):
            for j in range(7):
                walls_coord.append([i, j])
        for i in range(self.observation_space.shape[0]):
            for j in range(7):
                walls_coord.append([i, self.observation_space.shape[1] - j - 1])
        for i in range(7):
            for j in range(self.observation_space.shape[1]):
                walls_coord.append([i, j])
        for i in range(7):
            for j in range(self.observation_space.shape[1]):
                walls_coord.append([self.observation_space.shape[0] - i - 1, j])

        for wall in walls_coord:
            gridworld[wall[0], wall[1]] = 1.
        
        #array of float: The static part of the gridworld image, i.e. walls.
        self._img_static = gridworld

        # Draw agent image.
        agent_draw = np.zeros((7,7))
        agent_draw[0, 3] = 0.8
        agent_draw[1, 0:7] = 0.9
        agent_draw[2, 2:5] = 0.9
        agent_draw[3, 2:5] = 0.9
        agent_draw[4, 2] = 0.9
        agent_draw[4, 4] = 0.9
        agent_draw[5, 2] = 0.9
        agent_draw[5, 4] = 0.9
        agent_draw[6, 1:3] = 0.9
        agent_draw[6, 4:6] = 0.9

        #array of float: The static 7 x 7 image of the agent.
        self._img_agent = agent_draw

        # Draw reward image.
        reward_draw = np.zeros((7,7))
        for i in range(7):
            reward_draw[i, i] = 0.7
            reward_draw[i, 6-i] = 0.7

        #array of float: The static 7 x 7 image of the reward.
        self._img_reward = reward_draw
    
    def _get_image(self) -> np.ndarray:
        """
        Generate an image from the current state of the environment.

        Returns:
            image (numpy.ndarray): An array representing an image of the environment.
        """
        image = self._img_static.copy()
        #np.ndarray: the coordinate for the position of the agent in the image including walls
        agent_coord = np.array(self._agent_pos) * 7 + [7,7]
        #np.ndarray: the coordinate for the position of the reward in the image including walls
        reward_coord = np.array(self._reward_pos) * 7 + [7,7]

        # Draw agent into static image.
        image[agent_coord[0]:agent_coord[0]+7, agent_coord[1]:agent_coord[1]+7] = self._img_agent

        # Draw reward into static image.
        image[reward_coord[0]:reward_coord[0]+7, reward_coord[1]:reward_coord[1]+7] = self._img_reward

        return image

    def _get_observation(self) -> np.ndarray:
        """
        Generates an observation from two sequenced images.

        Returns:
            observation (numpy.ndarray): A 2 x (grid_size*7 + 2*7) x (grid_size*7 + 2*7) array representing a 
                                         time sequence of images of the environment.
        """
        observation = np.concatenate((self._img_previous, self._img_current), axis=0)
        observation = np.reshape(observation, (2,self.observation_space.shape[0],self.observation_space.shape[1]))

        return observation
