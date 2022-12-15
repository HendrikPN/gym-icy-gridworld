import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from operator import add, sub
import typing
from typing import List, Tuple
import cv2
from collections import deque

class IcyGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        """
        This environment is a 1 x N gridworld on a torus with two objects
        (square and star) which can be moved independent of each other by two
        agents.
        Starting the environment, a single agent is assigned either the square or
        star which it can accelerate by each action while the other object is
        fixed. The agent observes the whole world and has to find a way to
        quickly capture the other object in order to obtain a reward = 1/v where
        v is the current speed of the agent.
        By default there is no restriction of the number of steps.
        The agent's speed is limited to 1/2 the size of the grid. It cannot
        exceed this speed. At each reset, the objects are moved to a random
        position on the grid at a certain minimum distance from each other.

        Args:
            **kwargs:
                grid_size (int): The size of the 1D grid. Defaults to 20.
                agent_id (int): The agent id. Two agents are allowed, i.e. 0,1. Defaults to 0.
                max_steps (int): The maximum number of allowed time steps. Defaults to 0, i.e. no restriction.
        """
        if 'grid_size' in kwargs and type(kwargs['grid_size']) is int:
            setattr(self, '_grid_size', kwargs['grid_size'])
        else:
            setattr(self, '_grid_size', 20)
        if 'agent_id' in kwargs and type(kwargs['agent_id']) is int:
            setattr(self, '_agent_id', kwargs['agent_id'])
        else:
            setattr(self, '_agent_id', 0)
        if 'max_steps' in kwargs and type(kwargs['max_steps']) is int:
            setattr(self, '_max_steps', kwargs['max_steps'])
        else:
            setattr(self, '_max_steps', 0)
        self._img_size = np.array([1, self._grid_size]) * 7 #: image size is [7, size * 7] pixels

        #:class:`gym.Box`: The specifications of the image to be used as observation.
        self.observation_space=gym.spaces.Box(low=0, high=1, shape=(self._img_size[0],self._img_size[1]), dtype=np.float32)
        #:class:`gym.Discrete`: The space of actions available to the agent.
        self.action_space=gym.spaces.Discrete(2)

        #array of int: The current position of the agents.
        self._agents_pos = [0, 0]
        #array of int: The current velocity of the agents.
        self._agents_velocity = [0, 0]
        
        #function: Sets the static part of the observed image, i.e. walls.
        self._get_static_image()
        #np.array of float: The currently observed image.
        self._img_current = np.zeros(self.observation_space.shape)
        #np.array of float: The previously observed image.
        self._img_previous = np.zeros(self.observation_space.shape)
        #int: Number of time steps since last reset.
        self._time_step = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        An action increases the velocity of an object in one direction.
        A move is forced according to the object's velocity.
        The object moves on a torus and reappears on the other side when crossing the border.
        If the maximum number of steps is exceeded, the agent receives a negative reward, and the game resets.
        The speed is limited by 1/2 the size of the grid.
        Once the other object is hit, the reward is rescaled by the inverse velocity.

        Args:
            action (int): The index of the action to be taken.

        Returns:
            observation (numpy.ndarray): An array representing the current and the previous image of the environment.
            reward (float): The reward given after this time step.
            done (bool): The information whether or not the episode is finished.

        """
        # Action increases velocity according to direction. 
        if action == 0:
            self._agents_velocity[self._agent_id] += 1
        elif action == 1:
            self._agents_velocity[self._agent_id] += -1
        else:
            raise TypeError(f'The action is not valid. The action should be an integer 0 <= action <= {self.action_space.n}')

        # Maximum speed is given by the grid size.
        if abs(self._agents_velocity[self._agent_id]) > int(self._grid_size/2):
            self._agents_velocity[self._agent_id] = int(self._grid_size/2)

        # Move according to velocity.
        self._agents_pos = list(map(add, self._agents_pos, self._agents_velocity))

        # If agent crosses the boundary of the gridworld, it is moved to the other side.
        for index, pos in enumerate(self._agents_pos):
            if pos >= self._grid_size:
                self._agents_pos[self._agent_id] = pos - self._grid_size
            elif pos < 0:
                self._agents_pos[self._agent_id] = self._grid_size + pos

        # Check whether reward was found. Last step may get rewarded.
        # Reward is rescaled by velocity.
        self._time_step += 1
        if self._agents_pos[0] == self._agents_pos[1]:
            rescale = abs(self._agents_velocity[self._agent_id])
            reward = 1. * 1/rescale if rescale != 0 else 1.
            done = True
        # Check whether maximum number of time steps has been reached.
        elif self._max_steps and self._time_step >= self._max_steps:
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
        Objects are reset to a random position with velocity 0 and a distance
        >=2 to each other. 

        Returns:
            observation (numpy.ndarray): An array representing the current and the previous image of the environment.
        """
        # Reset internal timer.
        self._time_step = 0

        # Place the agents randomly at distance at least 2.
        dist = 0
        while dist < 2:
            self._agents_pos = [np.random.choice(range(self._grid_size)), np.random.choice(range(self._grid_size))]
            dist = abs(self._agents_pos[0] - self._agents_pos[1])

        # Reset velocity.
        self._agents_velocity = [0, 0]

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
    
    def idealize_observation(self):
        """
        Calculates an ideal representation of the observation. 
        The representation is given by (distance, velocity_1, velocity_2, position_1).

        Returns:
            observation (numpy.ndarray): The ideal representation as observation.
        """
        distance = [self._agents_pos[0] - self._agents_pos[1]]
        velocity = self._agents_velocity
        position = [self._agents_pos[0]]
        observation = np.array([[*distance, *velocity, *position]])
        return observation

    def simplify_observation(self):
        """
        Calculates an simplified representation of the observation. 
        The representation is given by (position_1, position_2, position_3, position_4)
        where position_3 and _4 are the positions on the previous image.

        Returns:
            observation (numpy.ndarray): The simplified representation as observation.
        """
        pos_current = [self._agents_pos[0], self._agents_pos[1]]
        pos_previous = list(map(sub, self._agents_pos, self._agents_velocity))

        # If agent crosses the boundary of the gridworld, it is moved to the other side.
        for index, pos in enumerate(pos_previous):
            if pos >= self._grid_size:
                pos_previous[self._agent_id] = pos - self._grid_size
            elif pos < 0:
                pos_previous[self._agent_id] = self._grid_size + pos
        observation = np.array([[*pos_current, *pos_previous]])
        return observation

    def get_optimal_sequence(self) -> deque:
        """
        Given an initial state, returns the optimal action sequence.

        Returns:
            action_set (numpy.ndarray): The optimal action sequence.
        """
        if (np.array(self._agents_velocity) > 0.).any() or self._time_step > 0:
            raise ValueError("Cannot evaluate optimal policy since the provided observation is not initial.")
        if self._grid_size != 20:
            raise NotImplementedError('We can only generate the optimal policy for grid size 20.')
        
        #
        distance = self._agents_pos[self._agent_id] - self._agents_pos[(self._agent_id+1)%2]
        if distance > 0:
            actions = [1, 0] # reward is left to agent on grid
        else:
            distance = -1. * distance
            actions = [0, 1] # reward is right to agent on grid
        if distance > int(self._grid_size/2.): 
            actions = list(reversed(actions)) # reward is closer in the other direction
            distance = self._grid_size - distance
        
        return self._generate_optimal_sequence(distance, actions)
 
    # ----------------- helper methods ---------------------------------------------------------------------

    def _get_static_image(self) -> None:
        """
        Generate the static part of the gridworld image, images of the objects.

        NOTE: By default, the gridworld on a torus does not have any walls.
        """
        # Empty world.
        gridworld = np.zeros(self.observation_space.shape)

        # No walls.
        
        #array of float: The static part of the gridworld image, i.e. walls.
        self._img_static = gridworld

        # Draw first agent image, i.e. a square.
        square_draw = np.zeros((7,7))
        for i in range(7):
            square_draw[0, i] = 1.
            square_draw[6, i] = 1.
            square_draw[i, 0] = 1.
            square_draw[i, 6] = 1.

        #array of float: The static 7 x 7 image of the first agent.
        self._img_square = square_draw

        # Draw second agent image, i.e. a star.
        star_draw = np.zeros((7,7))
        for i in range(7):
            star_draw[i, i] = 1.
            star_draw[i, 6-i] = 1.
            star_draw[3, i] = 1.
            star_draw[i, 3] = 1.

        #array of float: The static 7 x 7 image of the second agent.
        self._img_star= star_draw
    
    def _get_image(self) -> np.ndarray:
        """
        Generate an image from the current state of the environment.

        Returns:
            image (numpy.ndarray): An array representing an image of the environment.
        """
        image = self._img_static.copy()
        #np.ndarray: the coordinate for the position of the agent in the image including walls
        square_coord = self._agents_pos[0] * 7
        #np.ndarray: the coordinate for the position of the reward in the image including walls
        star_coord = self._agents_pos[1] * 7

        # Draw agent 1 into static image.
        image[0:7, square_coord:square_coord+7] = self._img_square

        # Draw agent 2 into static image.
        image[0:7, star_coord:star_coord+7] = self._img_star

        return image

    def _get_observation(self) -> np.ndarray:
        """
        Generates an observation from two sequenced images.

        Returns:
            observation (numpy.ndarray): A 2 x 7 x (grid_size*7) array representing a 
                                         time sequence of images of the environment.
        """
        observation = np.concatenate((self._img_previous, self._img_current), axis=0)
        observation = np.reshape(observation, (2,self.observation_space.shape[0],self.observation_space.shape[1]))

        return observation

    def _generate_optimal_sequence(self, distance, actions) -> deque:
        action_sequence = deque()
        # preliminary: if-else
        if distance == 1:
            sequence = [0]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 2 :
            sequence = [0,1,0]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 3 :
            sequence = [0,1,0,1,0]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 4 :
            sequence = [0,0,1]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 5 :
            sequence = [0,0,1,1,0]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 6 :
            sequence = [0,0,1,1,0,1,0]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 7 :
            sequence = [0,0,1,0,1]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 8 :
            sequence = [0,0,1,0,1,1,0]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 9 :
            sequence = [0,0,0,1,1]
            for i in sequence:
                action_sequence.append(actions[i])
        elif distance == 10 :
            sequence = [0,0,0,1,1,1,0]
            for i in sequence:
                action_sequence.append(actions[i])
        else:
            ValueError(f"Distance should not be > 10, instead we got {distance}.")
        
        return action_sequence
