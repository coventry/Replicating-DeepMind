"""

Memory stores game data and provides means work with it

"""

import numpy as np
import random

class MemoryD:

    #: N x 84 x 84 matrix, where N is the number of game steps we want to keep
    screens = None
    
    #: list of size N, stores actions agent took
    actions = None
    
    #: list of size N, stores rewards agent received upon doing an action
    rewards = None

    #: list of size N, stores game internal time
    time = None

    #: global index counter
    count = None

    def __init__(self, n):
        """
        Initialize memory structure 
        :type self: object
        @param n: the number of game steps we need to store
        """
        self.n = n
        self.screens = np.zeros((n, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((n,), dtype=np.uint8)
        self.rewards = np.zeros((n,), dtype=np.uint8)
        self.time = np.zeros((n,), dtype=np.uint32)
        self.count = -1

    def next_count(self, count):
        return (count + 1) % self.n

    def add_first(self, next_screen):
        """
        When a new game start we add initial game screen to the memory
        @param next_screen: 84 x 84 np.uint8 matrix
        """
        next_count = self.next_count(self.count)
        self.screens[next_count] = next_screen
        self.time[next_count] = 0
        self.count = next_count

    def add(self, action, reward, next_screen):
        """
        During the game we add few thing to memory
        @param action: the action agent decided to do
        @param reward: the reward agent received for his action
        @param next_screen: next screen of the game
        """
        next_count = self.next_count(self.count)
        self.actions[self.count] = action
        # Since the scale of scores  varies greatly from game to game,
        # we fixed all positive reward to be 1 and all negative
        # rewards to be  -1, leaving 0 rewards unchanged.
        self.rewards[self.count] = np.sign(reward)
        self.time[next_count] = self.time[self.count] + 1
        self.screens[next_count] = next_screen
        self.count = next_count

    def add_last(self):
        """
        When the game ends, there is  no need to record it, but it
        should be penalized.
        """
        # Penalize the last action
        self.rewards[self.count] -= 1

    def get_minibatch(self, size):
        """
        Take n Transitions from the memory.
        One transition includes (state, action, reward, state)
        @param size: size of the minibatch (in our case it should be 32)
        """

        transitions = []

        #: Pick random n indices and save dictionary if not terminal state
        while len(transitions) < size:
            i = random.randint(0, self.count - 1)
            if self.time[i] >= 3:
                transitions.append({'prestate': self.get_state(i),
                                    'action': self.actions[i],
                                    'reward': self.rewards[i],
                                    'poststate': self.get_state(self.next_count(i))})

        return transitions

    def get_state(self, index):
        """
        Extract one state (4 images) given last image position
        @param index: global location of the 4th image in the memory
        """
        assert self.time[index] >= 3
        current_idx = (index - 3 + self.n) % self.n
        state = []
        for dummy in range(4):
            state.append(self.screens[current_idx])
            current_idx = self.next_count(current_idx)
        assert current_idx == self.next_count(index)
        return state

    def get_last_state(self):
        """
        Get last 4 images from the memory. Those images will go as an input for
        the neural network
        """
        return self.get_state(self.count)
