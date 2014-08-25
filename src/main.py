"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from memory.memoryd import MemoryD
from ale.ale import ALE
import games
import random, os, cPickle, time, numpy as np

for direction in ['in', 'out']:
    path = 'ale_fifo_' + direction
    if os.path.exists(path):
        os.remove(path)

savedir = '/var/tmp/nets/'
if os.path.exists(savedir):
    os.rename(savedir, savedir[:-1] + '-' + str(time.time()))
os.mkdir(savedir)

class Main:

    def __init__(self):
        self.minibatch_size = 32  # Given in the paper
        self.number_of_actions = 4  # XXX Game "Breakout" has 4 possible actions

        # Properties of the neural net which come from the paper
        self.nnet = NeuralNet([1, 4, 84, 84], filter_shapes=[[16, 4, 8, 8], [32, 16, 4, 4]],
                              strides=[4, 2], n_hidden=256, n_out=self.number_of_actions)
        self.ale = ALE()
        self.frames_played = 0
        self.iterations_per_choice = 4
        self.games = games.Games()

    def compute_epsilon(self):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        return max(1 - 0.9* (float(self.frames_played) / 1e6), 0.1)

    def play_game(self, training_callback):
        """Play  a game,  calling  training_callback every  iteration.
        Returns a history of the game."""
        game = games.Game(self.ale)
        while not game.game_over():
            if ((random.uniform(0,1) < self.compute_epsilon()) or
                (game.number_frames() < 4)):
                action = random.choice(range(self.number_of_actions))
                print 'chose randomly', action, 
            else:
                rewards = self.nnet.predict_rewards([game.last_state()])
                action = np.argmax(rewards)
                print 'chose q-func  ', action, 'based on rewards', rewards, 
            self.frames_played += 1
            game.move(action)
            print 'reward', game.rewards[-1]
            if (self.frames_played % self.minibatch_size) == 0:
                training_callback()
        return game

    def play_games(self):
        while True:
            if self.games.number_unprocessed() < 20 * self.minibatch_size:
                training_callback = lambda: None
            else:
                training_callback = lambda: self.nnet.train(
                    self.games.get_minibatch(self.minibatch_size))
            print 'Currently', len(self.games.games)+1, 'games'
            self.games.add_game(self.play_game(training_callback))
            
if __name__ == '__main__':
    m = Main()
    m.play_games()
