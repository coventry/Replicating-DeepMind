"""

This is the main class where all thing are put together

"""

from ai.NeuralNet import NeuralNet
from memory.memoryd import MemoryD
from ale.ale import ALE
import random, os, cPickle, time

for direction in ['in', 'out']:
    path = 'ale_fifo_' + direction
    if os.path.exists(path):
        os.remove(path)

savedir = '/var/tmp/nets/'
if os.path.exists(savedir):
    os.rename(savedir, savedir[:-1] + '-' + str(time.time()))
os.mkdir(savedir)

class Main:
    # How many transitions to keep in memory?
    memory_size = 100000
    # My  machine probably  can't  handle 10  000  000 transitions  IN
    # MEMORY, but  that is the  number in the paper.   Compressing the
    # images would probably work...

    # Memory itself
    memory = None

    # Neural net
    nnet = None

    # Communication with ALE
    ale = None

    # Size of the mini-batch which will be sent to learning in Theano
    minibatch_size = None

    # Number of possible actions in a given game
    number_of_actions = None

    def __init__(self):
        self.memory = MemoryD(self.memory_size)
        self.minibatch_size = 128 # 32  # Given in the paper
        self.number_of_actions = 4  # Game "Breakout" has 4 possible actions

        # Properties of the neural net which come from the paper
        self.nnet = NeuralNet([1, 4, 84, 84], filter_shapes=[[16, 4, 8, 8], [32, 16, 4, 4]],
                              strides=[4, 2], n_hidden=256, n_out=self.number_of_actions)
        self.ale = ALE(self.memory)

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        epsilon = max(0.9 - float(frames_played) / self.memory_size, 0.1)
        print 'epsilon', epsilon
        return epsilon

    frames_played = 0

    def play_games(self, n):
        """
        Main cycle: plays many games and many frames in each game. Also learning is performed.
        @param n: total number of games allowed to play
        """

        games_to_play = n
        games_played = 0
        frames_played = self.frames_played

        # Play games until maximum number is reached
        while games_played < games_to_play:
            # Start a new game
            self.ale.new_game()

            # Play until game is over
            while not self.ale.game_over:

                game_time = self.memory.time[self.memory.count]
                enough_frames = game_time >= 3

                if enough_frames:
                    current_state = self.memory.get_last_state()
                    predicted_values = self.nnet.predict_rewards([current_state])
                    print 'predict_rewards', predicted_values

                # Some times random action is chosen
                U = random.uniform(0, 1)
                # Epsilon decreases over time
                epsilon = self.compute_epsilon(frames_played)
                force_random = U < epsilon

                if (not enough_frames) or force_random:
                    action = random.choice(range(self.number_of_actions))
                    print "chose randomly ", action

                # Usually neural net chooses the best action
                else:
                    assert enough_frames
                    print "chose by neural net"
                    action = self.nnet.predict_best_action([current_state])
                    print action

                # Make the move
                self.ale.move(action)

                # Store new information to memory
                self.ale.store_step(action)

                # Start a training session

                self.nnet.train(self.memory.get_minibatch(self.minibatch_size))

            # After "game over" increase the number of games played
            games_played += 1

            # And do stuff after end game (store information, let ALE know etc)
            self.ale.end_game()


if __name__ == '__main__':
    m = Main()
    m.play_games(1000)
