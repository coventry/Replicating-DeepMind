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
        epsilon = max(0.9 - 0.1 * float(frames_played) / self.memory_size, 0.1)
        print 'epsilon', epsilon
        return epsilon

    frames_played = 0
    show_rewards = False

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
                    if self.show_rewards:
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
                images = [] # For detecting static positions
                for i in range(4):
                    # "the agent sees and selects actions on every kth
                    # frame  instead  of  every frame,  and  its  last
                    # action  is repeated  on  skipped frames.   Since
                    # running  the  emulator   forward  for  one  step
                    # requires much  less computation than  having the
                    # agent  select an  action, this  technique allows
                    # the  agent to  play roughly  k times  more games
                    # without  significantly  increasing the  runtime.
                    # We use k = 4 for all games except Space Invaders
                    # where  we noticed  that using  k =  4 makes  the
                    # lasers invisible because of  the period at which
                    # they blink.   We used k  = 3 to make  the lasers
                    # visible
                    self.ale.move(action)

                    # Store  the results  from the  last iteration  of the
                    # chosen action.
                    self.ale.store_step(action)
                    frames_played += 1

                    # Store new information to memory
                    images.append(self.ale.next_image)
                if len(set(images)) < len(images):
                    # In breakout, if  the puck is lost  the next puck
                    # isn't put in play until you hit the fire button.
                    # Here I'm checking whether the board is frozen by
                    # looking  for  identical  images  over  the  four
                    # frames, and hitting 'fire' if I find any.
                    self.ale.move(1)

                # Start a training session

                if (frames_played % self.minibatch_size) == 0:
                    # Don't do any training  until we've got something
                    # large  enough  to  avoid oversampling  from  the
                    # history...  This  is still going to  over sample
                    # eventually,  but it  is exactly  what the  paper
                    # says to do... Hmm.
                    print 'getting minibatch'
                    minibatch = self.memory.get_minibatch(self.minibatch_size)
                    print 'training net'
                    self.nnet.train(minibatch)
                    print 'dumping net...'
                    savepath = savedir + ('%i.pickle' % frames_played)
                    cPickle.dump([minibatch, self.nnet], open(savepath, 'w'))
                    print 'done'

            # After "game over" increase the number of games played
            games_played += 1

            # And do stuff after end game (store information, let ALE know etc)
            self.ale.end_game()


if __name__ == '__main__':
    m = Main()
    m.play_games(1000)
