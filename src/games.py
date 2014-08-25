import random, numpy as np
from collections import deque
from ale.preprocessor import Preprocessor
from ale import ale

clean_image = Preprocessor().process

class Game:

    """Keep  track of  the actions  and transitions  in a  given game.
    This  is  used  to  track  actions  backwards  from  actions  with
    nontrivial rewards."""

    def __init__(self, ale):
        # A bit  ugly that  different games are  sharing the  same ale
        # instance.  Should make a new one for each game?
        self.ale = ale
        if ale is not None:
            ale.end_game()
            ale.new_game()
        self.screens, self.actions, self.rewards = [], [], []
        # Set of frame indices with non-zero reward
        self.reward_frames = set()
        self.current_training_frame = None
        self.game_over_p = False

    def new_training_frame(self):
        """Randomly  choose  a  frame  which  received  a  non-trivial
        reward,  to  start  the  next round  of  training  from.   The
        simplest way of  doing this would be to  simply work backwards
        through  the   entire  game  history,  but   that  would  risk
        correlations between  the events  in the different  games from
        which a minibatch  is selected.  E.g., all games  end with the
        end-of-game penalty."""
        assert self.reward_frames
        self.current_training_frame = random.sample(self.reward_frames, 1).pop()
        self.reward_frames.remove(self.current_training_frame)
        return self.current_training_frame

    def number_frames(self):
        return len(self.screens)

    def game_over(self):
        return self.game_over_p

    def get_state(self, index):
        "Return the last four screens"
        assert index >= 4-1
        return self.screens[index-(4-1):index+1]

    def last_state(self):
        return self.get_state(len(self.screens)-1)

    def move(self, action):
        assert not self.game_over_p
        self.ale.move(action)
        self.screens.append(clean_image(self.ale.next_image))
        self.actions.append(action)
        self.game_over_p = self.ale.game_over
        # If this  action led  to the  end of the  game, it  should be
        # penalized.  XXX This penalty is  not mentioned in the paper,
        # and in breakout it could be a significant advantage, because
        # when the  paddle is near the  point where the puck  runs off
        # the screen  at the  end of the  game, the  consequences will
        # propagate  through the  Q  function  quite quickly,  whereas
        # rewards for  hitting the  barrier at the  top of  the screen
        # occur  much  later,  following correct  positioning  of  the
        # paddle.
        reward = -1 if self.game_over_p else np.sign(self.ale.current_reward) 
        self.rewards.append(reward)
        if reward:
            self.reward_frames.add(len(self.rewards)-1)

    def pop_transition(self):
        """Return the  most recent transition in  the current sequence
        of frames."""
        # Only use this  game if it's finished playing,  give the best
        # chance to work backwards through the results.
        assert self.game_over_p
        if self.current_training_frame is None:
            self.new_training_frame()
        game_end = self.current_training_frame == len(self.screens)-1
        transition = {'prestate': self.get_state(self.current_training_frame),
                      'action': self.actions[self.current_training_frame],
                      'reward': self.rewards[self.current_training_frame],
                      'poststate': (self.get_state(self.current_training_frame+1)
                                    if (not game_end) else None),
                      'game_end': game_end}
        if self.current_training_frame in self.reward_frames:
            self.reward_frames.remove(self.current_training_frame)
        self.current_training_frame -= 1
        if (self.current_training_frame < (4-1)) and self.reward_frames:
            self.new_training_frame()
        return transition

    def transitions_remaining_p(self):
        return (bool(self.reward_frames) or
                ((self.current_training_frame is not None) and
                 (self.current_training_frame >= 3)))

    def __getinitargs__(self):
        return [None]

    def __getstate__(self):
        # Don't pickle the ALE instance
        assert self.game_over()
        return dict(
            (attr, getattr(self, attr))
            for attr in 'screens actions rewards reward_frames current_training_frame game_over_p'.split())

class Games:

    def __init__(self):
        self.games = deque()

    def add_game(self, game):
        self.games.append(game)

    def get_minibatch(self, minibatch_size):
        assert len(self.games) >= minibatch_size
        minibatch = []
        for dummy in minibatch_size * [None]:
            game = self.games.pop()
            minibatch.append(game.pop_transition())
            if game.transitions_remaining_p():
                self.games.appendleft(game)
        return minibatch

    def number_unprocessed(self):
        return len(self.games)
        
