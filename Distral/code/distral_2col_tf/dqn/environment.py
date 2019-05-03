import gym
import random
from .utils import imresize
from .utils import rgb2gray


class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)

        screen_width, screen_height, self.action_repeat, self.random_start = \
            config.screen_width, config.screen_height, config.action_repeat, config.random_start

        self.dims = (screen_width, screen_height)
        self._screen = None
        self.reward = 0
        self.terminal = True

        self.info = None # 这个是为了改一个gym 版本不匹配的bug

        self.display = config.display

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._random_step()
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, self.info = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self):
        self.render()

    @property
    def lives(self):
        if self.info is None:
            return 0
        else:
            return self.info['ale.lives']

    @property
    def screen(self):
        return imresize(rgb2gray(self._screen)/255.0, self.dims)

    @property
    def srt(self):
        # state, reward, terminal
        return self.screen, self.reward, self.terminal

    @property
    def action_size(self):
        return self.env.action_space.n


class SimpleGymEnvironment(Environment):
    def __init__(self, config):
        super(SimpleGymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        self._step(action)
        self.after_act()
        return self.srt


class GymEnvironment(Environment):
    def __init__(self, config):
        super(GymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        accumulated_reward = 0
        for _ in range(self.action_repeat):
            self._step(action)
            accumulated_reward += self.reward

            if self.terminal:
                break

        self.reward = accumulated_reward
        self.after_act()
        return self.srt
