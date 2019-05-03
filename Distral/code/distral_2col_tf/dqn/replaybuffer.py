import numpy as np
import random


class ReplayBuffer:
    # config : memory_size, batch_size, history_length, state_format, screen_height, screen_width,
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size

        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype=np.float16)
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = config.history_length # state使用多少张screens拼接在一起，论文中是4张
        self.state_format = config.state_format
        self.dims = (config.screen_height, config.screen_width)
        # state and next_state
        self.states = np.empty((self.batch_size, self.history_length)+self.dims, dtype=np.float16)
        self.next_states = np.empty((self.batch_size, self.history_length)+self.dims, dtype=np.float16)

        self.count = 0  # 记录总共有多少条记录
        self.current = 0 # 获取当前是第几条

    def add(self, screen, action, reward, terminal):
        self.screens[self.current] = screen
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.count = max(self.current + 1, self.count)
        self.current = (self.current + 1) % self.memory_size

    def __len__(self):
        return self.count

    def clear(self):
        self.current = 0
        self.count = 0

    def getState(self, index):
        assert self.count > 0
        # 每一个样本都要取self.history_length那么长。
        if index >= self.history_length - 1:
            return self.screens[index-(self.history_length - 1):index+1, ...]
        else:
            # 如果当前下标比self.history_length还要小，那么就要从buffer的结尾处取了。
            indexes = [(index - i )% self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]


    def sample(self):
        assert self.count > self.history_length
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.history_length, self.count + 1)    # 相当于从self.histor_length之后进行采样
                # 如果包含current，就重新采样。（current是刚生成的样本）
                if index > self.current and self.current - self.history_length <= index:
                    continue
                # 如果包含一个episode的结束状态，重新采样
                if self.terminals[(index - self.history_length):self.history_length].any():
                    continue
                break

            self.states[len(indexes),...] = self.getState(index - 1)
            self.next_states[len(indexes),...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        if self.state_format == 'NHWC':
            return np.transpose(self.states, (0, 2, 3, 1)), actions, rewards, np.transpose(self.next_states, (0, 2, 3, 1)),terminals
        else:
            return self.states, actions, rewards, self.next_states, terminals