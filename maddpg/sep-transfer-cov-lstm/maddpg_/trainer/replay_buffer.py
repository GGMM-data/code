import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, history_length=1):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.history_length = history_length

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            temp_obs, temp_next_obs = [], []
            for j in range(i - self.history_length + 1, i):
                data = self._storage[j]
                obs_t, action, reward, obs_tp1, done = data
                temp_obs.append(np.array(obs_t, copy=False))
                temp_next_obs.append(np.array(obs_tp1, copy=False))
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            temp_obs.append(np.array(obs_t, copy=False))
            temp_next_obs.append(np.array(obs_tp1, copy=False))
            
            obses_t.append(temp_obs)
            obses_tp1.append(temp_next_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        indexes = []
        while len(indexes) < batch_size:
            while True:
                index = random.randint(self.history_length - 1, len(self._storage) - 1)
                # sample one not wraps current pointer
                if index - self.history_length < self._next_idx <= index:
                    continue
                # todo change code to fit this line
                if index > self._next_idx:
                    continue
                if (np.array(self._storage[index - self.history_length:index])[:, 4]).any():
                    continue
                break
            indexes.append(index)
        # indexes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return indexes

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
