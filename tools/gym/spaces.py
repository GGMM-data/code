import gym
from gym import spaces
import numpy as np

env = gym.make('CartPole-v0')
action_space = env.action_space
print(action_space)
print(action_space.n)
observation_space = env.observation_space
print(observation_space)
env.reset()

## 0.抽象类
# space = spaces.Space()

### 1.Discrete
# 取值是{0, 1, ..., n - 1}
print("==================")
dis = spaces.Discrete(8)
print(dis.shape)
print(dis.n)
print(dis)
dis.seed(4)
for _ in range(5):
    print(dis.sample())

### 2.Box
#
print("==================")
# def __init__(self, low=None, high=None, shape=None, dtype=None):
"""
Two kinds of valid input:
    Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
    Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
"""
box = spaces.Box(low=3.0, high=4, shape=(2,2))
print(box) 
box.seed(4)
for _ in range(2):
    print(box.sample())

### 3.multi binary
print("==================")
mb = spaces.MultiBinary(5)
print(mb)
print(mb.shape)
mb.seed(4)
for _ in range(2):
    print(mb.sample())

### 4.multi discrete
# 取值是多个{0, 1, ..., n - 1}
print("==================")
md = spaces.MultiDiscrete([3, 4]) # 指定每个discrete的取值范围
print(md)
print(md.shape)
md.seed(4)
for _ in range(10):
    print(md.sample())

### 5.
# self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
tup = spaces.Tuple([spaces.Discrete(3), spaces.MultiDiscrete([3.0, 2])])
print(tup)
print(tup.shape)
tup.seed(4)
for _ in range(3):
    print(tup.sample())

