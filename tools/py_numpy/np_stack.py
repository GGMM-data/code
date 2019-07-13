import numpy as np

state = [np.ones([1, 32]), np.ones([1, 32]) , np.ones([1, 32]), np.ones([1, 32])]

print(np.vstack(state).shape)

action = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9]
reward = [3, 4, 5, 6, 7, 8, 9, 0, 0, 9]

print(np.vstack(action).shape)
print(np.array(reward).shape)
