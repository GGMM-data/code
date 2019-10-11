import numpy as np

size = 16
actions = np.ones(16)
mask = np.random.random(16) < 0.3

print(mask.shape)
print(mask)
print(actions)

temp = np.random.choice(3, sum(mask))
print("temp: ", temp)
print("actions[mask]: ", actions[mask])
actions[mask] = temp
print(actions)
