import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table


world_size = 5
#
A_pos = (1, 0)
A_prime_pos = (1, 4)
B_pos = (3, 0)
B_prime_pos = (3, 2)

discount = 0.9

ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
prob = 0.25 


def step(state, action):
  if state == A_pos:
    return A_prime_pos, 10
  if state == B_pos:
    return B_prime_pos, 5

  next_state = (state[0]+action[0], state[1]+action[1])
  if next_state[0] < 0 or next_state[0] > world_size -1 or next_state[1] < 0 or next_state[1] > world_size -1:
    next_state = state
    reward = -1
  else:
    reward = 0
  return next_state, reward 


def draw_image(image):
  plt.figure()
  ax = plt.gca()
  ax.set_axis_off()

  table = Table(ax, bbox=[0,0,1,1]) 

  width, height = 1.0/world_size, 1.0/world_size

  for i in range(world_size):
    for j in range(world_size):
      # image, i is x, j is y
      table.add_cell(j, i, width, height, text=image[i, j], loc='center')

  ax.add_table(table)


def figure_3_2():
  values = np.zeros((world_size,world_size))
  while True:
    new_values = np.zeros((world_size, world_size))
    for i in range(world_size):
      for j in range(world_size):
        temp = 0
        for action in ACTIONS:
          next_state, reward = step((i, j), action)
          temp = temp + prob*(reward + discount * values[next_state[0], next_state[1]])
        new_values[i,j] = temp
    if np.sum(np.abs(values-new_values)) < 1e-4:
      draw_image(np.round(values, 1))
      plt.savefig("3_2.png")
      plt.close()
      break
    values = new_values


def figure_3_5():
  values = np.zeros((world_size, world_size))
  while True:
    new_values = np.zeros((world_size, world_size))
    for i in range(world_size):
      for j in range(world_size):
        max = 0
        for action in ACTIONS:
          next_state, reward = step((i, j), action)
          temp = (reward + discount * values[next_state[0], next_state[1]])
          if max < temp:
            max = temp
        new_values[i, j] = max
    if np.sum(np.abs(values - new_values)) < 1e-4:
      draw_image(np.round(values, 1))
      plt.savefig("3_5.png")
      plt.close()
      break
    values = new_values


if __name__ == "__main__":
   figure_3_2()
   figure_3_5()
