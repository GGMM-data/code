import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strenth
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# action
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# 
EPSILON = 0.1

# 
ALPHA = 0.5

REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]


def episode():

def sarsa():
  value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
  episodes = 500
   
  steps = []
  for i in range(episodes):
    steps.append(episode(q))
    
   
  
if __name__ == "__main__":
  sarsa()
