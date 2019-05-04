import numpy as np
from math import exp, factorial
import seaborn as sns
import matplotlib.pyplot as plt


#### error :
# 1. in-policy policy iteration
# 2. float use %f???


# state, action, reward, probabilitied

# reward for rental a car
REWARD = 10

# cost for move a car
COST = 2

# number of car one place can hold
MAX_NUMBER_CARS = 20

# number of car can be moved 
MAX_MOVE_CAR_NUMBER = 5

# requset and request number of car in place A and B
EXPECTED_REQUEST_NUMBER_CARS_A = 3
EXPECTED_RETURN_NUMBER_CARS_A = 3
EXPECTED_REQUEST_NUMBER_CARS_B = 4
EXPECTED_RETURN_NUMBER_CARS_B = 2

# discount factor
GAMMA = 0.9

# upper bound of possion distribution
UPPER_BOUND = 11

# possion dict
pBackup = dict()

# calculate possion distribution
def possion(x, lamda):
  global pBackup
  # here the key depend on the lamda and x
  key = x * 10 + lamda 
  if key not in pBackup.keys():
    pBackup[key] = exp(-lamda)*np.power(lamda, x)/factorial(x)
  return pBackup[key]

# calculate the expected return 
def expected_return(state, action, states_value, constant_return_cars=False):
  returns = 0.0
   
  returns -= COST * np.absolute(action)
  
  # request for place a,b 
  for request_number_of_cars_a in range(UPPER_BOUND):
    for request_number_of_cars_b in range(UPPER_BOUND):
      # these two lines must be placed here, since it will change during every loop 
      # next state after action
      number_of_cars_a = int(min(state[0] - action, MAX_NUMBER_CARS))
      number_of_cars_b = int(min(state[1] + action, MAX_NUMBER_CARS))

      # the real number user can request
      real_request_number_of_cars_a = min(number_of_cars_a, request_number_of_cars_a)
      real_request_number_of_cars_b = min(number_of_cars_b, request_number_of_cars_b) 
      
      # reward
      reward = (real_request_number_of_cars_a + real_request_number_of_cars_b) * REWARD

      # probability of current (request_a, request_b)
      prob_request = possion(request_number_of_cars_a, EXPECTED_REQUEST_NUMBER_CARS_A) * possion(request_number_of_cars_b, EXPECTED_REQUEST_NUMBER_CARS_B)

      number_of_cars_a -= real_request_number_of_cars_a
      number_of_cars_b -= real_request_number_of_cars_b
      if constant_return_cars:
        return_number_of_cars_a = EXPECTED_RETURN_NUMBER_CARS_A
        return_number_of_cars_b = EXPECTED_RETURN_NUMBER_CARS_B
        next_number_of_cars_a = min(number_of_cars_a + return_number_of_cars_a, MAX_NUMBER_CARS) 
        next_number_of_cars_b = min(number_of_cars_b + return_number_of_cars_b, MAX_NUMBER_CARS) 
        myreturn = states_value[next_number_of_cars_a, next_number_of_cars_b]
        returns += prob_request * (reward + GAMMA * myreturn)
      else:
        # return for place a, b
        for return_number_of_cars_a in range(UPPER_BOUND):
          for return_number_of_cars_b in range(UPPER_BOUND):
            # next state
            next_number_of_cars_a = min(number_of_cars_a + return_number_of_cars_a, MAX_NUMBER_CARS)
            next_number_of_cars_b = min(number_of_cars_b + return_number_of_cars_b, MAX_NUMBER_CARS) 
            prob_return = possion(return_number_of_cars_a, EXPECTED_RETURN_NUMBER_CARS_A) * possion(return_number_of_cars_b, EXPECTED_RETURN_NUMBER_CARS_B)
            prob = prob_request * prob_return
            myreturn = states_value[next_number_of_cars_a, next_number_of_cars_b]
            returns += prob * (reward + GAMMA * myreturn)

  return returns 

def figure4_2(constant_return_cars=True):
  # state, policy, value, actions
  states_value = np.zeros((1 + MAX_NUMBER_CARS, 1 + MAX_NUMBER_CARS))
  policy = np.zeros((1 + MAX_NUMBER_CARS, 1 + MAX_NUMBER_CARS))
  states = [] 
  for i in range(MAX_NUMBER_CARS+1):
    for j in range(MAX_NUMBER_CARS+1):
      states.append([i,j])
  actions = np.arange(- MAX_MOVE_CAR_NUMBER, MAX_MOVE_CAR_NUMBER + 1)

  # policy iteration
  new_states_value = states_value.copy()
  policy_improve = True
  iteration = 0
  figure, axes = plt.subplots(2, 3, figsize=[40,20])
  axes = axes.flatten()
  while policy_improve:
    fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iteration])
    fig.set_xlabel("second place", fontsize=30)
    fig.set_xlabel("first place", fontsize=30)
    fig.set_yticks(list(reversed(range(MAX_NUMBER_CARS+1))))
    fig.set_title("policy %d" % iteration, fontsize=30)

    # policy evaluation
    while True:
      for i in range(1 + MAX_NUMBER_CARS):
        for j in range(1 + MAX_NUMBER_CARS):
          # here use new_state_values, in-place
          new_states_value[i][j] = expected_return(state=[i,j], action=policy[i][j], states_value=new_states_value, constant_return_cars=constant_return_cars)
      error = np.abs((new_states_value -states_value)).sum()
      # !!!!why must add the blow line
      #print('value change %f' % error)
      print(error)
      if error < 1e-4:
        break
      # pay attention to here, can't use = , must use copy()
      states_value = new_states_value.copy()

    print(states_value)
    # policy improvement
    count = 0
    policy_improve = False
    for i in range(1 + MAX_NUMBER_CARS):
      for j in range(1 + MAX_NUMBER_CARS):
        max_reward = 0
        a = 0
        for action in actions:
          if (action > 0 and action > i) or (action < 0 and np.math.fabs(action)>j):
            continue
          reward = expected_return(state=[i,j], action=action, states_value=states_value, constant_return_cars=True) 
          if reward > max_reward:
            max_reward = reward
            a = action
        if a != policy[i,j]:
          count +=1
          policy[i,j] = a
          policy_improve = True
    print("policy state change", count)
    
    iteration += 1

    if count == 0:
      # np.flipud(matrix)  flip each column in the up/down direction, rows are preserved
      fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iteration])
      fig.set_xlabel("first location", fontsize=30)
      fig.set_ylabel("second loation", fontsize=30)
      fig.set_title("optimal policy", fontsize=30)
      fig.set_yticks(list(reversed(range(MAX_NUMBER_CARS))))
      
      plt.savefig("figure_4_2.png")
      plt.show()
      plt.close()
    

if __name__ == "__main__":
  figure4_2()

