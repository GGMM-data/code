import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAX_CARS_NUMBER = 20
MAX_MOVE_CARS_NUMBER = 5
UPPER_BOUND = 11
CREDIT = 10
GAMMA = 0.9
EXPECTED_REQUEST_A = 3
EXPECTED_RETURN_A = 3
EXPECTED_REQUEST_B = 4
EXPECTED_RETURN_B = 2
COST = 2


poisson_dict = dict()
def poisson(x, lamda):
  key = x*10 + lamda
  if key not in poisson_dict.keys():
     poisson_dict[key] = np.power(lamda,x)*np.exp(-lamda)/np.math.factorial(x) 
  return poisson_dict[key]


def expected_return(state, action, states_values, constant_returns=True):
  # state->action->reward
  # move the car
  returns = 0

  # the employ will move one car from place A to place B 
  if action < 0: 
    returns -= np.absolute(action) * COST
  else:
    returns -= np.absolute(action - 1) * COST 
  
  # if state > 10, will need another 4 cost to place the car
  # car in place A more than 10
  if state[0] > 10:
    returns -= 4
  # car in place B more than 10
  if state[1] > 10:
    returns -= 4
   
  # in the next day, the car can be rental and return 
  # request
  for request_a in range(UPPER_BOUND):
    for request_b in range(UPPER_BOUND):
      # after action, the next states
      cars_a = int(min(state[0] - action, MAX_CARS_NUMBER))
      cars_b = int(min(state[1] + action, MAX_CARS_NUMBER))
      # the number of cars may below than request
      real_request_a = min(cars_a, request_a)
      real_request_b = min(cars_b, request_b)
 
      # after rental, how many cars still left
      cars_a -= real_request_a
      cars_b -= real_request_b 
      
      reward = (real_request_a + real_request_b)*CREDIT
      
      request_prob = poisson(request_a, EXPECTED_REQUEST_A)*poisson(request_b, EXPECTED_REQUEST_B) 

      if constant_returns:
        real_return_a = EXPECTED_RETURN_A
        real_return_b = EXPECTED_RETURN_B
        
        # can't beyond the MAX_CARS_NUMBER
        next_cars_a = min(cars_a + real_return_a, MAX_CARS_NUMBER) 
        next_cars_b = min(cars_b + real_return_b, MAX_CARS_NUMBER) 
        returns += request_prob*(reward + GAMMA * states_values[next_cars_a, next_cars_b]) 
      else:
        for return_a in range(UPPER_BOUND):
          for return_b in range(UPPER_BOUND):
            next_cars_a = min(cars_a + return_a, MAX_CARS_NUMBER)
            next_cars_b = min(cars_b + return_b, MAX_CARS_NUMBER) 
            return_prob = poisson(return_a, EXPECTED_RETURN_A) * poisson(return_b, EXPECTED_RETURN_B)
            prob = request_prob * return_prob 
            returns += prob*(reward + GAMMA * states_values[next_cars_a, next_cars_b])
            
  return returns

def problem_4_7():
  states_values = np.zeros((MAX_CARS_NUMBER + 1, MAX_CARS_NUMBER + 1))
  policy = np.zeros((MAX_CARS_NUMBER + 1, MAX_CARS_NUMBER + 1))
  states = []
  actions = np.arange(- MAX_MOVE_CARS_NUMBER, MAX_MOVE_CARS_NUMBER + 1)
  for i in range(MAX_CARS_NUMBER + 1):
    for j in range(MAX_CARS_NUMBER + 1):
      states.append([i,j])
  
  figure, axes = plt.subplots(2, 3, figsize=[40,20])
  axes = axes.flatten()
  iteration = 0 

  # policy iteration
  new_states_values = states_values.copy() 
  while True:
    fig = sns.heatmap(np.flipud(states_values), cmap="YlGnBu", ax=axes[iteration])
    fig.set_xlabel("first place", fontsize=30)
    fig.set_ylabel("second place", fontsize=30)
    fig.set_title("policy %d" % iteration)
    fig.set_yticks(list(reversed(range(MAX_CARS_NUMBER+1))))

    # policy evaluation
    while True:
      for i in range(MAX_CARS_NUMBER + 1):
        for j in range(MAX_CARS_NUMBER + 1):
          new_states_values[i,j] = expected_return(state=[i,j], action=policy[i,j], states_values=new_states_values, constant_returns=True)
           
      error = np.abs((new_states_values - states_values)).sum()
      print(error)
      if error < 1e-4:
        break
      states_values = new_states_values.copy()
     
    # policy improvement
    count = 0
    for i in range(MAX_CARS_NUMBER + 1):
      for j in range(MAX_CARS_NUMBER + 1):
        max_returns = 0
        a = 0 
        # find max reward(greedy)
        for action in actions:
          # cut leaves
          if (action > 0 and action > i) or (action < 0 and np.absolute(action) > j):
            continue
          returns = expected_return(state=[i, j], action=action, states_values=states_values, constant_returns=True) 
          if returns > max_returns:
            max_returns = returns
            a = action

        # count how many states' policy change
        if a != policy[i,j]:
          policy[i,j] = a
          count += 1

    print("%d policy states changed" % count)
    iteration += 1

    # no policy change, break the policy iteration
    if count == 0 :
      fig = sns.heatmap(np.flipud(states_values), cmap="YlGnBu", ax=axes[iteration])
      fig.set_xlabel("first place", fontsize=30)
      fig.set_ylabel("second place", fontsize=30)
      fig.set_title("optimal policy")
      fig.set_yticks(list(reversed(range(MAX_CARS_NUMBER+1))))
      
      plt.savefig("problem_4_7.png")
      plt.show()
      plt.close()
      break

if __name__ == "__main__":
  problem_4_7()
