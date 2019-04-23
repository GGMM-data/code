import numpy as np
import matplotlib.pyplot as plt

MAX_CAPTIAL = 100
PROBABILITY_HEAD = 0.25
PROBABILITY_TAIL = 1 - PROBABILITY_HEAD
WIN_REWARD = 1
LOSE_REWARD = 0
TRANISITION_REWARD = 0
BOUND = 50

def expected_return(state, action, states_values):
  # terminal state
  if state == 100:
    return WIN_REWARD 
  if state == 0 :
    return LOSE_REWARD

  # returns contains two part, immediately reward, next state reward 
  returns = TRANISITION_REWARD + PROBABILITY_HEAD * states_values[state + action] + PROBABILITY_TAIL*states_values[state - action] 
  return returns


def figure_4_3():
  states_values = np.zeros([MAX_CAPTIAL + 1])
  policy = np.zeros([MAX_CAPTIAL + 1])
  states = np.arange(MAX_CAPTIAL + 1)

  new_states_values = states_values.copy()
  # value iteration 
  while True: 
    for state in states:
      actions = np.arange(min(state, MAX_CAPTIAL - state) + 1)
      
      max_returns = states_values[state]
      # find max state value during all actions
      for action in actions:
        returns = expected_return(state, action, new_states_values) 
        if returns > max_returns:
          max_returns = returns
      # update value function 
      new_states_values[state] = max_returns
     
    error = np.abs(new_states_values - states_values).sum()
    if error < 1e-9:
      break
    states_values = new_states_values.copy()
    # print(states_values)
 
  # output optimal policy
  for state in states[1:MAX_CAPTIAL]:
    actions = np.arange(min(state, MAX_CAPTIAL - state) + 1)
    a = policy[state]
    max_returns = 0
    action_returns = []
    for action in actions:
      returns = expected_return(state, action, states_values)
      action_returns.append(returns)
      if returns > max_returns:
        max_returns = returns
        a = action
    policy[state] = a
    policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
  
  figure,axes = plt.subplots(2,1,figsize=[10,20])
  axes = axes.flatten()
  axes[0].plot(states, states_values)
  axes[1].scatter(states, policy)
  
  plt.savefig("problem_4.9(0.25).pdf")
  plt.show()
  plt.close()


   
if __name__ == "__main__":
  figure_4_3() 
