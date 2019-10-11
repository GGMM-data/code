import ptan
import gym
import numpy as np
from typing import List, Any, Optional, Tuple

import matplotlib.pylab as plt


q_values = np.array([[1, 2, 3], [1, -1, 0]])
print(q_values)

selector = ptan.actions.ArgmaxActionSelector()
a1 = selector(q_values)
print("argmax selector: \n", a1)

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0)
a2 = selector(q_values)
print("epsilon(0.0) greedy selector: \n", a2)

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
a3 = selector(q_values)
print("epsilon(1.0) greedy selector: \n", a3)

selector = ptan.actions.ProbabilityActionSelector()
prob = np.array([[0.3, 0.7], [0.4, 0.6]])
a4 = selector(prob)
print("probability selector: \n", a4)

