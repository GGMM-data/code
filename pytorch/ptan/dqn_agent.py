import ptan
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, actions: int):
        super(Net, self).__init__()
        self.actions = actions

    def forward(self, x):
        return torch.eye(x.size()[0], self.actions)

net = Net(actions=3)
state = torch.zeros(10, 5)
q = net(state)
print(q)

# 1. argmax action selector
selector = ptan.actions.ArgmaxActionSelector()
agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector, device="cpu")
state = torch.zeros(10, 5)
actions, agent_state = agent(state)
print(actions, agent_state)

# 2. epsilon greedy(1.0) action selector
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector, device="cpu")
state = torch.zeros(10, 5)
actions, agent_state = agent(state)
print(actions, agent_state)

# 3. epsilon greedy(0.5) action selector
selector.epsilon = 0.5
state = torch.zeros(10, 5)
actions, agent_state = agent(state)
print(actions, agent_state)

# 4. epsilon greedy(0.1) action selector
selector.epsilon = 0.1
state = torch.zeros(10, 5)
actions, agent_state = agent(state)
print(actions, agent_state)

