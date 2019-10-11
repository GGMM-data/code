import ptan
import numpy as np
import torch
import torch.nn as nn


# dqn输出的是q值。
# 输出discrete actions的概率分布，可以是logis也可以是normalized distribution。
class Net(nn.Module):
    def __init__(self, actions: int):
        super(Net, self).__init__()
        self.actions = actions

    def forward(self, x):
        res = torch.zeros(x.size()[0], self.actions)
        # res is logits，即unnormalized probability
        res[:, 0] = 1
        res[:, 1] = 1
        return res

net = Net(actions=5)
state = torch.zeros(10, 5)
logits = net(state)
print(logits)

# 1. probability action selector
selector = ptan.actions.ProbabilityActionSelector()
agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
state = torch.zeros(10, 5)
actions, agent_state = agent(state)
print(actions, agent_state)


