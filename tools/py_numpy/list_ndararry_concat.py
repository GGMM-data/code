import numpy as np
import time

length = 4
task = 2

action = []
for i in range(length):
    action.append(np.array([[1+i,2+i, 3+i], [4+i,5+i, 6+i]]))

action_array = np.array(action)
print(action_array.shape)
for a in action_array:
    print(a)
print("==================================")
print(action_array[:,0,:])
print(action_array[:,1,:])

# action_array_T = action_array.transpose()
# print(action_array_T.shape)
# for a in action_array_T:
    # print(a)



