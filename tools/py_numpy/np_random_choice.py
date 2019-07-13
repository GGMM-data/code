import numpy as np

a0 = np.random.choice([8, 9, -1, 2, 0], 3)
print(a0)

# 从np.arange(5)从使用均匀分布采样一个shape为4的样本
a1 = np.random.choice(5, 4)
print(a1)

a2 = np.random.choice(5, 8, p=[0.1, 0.2, 0.5, 0.2, 0])
print(a2)


a3 = np.random.choice([1, 2, 3, 8, 9], 5, replace=False)
print(a3)
