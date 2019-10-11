import numpy as np

# np.random.choice(
#     a,    # 采样范围
#     size=None,    # 返回采样的shape
#     replace=True, # 是否重复用
#     p=None    # a中每个值的概率
# )
a0 = np.random.choice([8, 9, -1, 2, 0], 3)
print(a0)

# 从np.arange(5)从使用均匀分布采样一个shape为4的样本
a1 = np.random.choice(5, 4)
print(a1)

a2 = np.random.choice(5, 8, p=[0.1, 0.2, 0.5, 0.2, 0])
print(a2)


a3 = np.random.choice([1, 2, 3, 8, 9], 5, replace=False)
print(a3)

a4 = np.random.choice(3, 5, p=[0.9, 0.05, 0.05])
print(a4)
