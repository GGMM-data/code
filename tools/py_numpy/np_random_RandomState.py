import numpy as np

rdm = np.random.RandomState()

for _ in range(5):
    rdm.seed(4)
    print(rdm.rand())
    print(rdm.rand())
    print(rdm.rand())
    print("\n")
