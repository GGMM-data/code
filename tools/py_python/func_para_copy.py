import numpy as np

a = np.ones([2])


def add(para):
    para += 1
    
def test():
    b = np.copy(a)
    add(b)
    print(b)

if __name__ == "__main__":
    test()
    
