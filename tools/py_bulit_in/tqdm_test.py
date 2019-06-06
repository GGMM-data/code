import numpy as np
from tqdm import tqdm
import time


def main():
    for i in tqdm(range(100)):
        time.sleep(5)
        print(i)

if __name__ == '__main__':
    main()
