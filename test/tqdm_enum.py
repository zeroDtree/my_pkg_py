import time

from tqdm import tqdm

n = 100
a = [10 * i for i in range(n)]
for i, x in enumerate(tqdm(a)):
    time.sleep(0.1)
