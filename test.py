import numpy as np

from mathelp import *

inp = np.random.randint(2, size=[5, 5, 2])
inp2 = np.random.randint(2, size=[5, 4, 2])
ker = np.random.randint(3, size=[2, 2, 2])

def padwithtens(vector, pad_width, iaxis, kwargs):
    if iaxis == 0:
        vector[:pad_width[0]] += 0
        vector[-pad_width[1]:] += 1
    if iaxis == 1:
        vector[:pad_width[0]] += 20
        vector[-pad_width[1]:] += 21
    if iaxis > 1:
        vector[:pad_width[0]] += 100
        vector[-pad_width[1]:] += 101
    return vector

test2d = np.arange(6).reshape((2,3))



if __name__ == "__main__":
    print3dmat(inp2)
