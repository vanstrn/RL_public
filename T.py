import gym
import gym_cap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


x = np.matrix([
    [1,2,3,4,5,6,7,8],
    [1,2,3,4,5,6,7,8],
    [2,3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2,3],
    ])
noise = np.random.rand(8,8)

res = x +noise*0

w,v = np.linalg.eig(res)
print(v)
