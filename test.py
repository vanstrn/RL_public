import numpy as np
import random

size = 30
occupancy = np.zeros([size,size])
for i in range(1000):
    state = [5,5]
    for j in range(150):
        action = random.randint(0,4)
        if action==0:
            pass
        elif action == 1:
            if state[0] == size-1:
                state=state
            else:
                state[0]+=1
        elif action==2:
            if state[1] == size-1:
                state=state
            else:
                state[1]+=1
        elif action == 3:
            if state[0] == 0:
                state=state
            else:
                state[0]-=1
        elif action==4:
            if state[1] == 0:
                state=state
            else:
                state[1]-=1


        occupancy[state[0],state[1]] += 1

print(occupancy)
