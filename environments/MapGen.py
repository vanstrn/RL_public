import os
import numpy as np

mapPath="/home/capturetheflag/RL/environments/fair_map"
mapPathNew="/home/capturetheflag/RL/environments/fair_map_1v0"
map_list = [os.path.join(mapPath, path) for path in os.listdir(mapPath)]
for path in os.listdir(mapPath):
    with open(os.path.join(mapPath, path)) as textFile:
        lines = np.asarray([line.split() for line in textFile],dtype=int)
    lines = np.where(lines==4, 1, lines)
    locs = np.where(lines==2)
    for i in range(len(locs[0])):
        if i == 0:
            continue
        lines[locs[0][i]][locs[1][i]]=0
    np.savetxt(os.path.join(mapPathNew, path),lines,fmt="%i")
