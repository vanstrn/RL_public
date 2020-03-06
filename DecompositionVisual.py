import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors

dimension = 7
x = np.zeros((dimension**2,dimension**2))

TerminalLocations = [[0,6],[6,0],[6,6]]
Rewards = [-1,1,2]



skips = []
w_i = []
w = np.zeros((dimension**2))
for i,loc in enumerate(TerminalLocations):
    skips.append(loc[0]+dimension*loc[1])
    w[loc[0]+dimension*loc[1]] = Rewards[i]

    tmp = np.zeros((dimension**2))
    tmp[loc[0]+dimension*loc[1]] = Rewards[i]
    w_i.append(tmp)

for i in range(dimension**2):
    if i in skips:
        continue
    sum = 0
    if i % dimension != dimension-1:
        x[i,i+1] = 0.25
        sum += 0.25
    if i % dimension != 0:
        x[i,i-1] = 0.25
        sum += 0.25
    if i >= dimension:
        x[i,i-dimension] = 0.25
        sum += 0.25
    if i <= dimension**2-dimension-1:
        x[i,i+dimension] = 0.25
        sum += 0.25
    x[i,i]= 1.0-sum

I = np.identity(dimension**2)
psi = np.linalg.inv(I-x)


images = []
fig,axs = plt.subplots(2, 2)

value = np.reshape(np.matmul(psi,w),(dimension,dimension))
images.append(axs[0,0].imshow(value))
axs[0,0].set_title("Global Value Estimate")
axs[0,0].label_outer()
plt.show()

for i,w_ in enumerate(w_i):
    value_i = np.reshape(np.matmul(psi,w_),(dimension,dimension))
    images.append(axs[(i+1)//2,(i+1)%2].imshow(value_i))
    axs[(i+1)//2,(i+1)%2].set_title("Value Estimate from " + str(i))
    axs[(i+1)//2,(i+1)%2].label_outer()

vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
plt.show()
