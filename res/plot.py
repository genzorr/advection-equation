import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

M = 1000
K = 1000
xMin, xMax = 0, 10
tMax = 3

h = (xMax-xMin)/(M-1)
tau = tMax/(K-1)

data = []
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
x = np.arange(xMin, xMax+h, h)
line, = ax.plot(x, x*0, '-')

def update(i):
    line.set_data(x, data[i])
    return line,

with open('out.dat', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        del row[-1]
        row = list(map(float, row))
        data.append(row)

min_value = min(map(min, data))
max_value = max(map(max, data))
ax.set_ylim([min_value, max_value])
anim = animation.FuncAnimation(fig, update, interval=1, frames=K, blit=True)

anim.save('animation.mp4', writer=animation.FFMpegWriter(fps=60*tMax))
plt.show()
