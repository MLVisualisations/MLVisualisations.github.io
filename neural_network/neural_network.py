import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('fast')


def sigmoid(x):
    expx = np.exp(x)
    return expx/(expx+1)

def step(x):
    return 1 if x>=0 else 0

#############
# Figure 1
#############
# Neuron

#############
# Figure 2
#############
# Inside neuron

#############
# Figure 3
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
axes.plot([-2, 0, 0, 2], [0, 0, 1, 1], 'k')
plt.yticks(ticks = [0, 1], labels=['on', 'off'])
plt.title('$\sigma(x)$')
plt.savefig('img/fig3')

#############
# Figure 4
#############
fig, axes = plt.subplots(ncols=2, nrows=2)
plt.subplots_adjust(hspace=0.4)
(ax1, ax2, ax3, ax4) = axes.ravel()
lx = np.linspace(-1, 1, 100)
ly = np.linspace(-1, 1, 100)
D1 = np.zeros([100, 100])
D2 = np.zeros([100, 100])
D3 = np.zeros([100, 100])
D4 = np.zeros([100, 100])
for i, x in enumerate(lx):
    for j, y in enumerate(ly):
        if step(x):
            D1[i, j] = 1
        if step(x + y):
            D2[i, j] = 1
        if step(2*x - y):
            D3[i, j] = 1
        if step(-x + 2*y):
            D4[i, j] = 1
ax1.imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax1.title.set_text('$\sigma(x)$')
ax2.imshow(D2, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax2.title.set_text('$\sigma(x + y)$')
ax3.imshow(D3, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax3.title.set_text('$\sigma(2x - y)$')
ax4.imshow(D4, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax4.title.set_text('$\sigma(-x + 2y)$')
plt.savefig('img/fig4')

#############
# Figure 5
#############
fig, axes = plt.subplots(ncols=2, nrows=2)
plt.subplots_adjust(hspace=0.4)
(ax1, ax2, ax3, ax4) = axes.ravel()
lx = np.linspace(-1, 1, 100)
ly = np.linspace(-1, 1, 100)
D1 = np.zeros([100, 100])
D2 = np.zeros([100, 100])
D3 = np.zeros([100, 100])
D4 = np.zeros([100, 100])
for i, x in enumerate(lx):
    for j, y in enumerate(ly):
        if step(x + 0.4):
            D1[i, j] = 1
        if step(x + y+ 0.4):
            D2[i, j] = 1
        if step(2*x - y+ 0.4):
            D3[i, j] = 1
        if step(-x + 2*y+ 0.4):
            D4[i, j] = 1
ax1.imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax1.title.set_text('$\sigma(x + 0.4)$')
ax2.imshow(D2, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax2.title.set_text('$\sigma(x + y + 0.4)$')
ax3.imshow(D3, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax3.title.set_text('$\sigma(2x - y + 0.4)$')
ax4.imshow(D4, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax4.title.set_text('$\sigma(-x + 2y + 0.4)$')
plt.savefig('img/fig5')

#############
# Figure 6
#############
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
plt.subplots_adjust(wspace=0.4)
lx = np.linspace(-1, 1, 100)
ly = np.linspace(-1, 1, 100)
D1 = np.zeros([100, 100])
D2 = np.zeros([100, 100])
for i, x in enumerate(lx):
    for j, y in enumerate(ly):
        if step(x):
            D1[i, j] += 1
        if step(x + y):
            D1[i, j] += 1

        if step(-x + 2*y + 0.4):
            D2[i, j] += 1
        if step(x + y + 0.4):
            D2[i, j] += 1
        if step(x):
            D2[i, j] += 1

ax1.imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax1.title.set_text('$\sigma(x) + \sigma(x+y)$')
ax2.imshow(D2, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
ax2.title.set_text('$\sigma(-x + 2y + 0.4) + \sigma(x + y + 0.4) + \sigma(x)$')
plt.savefig('img/fig6')

#############
# Figure 8
#############
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
x = np.linspace(-8, 8, 1000)
ax1.plot(x, sigmoid(x), 'k', label='$\sigma(x) = \exp(x)/(1 + \exp(x))$')
ax2.plot(x, np.tanh(x), 'k', label='$\sigma(x) = tanh(x)$')
ax1.set(xlim=(-8, 8), ylim=(-0.1, 1.1))
ax2.set(xlim=(-8, 8), ylim=(-1.1, 1.1))
ax1.legend()
ax2.legend()
plt.suptitle('Which we approximate by a continuous function $\sigma$, either:')
plt.savefig('img/fig8')
