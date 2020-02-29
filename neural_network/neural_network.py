import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
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
plt.yticks(ticks = [0, 1], labels=['off', 'on'])
plt.title('$\sigma(x)$')
plt.savefig('img/fig3')

#############
# Figure 4
#############
fig, axes = plt.subplots(ncols=2, nrows=2)
plt.subplots_adjust(hspace=0.4)
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
pos = axes[0, 0].imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[0, 0].title.set_text('$\sigma(x)$')
axes[0, 1].imshow(D2, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[0, 1].title.set_text('$\sigma(x + y)$')
axes[1, 0].imshow(D3, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[1, 0].title.set_text('$\sigma(2x - y)$')
axes[1, 1].imshow(D4, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[1, 1].title.set_text('$\sigma(-x + 2y)$')
fig.colorbar(pos, ax=axes[0:2], shrink=1)
plt.savefig('img/fig4')

#############
# Figure 5
#############
fig, axes = plt.subplots(ncols=2, nrows=2)
plt.subplots_adjust(hspace=0.4)
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
pos = axes[0, 0].imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[0, 0].title.set_text('$\sigma(x + 0.4)$')
axes[0, 1].imshow(D2, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[0, 1].title.set_text('$\sigma(x + y + 0.4)$')
axes[1, 0].imshow(D3, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[1, 0].title.set_text('$\sigma(2x - y + 0.4)$')
axes[1, 1].imshow(D4, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[1, 1].title.set_text('$\sigma(-x + 2y + 0.4)$')
fig.colorbar(pos, ax=axes[0:2], shrink=1)
plt.savefig('img/fig5')

#############
# Figure 6
#############
# Two neurons

#############
# Figure 7
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
axes.plot([-1, 1], [0, 0], 'k')
axes.plot([-1, 1], [-1, 1], 'k')
plt.text(0.5, -0.5, '[1, 1]')
plt.text(0.75, 0.25, '[1, 0]')
plt.text(-0.5, 0.5, '[0, 0]')
plt.text(-0.75, -0.25, '[0, 1]')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
plt.savefig('img/fig7')

#############
# Figure 8
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
axes.plot([1], [1], 'kx')
axes.plot([0], [0], 'kx')
axes.plot([1], [0], 'kx')
axes.plot([0], [1], 'kx')
plt.text(1+0.05, 1+0.05, '[1, 1]')
plt.text(1+0.05, 0+0.05, '[1, 0]')
plt.text(0+0.05, 0+0.05, '[0, 0]')
plt.text(0+0.05, 1+0.05, '[0, 1]')
axes.set_xlim([-0.5, 1.5])
axes.set_ylim([-0.5, 1.5])
plt.savefig('img/fig8')

#############
# Figure 9
#############
# 2 layer net

#############
# Figure 10
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
axes.plot([1], [1], 'kx')
axes.plot([0], [0], 'kx')
axes.plot([1], [0], 'kx')
axes.plot([0], [1], 'kx')
plt.text(1+0.05, 1+0.05, '[1, 1]')
plt.text(1+0.05, 0+0.05, '[1, 0]')
plt.text(0+0.05, 0+0.05, '[0, 0]')
plt.text(0+0.05, 1+0.05, '[0, 1]')
plt.plot([-0.5, 1], [0.5, 1.5], 'k')
axes.set_xlim([-0.5, 1.5])
axes.set_ylim([-0.5, 1.5])
plt.savefig('img/fig10')

#############
# Figure 11
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
axes.plot([-1, 1], [0, 0], 'k')
axes.plot([-1, 1], [-1, 1], 'k')
plt.text(0.5, -0.5, '[1, 1]')
plt.text(0.75, 0.25, '[1, 0]')
plt.text(-0.5, 0.5, '[0, 0]')
plt.text(-0.75, -0.25, '[0, 1]')
axes.fill_between([-1, 0], [0, 0], [-1, 0])
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
plt.savefig('img/fig11')

#############
# Figure 12
#############
x_coord = [0.90, 0.71, 0.79, 0.50, 0.69, 0.61, 0.98, 0.76, 0.77, 0.22, 0.45, 0.57, 0.59, 0.80, 0.91, 0.99, 0.24, 0.4, 0.35]
y_coord = [0.43, 0.52, 0.63, 0.80, 0.35, 0.72, 0.04, 0.19, 0.27, 0.61, 0.85, 0.25, 0.46, 0.81, 0.70, 0.99, 0.30, 0.95, 0.48]
label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

fig, axes = plt.subplots(ncols=1, nrows=1)
for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1), aspect=1)
plt.savefig('img/fig12')

#############
# Figure 13
#############
plt.plot([0, 1], [0.98, 0.65], 'k')
plt.plot([0.64, 0.64], [0, 1], 'k')
plt.plot([0.2, 1], [1, 0.2], 'k')
plt.savefig('img/fig13')

#############
# Figure 14
#############
plt.text(0.75, 0.15, '[0, 0, 1]')
plt.text(0.45, 0.75, '[0, 1, 0]')
plt.text(0.75, 0.55, '[0, 1, 1]')

plt.text(0.75, 0.9, '[1, 1, 1]')
plt.text(0.3, 0.4, '[0, 0, 0]')
plt.text(0.4, 0.9, '[1, 1, 0]')
plt.text(0.07, 0.96, '[1, 0, 0]')
plt.savefig('img/fig14')

#############
# Figure 15
#############
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([0, 0, 0], [0, 1, 1], [1, 0, 1], c='b')
ax.scatter([1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 0], c='r')
plt.savefig('img/fig15')

#############
# Figure 16
#############
x = np.linspace(0, 1, 4)
y = np.linspace(0, 1, 4)
X, Y = np.meshgrid(x, y)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])
ax.view_init(50, -89)
plt.savefig('img/fig16')

#############
# Figure 17
#############
x_coord = [0.90, 0.71, 0.79, 0.50, 0.69, 0.61, 0.98, 0.76, 0.77, 0.22, 0.45, 0.57, 0.59, 0.80, 0.91, 0.99, 0.24, 0.4, 0.35]
y_coord = [0.43, 0.52, 0.63, 0.80, 0.35, 0.72, 0.04, 0.19, 0.27, 0.61, 0.85, 0.25, 0.46, 0.81, 0.70, 0.99, 0.30, 0.95, 0.48]
label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
fig, axes = plt.subplots(ncols=1, nrows=1)
for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1), aspect=1)
plt.plot([0, 1], [0.98, 0.65], 'k')
plt.plot([0.64, 0.64], [0, 1], 'k')
plt.plot([0.2, 1], [1, 0.2], 'k')
axes.fill_between([0.64, 1], [0.77, 0.65], [0, 0], color = 'blue', alpha=0.5)
axes.fill_between([0.33, 0.64], [0.87, 0.77], [0.87, 0.56], color = 'blue', alpha=0.5)
plt.savefig('img/fig17')

#############
# Figure 18
#############
x_coord = [0.08, 0.10, 0.90, 0.40, 0.71, 0.79, 0.50, 0.69, 0.61, 0.98, 0.76, 0.77, 0.22, 0.45, 0.57, 0.59, 0.80, 0.91, 0.99, 0.24, 0.4, 0.35]
y_coord = [0.12, 0.82, 0.43, 0.18, 0.52, 0.63, 0.80, 0.35, 0.72, 0.04, 0.19, 0.27, 0.61, 0.85, 0.25, 0.46, 0.81, 0.70, 0.99, 0.30, 0.95, 0.48]
label = [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]
fig, axes = plt.subplots(ncols=1, nrows=1)
for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1), aspect=1)
plt.plot([0, 1], [0.75, 0.9], 'k')
plt.plot([0, 1], [0.5, 0.8], 'k')
plt.plot([0, 1], [0.18, 0.5], 'k')
plt.plot([0, 1], [0.38, 0.28], 'k')
plt.plot([0.45, 0.6], [0, 1], 'k')
plt.plot([0.9, 0.4], [0, 1], 'k')
plt.savefig('img/fig18')

#############
# Figure 19
#############
def relu(x):
    return x if x>0 else 0

def leaky_relu(x):
    if x>0:
        return x
    else:
        return 0.01*x
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=1, nrows=4)
x = np.linspace(-8, 8, 1000)
ax1.plot(x, sigmoid(x), 'k', label='$\sigma(x) = \exp(x)/(1 + \exp(x))$')
ax2.plot(x, np.tanh(x), 'k', label='$\sigma(x) = tanh(x)$')
ax3.plot(x, [relu(i) for i in x], 'k', label='$\sigma(x) = \max(x, 0)$')
ax4.plot(x, [leaky_relu(i) for i in x], 'k', label='$\sigma(x) = x$ if $x > 0$, and $0.01x$ otherwise')
ax1.set(xlim=(-8, 8), ylim=(-0.1, 1.1))
ax2.set(xlim=(-8, 8), ylim=(-1.1, 1.1))
ax3.set(xlim=(-8, 8), ylim=(-0.1, 1.1))
ax4.set(xlim=(-8, 8), ylim=(-1.1, 1.1))
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.savefig('img/fig19')

#############
# Figure 20
#############
fig, axes = plt.subplots(ncols=2, nrows=2)
plt.subplots_adjust(hspace=0.4)
lx = np.linspace(-4, 4, 400)
ly = np.linspace(-4, 4, 400)
D1 = np.zeros([400, 400])
D2 = np.zeros([400, 400])
D3 = np.zeros([400, 400])
D4 = np.zeros([400, 400])
for i, x in enumerate(lx):
    for j, y in enumerate(ly):
        D1[i, j] = sigmoid(x+0.4)
        D2[i, j] = sigmoid(x + y+ 0.4)
        D3[i, j] = sigmoid(2*x - y+ 0.4)
        D4[i, j] = sigmoid(-x + 2*y+ 0.4)
pos = axes[0, 0].imshow(D1, extent = [-4, 4, -4, 4], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[0, 0].title.set_text('$\sigma(x + 0.4)$')
axes[0, 1].imshow(D2, extent = [-4, 4, -4, 4], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[0, 1].title.set_text('$\sigma(x + y + 0.4)$')
axes[1, 0].imshow(D3, extent = [-4, 4, -4, 4], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[1, 0].title.set_text('$\sigma(2x - y + 0.4)$')
axes[1, 1].imshow(D4, extent = [-4, 4, -4, 4], cmap=plt.get_cmap('Greys'), aspect = 1)
axes[1, 1].title.set_text('$\sigma(-x + 2y + 0.4)$')
fig.colorbar(pos, ax=axes[0:2], shrink=1)
plt.savefig('img/fig20')


#############
# Figure 21
#############
N = 50
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
for x in np.linspace(-4, 4, N):
    for y in np.linspace(-4, 4, N):
        ax1.plot(x, y, 'o', color=(x/8 + 0.5, y/8 +0.5, 0))
ax1.plot([-4, 4], [-4, 4], 'k')
ax1.plot([-4, 4], [2, -2], 'k')
for x in np.linspace(-4, 4, N):
    for y in np.linspace(-4, 4, N):
        ax2.plot(sigmoid(2*x - y), sigmoid(x + y), 'o', color=(x/8 + 0.5, y/8 +0.5, 0))
plt.suptitle('$[x, y] -> [\sigma(2x - y), \sigma(x + y)]$')
plt.savefig('img/fig21')
