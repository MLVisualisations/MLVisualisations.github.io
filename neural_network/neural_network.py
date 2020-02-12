import numpy as np
np.random.seed(0)
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
# Figure 7
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
r = 0.5*np.random.rand(200)
theta = 2*np.pi*np.random.rand(200)
axes.plot(r*np.cos(theta), r*np.sin(theta), 'gx')
lx = np.linspace(-1, 1, 100)
ly = np.linspace(-1, 1, 100)
D1 = np.zeros([100, 100])
for i, x in enumerate(lx):
    for j, y in enumerate(ly):
        if step(x + 0.5):
            D1[i, j] += 1
        if step(-x + 0.5):
            D1[i, j] += 1
        if step(y + 0.5):
            D1[i, j] += 1
        if step(-y + 0.5):
            D1[i, j] += 1
axes.imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
plt.savefig('img/fig7')

#############
# Figure 8
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
r = 0.5*np.random.rand(200)
theta = 2*np.pi*np.random.rand(200)
axes.plot(r*np.cos(theta), r*np.sin(theta), 'gx')
lx = np.linspace(-1, 1, 100)
ly = np.linspace(-1, 1, 100)
D1 = np.zeros([100, 100])
for i, x in enumerate(lx):
    for j, y in enumerate(ly):
        if step(x + 0.5):
            D1[i, j] += 1
        if step(-x + 0.5):
            D1[i, j] += 1
        if step(y + 0.5):
            D1[i, j] += 1
        if step(-y + 0.5):
            D1[i, j] += 1
        if step(x + y + 0.71):
            D1[i, j] += 1
        if step(-x - y + 0.71):
            D1[i, j] += 1
        if step(-x + y + 0.71):
            D1[i, j] += 1
        if step(x - y + 0.71):
            D1[i, j] += 1
axes.imshow(D1, extent = [-1, 1, -1, 1], cmap=plt.get_cmap('Greys'), aspect = 1)
plt.savefig('img/fig8')

#############
# Figure 9
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
lx = np.linspace(-0.5, 0.5, 11)
ly = np.linspace(-0.5, 0.5, 11)
colors = ['r', 'b', 'g', 'y']
for x in lx:
    for y in ly:
        axes.plot(x, y, 'x', color=colors[np.random.randint(4)])

for x in lx:
    axes.plot([x-0.05, x-0.05], [-1, 1], 'k')
axes.plot([0.55, 0.55], [-1, 1], 'k')
for y in ly:
    axes.plot([-1, 1], [y-0.05, y-0.05], 'k')
axes.plot([-1, 1], [0.55, 0.55], 'k')
axes.set(xlim=(-1, 1), ylim=(-1, 1))
plt.savefig('img/fig9')

#############
# Figure 10
#############
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
x = np.linspace(-8, 8, 1000)
ax1.plot(x, sigmoid(x), 'k', label='$\sigma(x) = \exp(x)/(1 + \exp(x))$')
ax2.plot(x, np.tanh(x), 'k', label='$\sigma(x) = tanh(x)$')
ax1.set(xlim=(-8, 8), ylim=(-0.1, 1.1))
ax2.set(xlim=(-8, 8), ylim=(-1.1, 1.1))
ax1.legend()
ax2.legend()
plt.savefig('img/fig10')
