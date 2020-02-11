import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')


def sigmoid(x):
    expx = np.exp(x)
    return expx/(expx+1)

#############
# Figure 1
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
axes.plot([-2, 0, 0, 2], [0, 0, 1, 1], 'k')
plt.yticks(ticks = [0, 1], labels=['on', 'off'])
plt.suptitle('Neurons use switches')
plt.savefig('img/fig1')

#############
# Figure 2
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
plt.savefig('img/fig2')

#############
# Figure 3
#############
