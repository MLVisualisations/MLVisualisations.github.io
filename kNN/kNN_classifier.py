import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')

#############
# Figure 1
#############
x_coord = [0.90, 0.30, 0.71, 0.79, 0.50, 0.69, 0.61, 0.98, 0.76, 0.77, 0.22, 0.45, 0.57, 0.59, 0.80, 0.91, 0.99, 0.24, 0.4, 0.35]
y_coord = [0.43, 0.40, 0.52, 0.63, 0.80, 0.35, 0.72, 0.04, 0.19, 0.27, 0.61, 0.85, 0.25, 0.46, 0.81, 0.70, 0.99, 0.30, 0.95, 0.48]
label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

fig, axes = plt.subplots(ncols=1, nrows=1)
for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1))
plt.title('Consider two classes, blue and red.')
plt.savefig('img/fig1')

#############
# Figure 2
#############
axes.plot(0.5, 0.6, 'gx')
plt.title('Which class does the green cross belong to?')
plt.savefig('img/fig2')


#############
# Figure 3
#############
axes.annotate('', xy=(0.5,0.6), xytext=(0.50,0.80), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.61,0.72), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.59,0.46), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.45,0.85), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.35,0.48), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
plt.title('Well whats its nearest neighbour?')
plt.savefig('img/fig3')

#############
# Figure 4
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
# Run through the space
for x in np.linspace(0, 1, 100):
    for y in np.linspace(0, 1, 100):
        allocated_class = None
        min_dist = np.inf
        # Run through the training set
        for i in range(len(x_coord)):
            # Compute squared Euclidean distance (remove sqrt to save flops)
            d = (x - x_coord[i])**2 + (y - y_coord[i])**2
            # Check is this is closest seen so far
            if d < min_dist:
                min_dist = d
                allocated_class = label[i]
        # Plot
        if allocated_class == 1:
            axes.plot(x, y, 'ro', fillstyle='full', alpha=0.05)
        else:
            axes.plot(x, y, 'bo', fillstyle='full', alpha=0.05)

for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')

axes.set(xlim=(0, 1), ylim=(0, 1))
plt.title('How would this method divide the space?')
plt.savefig('img/fig4')

#############
# Figure 5
#############
axes.plot([0.2, 0.4, 0.4, 0.2, 0.2], [0.3, 0.3, 0.5, 0.5, 0.3], 'g')
plt.title('Is this an anomaly? Maybe consider more neighbours?')
plt.savefig('img/fig5')

#############
# Figure 6
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
# Run through the space
for x in np.linspace(0, 1, 100):
    for y in np.linspace(0, 1, 100):
        allocated_class = np.array([None, None, None])
        min_dist = np.array([np.inf, np.inf, np.inf])
        # Run through the training set
        for i in range(len(x_coord)):
            # Compute squared Euclidean distance (remove sqrt to save flops)
            d = (x - x_coord[i])**2 + (y - y_coord[i])**2
            # Check is this is closest seen so far
            if d < min_dist[2]:
                min_dist[2] = d
                allocated_class[2] = label[i]

                # Sort in descending order
                index = np.argsort(min_dist)
                min_dist = min_dist[index]
                allocated_class = allocated_class[index]

        # Plot
        list_mode = stats.mode(allocated_class.tolist())
        if list_mode[0] == [1]:
            axes.plot(x, y, 'ro', fillstyle='full', alpha=0.05)
        else:
            axes.plot(x, y, 'bo', fillstyle='full', alpha=0.05)

for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1))
plt.title('What if we take average of closest 3 neighbours?')
plt.savefig('img/fig6')

#############
# Figure 7
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
# Run through the space
for x in np.linspace(0, 1, 100):
    for y in np.linspace(0, 1, 100):
        allocated_class = np.array([None, None, None, None, None])
        min_dist = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        # Run through the training set
        for i in range(len(x_coord)):
            # Compute squared Euclidean distance (remove sqrt to save flops)
            d = (x - x_coord[i])**2 + (y - y_coord[i])**2
            # Check is this is closest seen so far
            if d < min_dist[4]:
                min_dist[4] = d
                allocated_class[4] = label[i]

                # Sort in descending order
                index = np.argsort(min_dist)
                min_dist = min_dist[index]
                allocated_class = allocated_class[index]

        # Plot
        list_mode = stats.mode(allocated_class.tolist())
        if list_mode[0] == [1]:
            axes.plot(x, y, 'ro', fillstyle='full', alpha=0.05)
        else:
            axes.plot(x, y, 'bo', fillstyle='full', alpha=0.05)

for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1))
plt.title('Or 5? Too many can be bad!')
plt.savefig('img/fig7')
