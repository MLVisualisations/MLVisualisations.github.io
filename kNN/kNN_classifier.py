import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('fast')

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
axes.set(xlim=(0, 1), ylim=(0, 1), aspect=1)
plt.savefig('img/fig1')

#############
# Figure 2
#############
axes.plot(0.5, 0.6, 'gx')
plt.savefig('img/fig2')


#############
# Figure 3
#############
axes.annotate('', xy=(0.5,0.6), xytext=(0.50,0.80), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.61,0.72), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.59,0.46), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.45,0.85), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
axes.annotate('', xy=(0.5,0.6), xytext=(0.35,0.48), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
plt.savefig('img/fig3')

#############
# Figure 4
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
D = np.zeros([100, 100])
# Run through the space
for i, x in enumerate(np.linspace(0, 1, 100)):
    for j, y in enumerate(np.linspace(1, 0, 100)):
        allocated_class = None
        min_dist = np.inf
        # Run through the training set
        for k in range(len(x_coord)):
            # Compute squared Euclidean distance (remove sqrt to save flops)
            d = (x - x_coord[k])**2 + (y - y_coord[k])**2
            # Check is this is closest seen so far
            if d < min_dist:
                min_dist = d
                allocated_class = label[k]
        # Plot
        if allocated_class == 1:
            D[j, i] = 1

for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')

axes.imshow(D, extent = [0, 1, 0, 1], cmap=plt.get_cmap('Pastel1'), aspect = 1)
plt.savefig('img/fig4')

#############
# Figure 5
#############
axes.plot([0.2, 0.4, 0.4, 0.2, 0.2], [0.3, 0.3, 0.5, 0.5, 0.3], 'g')
plt.savefig('img/fig5')

#############
# Figure 6
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
D = np.zeros([100, 100])
# Run through the space
for i, x in enumerate(np.linspace(0, 1, 100)):
    for j, y in enumerate(np.linspace(1, 0, 100)):
        allocated_class = np.array([None, None, None])
        min_dist = np.array([np.inf, np.inf, np.inf])
        # Run through the training set
        for k in range(len(x_coord)):
            # Compute squared Euclidean distance (remove sqrt to save flops)
            d = (x - x_coord[k])**2 + (y - y_coord[k])**2
            # Check is this is closest seen so far
            if d < min_dist[2]:
                min_dist[2] = d
                allocated_class[2] = label[k]

                # Sort in descending order
                index = np.argsort(min_dist)
                min_dist = min_dist[index]
                allocated_class = allocated_class[index]

        # Plot
        list_mode = stats.mode(allocated_class.tolist())
        if list_mode[0] == [1]:
            D[j, i] = 1

for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.imshow(D, extent = [0, 1, 0, 1], cmap=plt.get_cmap('Pastel1'), aspect = 1)
plt.savefig('img/fig6')

#############
# Figure 7
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
D = np.zeros([100, 100])
# Run through the space
for i, x in enumerate(np.linspace(0, 1, 100)):
    for j, y in enumerate(np.linspace(1, 0, 100)):
        allocated_class = np.array([None, None, None, None, None])
        min_dist = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        # Run through the training set
        for k in range(len(x_coord)):
            # Compute squared Euclidean distance (remove sqrt to save flops)
            d = (x - x_coord[k])**2 + (y - y_coord[k])**2
            # Check is this is closest seen so far
            if d < min_dist[4]:
                min_dist[4] = d
                allocated_class[4] = label[k]

                # Sort in descending order
                index = np.argsort(min_dist)
                min_dist = min_dist[index]
                allocated_class = allocated_class[index]

        # Plot
        list_mode = stats.mode(allocated_class.tolist())
        if list_mode[0] == [1]:
            D[j, i] = 1

for i, val in enumerate(label):
    if val == 1:
        axes.plot(x_coord[i], y_coord[i], 'ro')
    else:
        axes.plot(x_coord[i], y_coord[i], 'bo')
axes.imshow(D, extent = [0, 1, 0, 1], cmap=plt.get_cmap('Pastel1'), aspect = 1)
plt.savefig('img/fig7')

#############
# Figure 8
#############
x_coord = [0.90, 0.30, 0.71, 0.79, 0.50, 0.69, 0.61, 0.98, 0.76, 0.77, 0.22, 0.45, 0.57, 0.59, 0.80, 0.91, 0.99, 0.24, 0.4, 0.35]
y_coord = [0.43, 0.40, 0.52, 0.63, 0.80, 0.35, 0.72, 0.04, 0.19, 0.27, 0.61, 0.85, 0.25, 0.46, 0.81, 0.70, 0.99, 0.30, 0.95, 0.48]
label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
class_1 = []
class_2 = []
fig, axes = plt.subplots(ncols=1, nrows=1)
for i, val in enumerate(label):
    for _ in range(5):
        x = x_coord[i]+0.05*np.random.randn(1)
        y = y_coord[i]+0.05*np.random.randn(1)
        if val == 1:
            class_1.append([x, y])
            axes.plot(x, y, 'ro')
        else:
            class_2.append([x, y])
            axes.plot(x, y, 'bo')
axes.set(xlim=(0, 1), ylim=(0, 1), aspect=1)
plt.savefig('img/fig8')

#############
# Figure 9
#############
axes.plot(0.5, 0.6, 'gx')
plt.savefig('img/fig9')

#############
# Figure 10
#############
for [x,y] in class_1+class_2:
    if 0<x<1 and 0<y<1:
        axes.annotate('', xy=(0.5,0.6), xytext=(x,y), arrowprops=dict(arrowstyle='<->', color='black'), va='center')
plt.savefig('img/fig10')

#############
# Figure 11
#############
axes.plot([0, 0.2, 0.2, 0, 0], [0.7, 0.7, 1, 1, 0.7], 'g')
plt.savefig('img/fig11')
