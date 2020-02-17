from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

#############
# Figure 1
#############
fig, axes = plt.subplots(ncols=1, nrows=1)
time_series = np.sin(np.linspace(0, 6*np.pi, 1000)) + 0.5*np.random.randn(1000)
axes.plot(time_series, 'k')
plt.savefig('img/fig1')

#############
# Figure 2
#############
axes.plot(np.convolve(time_series, 1/3*np.ones(3), mode='same'), 'r')
axes.plot(np.convolve(time_series, 1/9*np.ones(9), mode='same'), 'g')
axes.plot(np.convolve(time_series, 1/27*np.ones(27), mode='same'), 'b')
axes.plot(np.convolve(time_series, 1/81*np.ones(81), mode='same'), 'y')
plt.savefig('img/fig2')

#############
# Figure 3
#############
cat_image = data.chelsea()
plt.imshow(cat_image)
plt.savefig('img/fig3')

#############
# Figure 4
#############
cat_image = data.chelsea()
bw_cat_image = np.mean(cat_image, axis=2)
plt.imshow(bw_cat_image, cmap='gray')
plt.savefig('img/fig4')

#############
# Figure 5
#############
# Box blur
filter = (1/9)*np.ones((3, 3))
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig5')

#############
# Figure 6
#############
# Box blur
filter = (1/49)*np.ones((7, 7))
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig6')

#############
# Figure 7
#############
# Gaussian blur
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


filter = gkern(l=7, sig=1)
#print(filter)
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig7')

#############
# Figure 8
#############
# Horizontal lines
filter = -1*np.ones((3, 3))
filter[1,:] = 2*np.ones((1, 3))
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig8')

#############
# Figure 9
#############
# Vertical lines
filter = -1*np.ones((3, 3))
filter[:,1] = 2*np.ones((1, 3))
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig9')

#############
# Figure 10
#############
# All edges lines
filter = -1*np.ones((3, 3))
filter[1,1] = 8
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig10')

#############
# Figure 11
#############
#Random convolve
filter = np.random.randn(9, 9)
plt.imshow(convolve2d(bw_cat_image, filter, mode='same'), cmap='gray')
plt.savefig('img/fig11')
