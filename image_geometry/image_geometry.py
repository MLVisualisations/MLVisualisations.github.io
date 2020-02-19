import matplotlib.pyplot as plt
import numpy as np
import imageio

#############
# Figure 1
#############
cat1 = imageio.imread('cat1.jpg')
cat1 = cat1[:, 10:290, :]
cat1 = np.mean(cat1, axis=2)
plt.imshow(cat1, cmap='gray')
plt.savefig('img/fig1')

#############
# Figure 2
#############
U, S, V = np.linalg.svd(cat1, full_matrices=True)
print(np.allclose(cat1, np.dot(U*S, V)))
plt.imshow(np.dot(U[:,:10]*S[:10], V[:10,:]), cmap='gray')
plt.savefig('img/fig2')

#############
# Figure 3
#############
plt.imshow(np.dot(U[:,:5]*S[:5], V[:5,:]), cmap='gray')
plt.savefig('img/fig3')

#############
# Figure 4
#############
