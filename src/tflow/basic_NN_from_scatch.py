import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


v = np.array([1,2,3])
w = np.array([4,5,6])

x = np.array([[1,2,3,5],[4,5,6,7],[8,19,10,0],[9,3,1,9]])

b = x[0:2,1:2]
print b

a = np.arange(4)
print a


img = imread('8.tiff')

img_tinted = img

img_tinted = imresize(img_tinted,(224,224))

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted))
plt.imshow(img_tinted)
plt.show()