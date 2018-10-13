from matplotlib import pyplot as plt
from sklearn.datasets import load_sample_image
import numpy as np
from kmeans import Kmeans

# 1: Load an image

img = load_sample_image('flower.jpg')
w, h, d = img.shape
img = np.array(img, dtype=np.float64)/255. # Normalize the image between 0. and 1.
img_arr = np.reshape(img, (w*h, d))        # Reshape the image as a data vector

# 2: Apply K-means

inst = Kmeans(img_arr,5)
inst.run(0.001)

# 3: Image quantization

img_quantization = np.zeros((w, h, d))
for i in range(w):
    for j in range(h):
        img_quantization[i,j,:] = inst.mu[inst.labels[i*h+j]]

# 4: Display

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
ax[0].imshow(img)
ax[1].imshow(img_quantization)
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
