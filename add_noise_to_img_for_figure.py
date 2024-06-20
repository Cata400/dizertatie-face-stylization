import PIL.Image
from skimage import io
import numpy as np
import PIL
from matplotlib import pyplot as plt


x = np.random.normal(size=(1024, 1024, 3))
# x = np.uint8(255 * (-x))

image = PIL.Image.open('../Datasets/celeba_hq_lmdb/raw_images/train/images/105409.jpg')
image = np.array(image)
image = image / 255
image = image + 0.75 * x
image = np.clip(image, a_min=0, a_max=1)

plt.imshow(image)
plt.axis('off')
plt.show()

