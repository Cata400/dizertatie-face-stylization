from skimage import io
import numpy as np



x = np.random.normal(size=(32, 32, 3))
x = np.uint8(255 * (-x))
io.imsave("reverse2.png", x)