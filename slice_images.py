import numpy as np
from skimage import io
import os


for file in os.listdir("inject_fusion_all"):
    img = io.imread(f"inject_fusion_all/{file}")
    h, w, c = img.shape
    img = img[2 * h // 3:, :, :]
    
    io.imsave(f'inject_fusion/{file}', img)