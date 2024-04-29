import os
from PIL import Image
import random
import numpy as np


if __name__ == '__main__':
    img_path = os.path.join('..', 'Images', 'JojoGAN', 'Aligned', 'arnold_aligned_aligned.jpg')
    save_path = os.path.join('..', 'Images', 'JojoGAN', 'Aligned', 'arnold_aligned_aligned_no_slice_10.jpg')
    
    image = Image.open(img_path)
    width, height = image.size
    
    ratio = 0.25
    slice_width = int(width * ratio)
    slice_height = int(height * ratio)
    
    print(f'Image size: {width}x{height}')
    print(f'Slice size: {slice_width}x{slice_height}')
    
    x = random.randint(0, width - slice_width)
    y = random.randint(0, height - slice_height)
    
    image_np = np.array(image)
    image_np[x: x + slice_height, y: y + slice_width] = 0
    
    output_image = Image.fromarray(image_np)
    output_image.save(save_path)