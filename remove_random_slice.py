import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    # img_path = os.path.join('..', 'Images', 'JojoGAN', 'Aligned', 'arnold_aligned_aligned.jpg')
    # save_path = os.path.join('..', 'Images', 'JojoGAN', 'Aligned', 'arnold_aligned_aligned_no_slice_10.jpg')
    
    ratio = 0.3
    dataset_name = 'ffhq1k'
    source_path = os.path.join('..', 'Datasets', dataset_name)
    dest_path = os.path.join('..', 'Datasets', f'{dataset_name}_random_slice_{ratio}')
    
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for img_name in tqdm(sorted(os.listdir(source_path))):
        img_path = os.path.join(source_path, img_name)
        save_path = os.path.join(dest_path, img_name)
        
        image = Image.open(img_path)
        width, height = image.size
        
        slice_width = int(width * ratio)
        slice_height = int(height * ratio)
        
        # print(f'Image size: {width}x{height}')
        # print(f'Slice size: {slice_width}x{slice_height}')
        
        x = random.randint(0, width - slice_width)
        y = random.randint(0, height - slice_height)
        
        image_np = np.array(image)
        image_np[x: x + slice_height, y: y + slice_width] = 0
        
        output_image = Image.fromarray(image_np)
        output_image.save(save_path)