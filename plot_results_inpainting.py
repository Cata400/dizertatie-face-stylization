import os
import PIL
import matplotlib.pyplot as plt
import random

num_images = 1
content_path = '/home/catalin/Desktop/Disertatie/Datasets/ffhq1k'
images = random.choices(os.listdir(content_path), k=num_images)
image_name = images[0]

styles = {
    'celeba': '/home/catalin/Desktop/Disertatie/Datasets/celeba_hq_lmdb/raw_images/test/images/',
    'aahq': '/home/catalin/Desktop/Disertatie/Datasets/aahq/aligned_used/',
    'sketches': '/home/catalin/Desktop/Disertatie/Datasets/sketches/sketches_all_resized/'
    }


algorithm = 'Projector'
output_dirs = [
    f'../Results/{algorithm}_ffhq_celeba_1k',
    f'../Results/{algorithm}_ffhq_aahq_1k',
    f'../Results/{algorithm}_ffhq_sketches_1k',
]

additional_datasets = ['', '_random_slice_0.1', '_random_slice_0.2', '_random_slice_0.3']


fig, axs = plt.subplots(4, 5, figsize=(20, 12))
fig.subplots_adjust(wspace=0.05)
axs[0, 0].axis('off')

# First row
content_images = []
for i, ad in enumerate(additional_datasets):
    image_path = os.path.join(content_path + ad, image_name)
    image = PIL.Image.open(image_path)
    
    axs[0][i + 1].imshow(image)
    axs[0][i + 1].axis('off')

    
    axs[0][1].set_title('Original')
    axs[0][2].set_title('Patch size 10%')
    axs[0][3].set_title('Patch size 20%')
    axs[0][4].set_title('Patch size 30%')
    
    axs[0][1].title.set_fontsize(20)
    axs[0][2].title.set_fontsize(20)
    axs[0][3].title.set_fontsize(20)
    axs[0][4].title.set_fontsize(20)



for output_dir in output_dirs:
    for img in os.listdir(output_dir):
        if image_name.split('.')[0] in img:
            output_image_name = img
            break
        
    style_image = output_image_name.split('+')[1]
    if 'JojoGAN' in algorithm:
        style_image = "_".join(style_image.split('_')[:-2]) + '.png'
    
    if 'celeba' in output_dir:
        style_image = style_image.replace('png', 'jpg')
    
    style = output_dir.split('_')[-2]
    style_path = os.path.join(styles[style], style_image)
    style_image = PIL.Image.open(style_path)
    
    axs[output_dirs.index(output_dir) + 1][0].imshow(style_image)
    
    if style == 'celeba':
        axs[output_dirs.index(output_dir) + 1][0].set_title('CelebA-HQ')
    elif style == 'aahq':
        axs[output_dirs.index(output_dir) + 1][0].set_title('AAHQ')
    elif style == 'sketches':
        axs[output_dirs.index(output_dir) + 1][0].set_title('Sketches')
        
    axs[output_dirs.index(output_dir) + 1][0].title.set_fontsize(20)

    
    for i, ad in enumerate(additional_datasets):
        image_path = os.path.join(output_dir + ad, output_image_name)
        image = PIL.Image.open(image_path)
        
        axs[output_dirs.index(output_dir) + 1][i + 1].imshow(image)
        axs[output_dirs.index(output_dir) + 1][i + 1].axis('off')
        axs[output_dirs.index(output_dir) + 1][i + 1].title.set_fontsize(20)
        
        axs[output_dirs.index(output_dir) + 1][i].axis('off')

fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.15, left=0.1, right=0.9)
plt.show()