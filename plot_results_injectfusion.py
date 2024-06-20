import os
import PIL
import matplotlib.pyplot as plt
import random

style = 'celeba'
content = 'ffhq'

num_images = 6

dataset_path = os.path.join('..', 'Results', f'injectfusion_ffhq_{style}_1k')
projector_path = os.path.join('..', 'Results', f'Projector_ffhq_{style}_1k')
images = os.listdir(projector_path)

images = random.choices(images, k=num_images)

fig, axs = plt.subplots(num_images, 4, figsize=(20, 12))
fig.subplots_adjust(wspace=0.05)

styles = {
    'celeba': '/home/catalin/Desktop/Disertatie/Datasets/celeba_hq_lmdb/raw_images/test/images/',
    'aahq': '/home/catalin/Desktop/Disertatie/Datasets/aahq/aligned_used/',
    'sketches': '/home/catalin/Desktop/Disertatie/Datasets/sketches/sketches_all_resized/'
    }

contents = {
    'ffhq': '/home/catalin/Desktop/Disertatie/Datasets/ffhq1k/'
}



for i, image_name in enumerate(images):
    image_path = os.path.join(dataset_path, image_name)
    image = PIL.Image.open(image_path)
    
    image_path2 = os.path.join(projector_path, image_name)
    image2 = PIL.Image.open(image_path2)
    
    content_image = image_name.split('+')[0] + '.png'
    content_path = os.path.join(contents[content], content_image)
    content_image = PIL.Image.open(content_path)
    
    style_image = image_name.split('+')[1]
    if 'JojoGAN' in image_path:
        style_image = "_".join(style_image.split('_')[:-2]) + '.png'
    
    if style == 'celeba':
        style_image = style_image.replace('png', 'jpg')
        
    style_path = os.path.join(styles[style], style_image)
    style_image = PIL.Image.open(style_path)
    
    axs[i][0].imshow(content_image)
    axs[i][0].axis('off')
    
    # place image_name to the left of the  image
    # axs[i][0].text(-150, 100, image_name, fontsize=8, color='black', weight='bold')#, rotation=90)
    
    axs[i][1].imshow(style_image)
    axs[i][1].axis('off')
    
    axs[i][2].imshow(image)
    axs[i][2].axis('off')
    
    axs[i][3].imshow(image2)
    axs[i][3].axis('off')
    
        
    if i == 0:
        axs[i][0].set_title('Content')
        axs[i][1].set_title('Style')
        axs[i][2].set_title('InjectFusion')
        axs[i][3].set_title('InjectFusionVGG')
        
        for x in axs[i]:
            x.title.set_fontsize(20)

fig.tight_layout()
# fig.subplots_adjust(wspace=0.05)
plt.subplots_adjust(wspace=0.01, hspace=0.1, left=0.28, right=0.72)
plt.show()