import os
import PIL
import matplotlib.pyplot as plt
import random

# dataset_path = os.path.join('..', 'Datasets', 'aahq', 'aligned')
# images = random.choices(os.listdir(dataset_path), k=16)

dataset_path = os.path.join('..', 'Results', 'VGG_latex', 'reverse')
images = sorted(os.listdir(dataset_path))


fig, axs = plt.subplots(3, 4, figsize=(20, 12))

for i, ax in enumerate(axs.flat):
    image_path = os.path.join(dataset_path, images[i])
    image = PIL.Image.open(image_path)
    ax.imshow(image)
    ax.axis('off')
        
    if i == axs.size - 1:
        ax.set_title('Output')
    else:
        ax.set_title(f'Timestep {int(images[i].split("_")[1].split(".")[0])}')
    ax.title.set_fontsize(20)


fig.tight_layout()
plt.show()