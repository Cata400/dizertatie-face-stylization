import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np


# img_paths = os.listdir('imgs_injectfusion')
# for img_path in img_paths:
#     img = Image.open('imgs_injectfusion/' + img_path)
#     w, h = img.size
#     img = img.crop((0, 2 * h // 3, w, h))
#     print(img_path, h, w, img.size)
#     img = img.resize((1024, 1024))
#     img.save('imgs_injectfusion_cropped/' + img_path)


# img_paths = os.listdir('forward')
# print(img_paths)
# imgs = [Image.open('forward/' + img_path) for img_path in sorted(img_paths)]
# fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# for i in range(3):
#     for j in range(4):
#         axes[i, j].imshow(imgs[i * 4 + j])
#         axes[i, j].axis('off')
#         if i == 2 and j == 3:
#             axes[i, j].set_title(f"Step 1000")
#         else:
#             axes[i, j].set_title(f"Step {90 * (i * 4 + j)}")
        
# plt.savefig('forward.png')


img = Image.open('Images/Aligned/Style/naruto_aligned.png')
# img = img.resize((1024, 1024))
# img.save('Images/Aligned/Style/naruto_aligned2.png')
print(img.size)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(max(*img.size)), transforms.CenterCrop(256), transforms.Resize(1024), transforms.ToPILImage()])
# img = transform(img)
# img.save('Images/Aligned/Style/naruto_aligned4.png')
img_np = np.array(img)
img_np = img_np[:, (300 - 224) // 2: - (300 - 224) // 2, :]
print(img_np.shape)
img_pil = Image.fromarray(img_np)
img_pil = img_pil.resize((1024, 1024))
img_pil.save('Images/Aligned/Style/naruto_aligned5.png')