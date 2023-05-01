import os
import matplotlib.pyplot as plt
import torch
import torchvision

from restyle.models.e4e import e4e

from invert_gan_utils import *

from argparse import Namespace
from configs import data_configs


if __name__ == '__main__':
    IMAGE_PATH = os.path.join('..', 'Images', 'Input', 'Lenna.png')
    INVERT_GAN_PATH = os.path.join('..', 'Models', 'restyle_gan.pt')

    image = torchvision.io.read_image(IMAGE_PATH)

    plt.figure(), plt.imshow(image.permute(1, 2, 0).numpy()), plt.show()

    checkpoint = torch.load(INVERT_GAN_PATH)
    invert_gan_state_dict = checkpoint['state_dict']
    invert_gan_opts = checkpoint['opts']
    invert_gan_opts = Namespace(**invert_gan_opts)
    
    invert_gan = e4e(invert_gan_opts)
    invert_gan.load_state_dict(invert_gan_state_dict)
    invert_gan.eval()
    invert_gan.to(invert_gan_opts.device)
    
    dataset_args = data_configs.DATASETS[invert_gan_opts.dataset_type]
    transforms_dict = dataset_args['transforms'](invert_gan_opts).get_transforms()
    
    print(dataset_args)
    
    print("-------------")
    
    print(transforms_dict)
    

    print(invert_gan_state_dict)
    print("-------------")
    print(invert_gan_opts)




