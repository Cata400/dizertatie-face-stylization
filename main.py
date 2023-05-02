import os

import numpy as np
from PIL import Image

from invert_gan_utils import invert_image


if __name__ == '__main__':
    REFERENCE_IMAGE_NAME = 'Lenna'
    REFERENCE_IMAGE_EXT = '.png'
    REFERENCE_IMAGE_PATH = os.path.join('..', 'Images', 'References', REFERENCE_IMAGE_NAME + REFERENCE_IMAGE_EXT)
    REFERENCE_INVERT_GAN_PATH = os.path.join('..', 'Models', 'restyle_gan.pt')
    
    REFERENCE_IMAGE_TRAINING_SET_PATH = os.path.join('..', 'Images', 'Training set', REFERENCE_IMAGE_NAME)
    
    INPUT_IMAGE_NAME = 'Lenna'
    INPUT_IMAGE_EXT = '.png'
    INPUT_IMAGE_PATH = os.path.join('..', 'Images', 'Input', INPUT_IMAGE_NAME + INPUT_IMAGE_EXT)

    OUTPUT_IMAGE_NAME = 'Lenna'
    OUTPUT_IMAGE_PATH = os.path.join('..', 'Images', 'OUTPUT', OUTPUT_IMAGE_NAME + INPUT_IMAGE_EXT)
    

    ### Step 1: GAN Invert
    # Read reference image
    print('Inverting reference image...')
    reference_image = Image.open(REFERENCE_IMAGE_PATH).convert('RGB')

    result, result_latent = invert_image(reference_image, REFERENCE_INVERT_GAN_PATH)

    result.save(os.path.join('..', 'Images', 'Inverted', REFERENCE_IMAGE_NAME + '_invert_gan_' + REFERENCE_IMAGE_EXT))
    np.save(os.path.join('..', 'Images', 'Inverted latents', REFERENCE_IMAGE_NAME + '_invert_latent_code.npy'), result_latent)
    
    
    ### Step 2: Create training set