import os
import time

import numpy as np
from PIL import Image
from torchvision import transforms

from invert_gan_utils import invert_image
from general_utils import *


from restyle.models.stylegan2.model import Generator, Discriminator
from copy import deepcopy


if __name__ == '__main__':
    # Step 1 parameters
    REFERENCE_IMAGE_NAME = 'Gulie'
    REFERENCE_IMAGE_EXT = '.jpeg'
    REFERENCE_IMAGE_PATH = os.path.join('..', 'Images', 'References', REFERENCE_IMAGE_NAME + REFERENCE_IMAGE_EXT)
    INVERT_MODEL_NAME = 'e4e'
    INVERT_GAN_PATH = os.path.join('..', 'Models', 'restyle_' + INVERT_MODEL_NAME + '.pt')
    
    # Step 2 parameters
    REFERENCE_IMAGE_TRAINING_SET_PATH = os.path.join('..', 'Images', 'Training set', REFERENCE_IMAGE_NAME)
    ALPHA = 0.5 # Strength of the finetuned style
    PRESERVE_COLOR = True # Whether to preserve the color of the reference image
    EPOCHS = 300
    
    # Step 4 parameters
    INPUT_IMAGE_NAME = 'Lenna'
    INPUT_IMAGE_EXT = '.png'
    INPUT_IMAGE_PATH = os.path.join('..', 'Images', 'Input', INPUT_IMAGE_NAME + INPUT_IMAGE_EXT)

    OUTPUT_IMAGE_NAME = 'Lenna'
    OUTPUT_IMAGE_PATH = os.path.join('..', 'Images', 'OUTPUT', OUTPUT_IMAGE_NAME + INPUT_IMAGE_EXT)
    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    
    ### Step 1: GAN Invert
    print('Inverting reference image...')
    reference_aligned = align_face(os.path.join('..', 'Models', 'face_lendmarks.dat'), REFERENCE_IMAGE_PATH)
    
    style_target, style_latent = invert_image(reference_aligned, INVERT_GAN_PATH, reps=1)

    style_target.save(os.path.join('..', 'Images', 'Inverted', REFERENCE_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_' + REFERENCE_IMAGE_EXT))
    np.save(os.path.join('..', 'Images', 'Inverted latents', REFERENCE_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_latent_code.npy'), style_latent)
    
    ### Step 2: Create training set
    # Prepare the data
    style_target = transform(style_target)
    style_target = style_target.unsqueeze(0).to('cuda')
    
    style_latent = torch.from_numpy(style_latent).unsqueeze(0).to('cuda')
    print(style_latent.shape, style_target.shape)
    
    # Load the generator
    generator = Generator(1024, 512, 8, 2).to('cuda')
    stylegan_checkpoint = torch.load(os.path.join('..', 'Models', 'stylegan2_config.pt'))
    generator.load_state_dict(stylegan_checkpoint['g_ema'], strict=False)

    # Generator to be finetuned
    # generator = deepcopy(original_generator)
    
    # Load the discriminator
    discriminator = Discriminator(1024, 2).to('cuda')
    discriminator.load_state_dict(stylegan_checkpoint['d'], strict=False)
    
    # Optimizer
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0,99))
    
    # Which layers to swap to generate a set of plausible real images
    if PRESERVE_COLOR:
        id_swap = [9, 11, 15, 16, 17]
    else:
        id_swap = list(range(7, generator.n_latent))
        
    # Finetune the generator
    for epoch in range(EPOCHS):
        mean_latent_code = generator.get_latent(torch.randn((style_latent.shape[0], style_latent.shape[-1])).to('cuda')).unsqueeze(1).repeat(1, generator.n_latent, 1)
        input_latent = style_latent.clone()
        input_latent[:, id_swap] = ALPHA * style_latent[:, id_swap] + (1 - ALPHA) * mean_latent_code[:, id_swap]
        
        new_style_image = generator(input_latent.cuda(), input_is_latent=True)
        print("new_style_image.shape:", new_style_image.shape)
        
        with torch.no_grad():
            real_features = discriminator(style_target)
        fake_features = discriminator(new_style_image)
        
        print("Len(real_features):", len(real_features))
        loss = sum([F.l1_loss(a, b) for a, b in zip(real_features, fake_features)]) / len(real_features)
        
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        
        print("Epoch:", epoch, "Loss:", loss.item())
        
        
    del discriminator
    torch.cuda.empty_cache()
    
    ### Step 4: Generate the output image
    print('Inverting input image...')
    input_aligned = align_face(os.path.join('..', 'Models', 'face_lendmarks.dat'), INPUT_IMAGE_PATH)
    
    input_target, input_latent = invert_image(input_aligned, INVERT_GAN_PATH, reps=1)

    input_target.save(os.path.join('..', 'Images', 'Inverted', INPUT_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_' + INPUT_IMAGE_EXT))
    np.save(os.path.join('..', 'Images', 'Inverted latents', INPUT_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_latent_code.npy'), input_latent)
    
    with torch.no_grad():
        generator.eval()
        new_style_image = generator(input_latent.cuda(), input_is_latent=True)
        
    new_style_image = new_style_image.cpu()
    new_style_image = 1 + new_style_image
    new_style_image /= 2
    print("new_style_image.shape:", new_style_image.shape)
    
    new_style_image = new_style_image[0]
    new_style_image = 255 * torch.clip(new_style_image, 0, 1)
    new_style_image = new_style_image.permute(1, 2, 0).numpy()
    new_style_image = Image.fromarray(new_style_image, mode='RGB')
    new_style_image.save(OUTPUT_IMAGE_PATH)