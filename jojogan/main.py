import os
import datetime

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from general_utils import *

from stylegan.model import Generator, Discriminator
from restyle.utils.common import tensor2im

from restyle.e4e_projection import invert_image

import time


if __name__ == '__main__':
    # Step 1 parameters
    REFERENCE_IMAGE_NAME = 'arcane_jinx'
    REFERENCE_IMAGE_EXT = '.png'
    REFERENCE_IMAGE_PATH = os.path.join('..', '..', 'Images', 'JojoGAN', 'Aligned', 'style', REFERENCE_IMAGE_NAME + REFERENCE_IMAGE_EXT)
    INVERT_MODEL_NAME = 'e4e'
    INVERT_GAN_PATH = os.path.join('..', '..', 'Models', INVERT_MODEL_NAME + '_ffhq_encode.pt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Step 2 parameters
    REFERENCE_IMAGE_TRAINING_SET_PATH = os.path.join('..', '..', 'Images', 'Training set', REFERENCE_IMAGE_NAME)
    ALPHA = 1 # Strength of the finetuned style
    PRESERVE_COLOR = False # Whether to preserve the color of the reference image
    EPOCHS = 300
    
    # Step 4 parameters
    INPUT_IMAGE_NAME = 'arnold_aligned'
    INPUT_IMAGE_EXT = '.jpg'
    INPUT_IMAGE_PATH = os.path.join('..', '..', 'Images', 'JojoGAN', 'Aligned', 'content', INPUT_IMAGE_NAME + INPUT_IMAGE_EXT)

    OUTPUT_IMAGE_NAME = INPUT_IMAGE_NAME + '+' + REFERENCE_IMAGE_NAME + '_ALPHA=' + str(ALPHA) + '_PC=' + str(PRESERVE_COLOR)
    OUTPUT_IMAGE_PATH = os.path.join('..', '..', 'Images', 'JojoGAN', 'Output', OUTPUT_IMAGE_NAME + INPUT_IMAGE_EXT)
    
    transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                    )
    
    start = time.time()
    ### Step 1: GAN Invert
    torch.cuda.empty_cache()

    print('Inverting reference image...')
    try:
        reference_aligned = align_face(os.path.join('..', '..', 'Models', 'face_lendmarks.dat'), REFERENCE_IMAGE_PATH)
    except AssertionError:
        print('Reference image is not a face. Skipping alignment.')
        reference_aligned = Image.open(REFERENCE_IMAGE_PATH).resize((1024, 1024))
    reference_aligned.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Aligned', REFERENCE_IMAGE_NAME + '_aligned' + REFERENCE_IMAGE_EXT))
    

    style_target, style_latent = invert_image(reference_aligned, INVERT_GAN_PATH)

    style_target.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Inverted', REFERENCE_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_' + REFERENCE_IMAGE_EXT))
    np.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Inverted latents', REFERENCE_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_latent_code.npy'), style_latent)
            
    
    ### Step 2: Create training set
    # Prepare the data
    style_target = transform(style_target)
    style_target = style_target.unsqueeze(0).to(device)
    
    style_latent = torch.from_numpy(style_latent).unsqueeze(0).to(device)
    
    print(style_target.shape)
    print(style_latent.shape)
    
    # Load the generator
    generator = Generator(1024, 512, 8, 2).to(device)
    stylegan_checkpoint = torch.load(os.path.join('..', '..', 'Models', 'stylegan2-ffhq-config-f.pt'))
    generator.load_state_dict(stylegan_checkpoint['g_ema'], strict=False)
    
    # Load the discriminator
    discriminator = Discriminator(1024, 2).to(device)
    discriminator.load_state_dict(stylegan_checkpoint['d'], strict=False)
    
    # Optimizer
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0, 0.99))
    
    # Which layers to swap to generate a set of plausible real images
    if PRESERVE_COLOR:
        id_swap = [9, 11, 15, 16, 17]
    else:
        id_swap = list(range(7, generator.n_latent))
        
    # Finetune the generator
    generator.train()
    discriminator.eval()
    ALPHA = 1 - ALPHA
    writer = SummaryWriter(log_dir=os.path.join('..', '..', 'Logs', OUTPUT_IMAGE_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    for epoch in range(EPOCHS):
        mean_latent_code = generator.get_latent(torch.randn((style_latent.shape[0], style_latent.shape[-1])).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        input_latent = style_latent.clone()
        input_latent[:, id_swap] = ALPHA * style_latent[:, id_swap] + (1 - ALPHA) * mean_latent_code[:, id_swap]
        
        new_style_image = generator(input_latent.to(device), input_is_latent=True)
        
        if epoch % 10 == 0:
            new_style_pil = tensor2im(new_style_image.squeeze(0))
            new_style_pil.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Training Set', REFERENCE_IMAGE_NAME + '_ALPHA=' + str(ALPHA) + '_PC=' + str(PRESERVE_COLOR) + '_' + str(epoch) + '.jpg'))
        
        with torch.no_grad():
            real_features = discriminator(style_target)
        fake_features = discriminator(new_style_image)
        
        loss = sum([F.l1_loss(x, y) for x, y in zip(fake_features, real_features)]) / len(real_features)
        
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        
        # print("Epoch ", epoch, "Loss:", loss.item())
        writer.add_scalar('Loss', loss.item(), epoch)
        
    del discriminator
    torch.cuda.empty_cache()
    
    ### Step 4: Generate the output image
    print('Inverting input image...')
    try:
        input_aligned = align_face(os.path.join('..', '..', 'Models', 'face_lendmarks.dat'), INPUT_IMAGE_PATH)
    except AssertionError:
        input_aligned = Image.open(INPUT_IMAGE_PATH).resize((1024, 1024))   
        input_aligned.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Aligned', INPUT_IMAGE_NAME + '_aligned' + INPUT_IMAGE_EXT))

    input_target, input_latent = invert_image(input_aligned, INVERT_GAN_PATH)
    
    input_target.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Inverted', INPUT_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_' + INPUT_IMAGE_EXT))
    np.save(os.path.join('..', '..', 'Images', 'JojoGAN', 'Inverted latents', INPUT_IMAGE_NAME + '_invert_' + INVERT_MODEL_NAME + '_latent_code.npy'), input_latent)
    
    input_latent = torch.from_numpy(input_latent).unsqueeze(0).to(device)

    generator.eval()
    with torch.no_grad():
        new_style_image = generator(input_latent, input_is_latent=True)
    print(new_style_image.shape)
    
    new_style_image = tensor2im(new_style_image.squeeze(0))
    new_style_image = new_style_image.resize((input_target.width, input_target.height))
    new_style_image.save(OUTPUT_IMAGE_PATH)
    
    del generator
    torch.cuda.empty_cache()
    end = time.time()
    print(f'Time: {end - start:.2f} seconds')