import os
import datetime

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from invert_gan_utils import invert_image
from general_utils import *

from stylegan.model import Generator, Discriminator
from restyle.utils.common import tensor2im
from copy import deepcopy

from restyle2.e4e_projection import invert_image as invert_image2

import random
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


if __name__ == '__main__':
    # Step 1 parameters
    INVERT_MODEL_NAME = 'e4e'
    INVERT_GAN_PATH = os.path.join('..', '..', 'Models', INVERT_MODEL_NAME + '_ffhq_encode.pt')
    # INVERT_GAN_PATH = os.path.join('..', 'Models', 'restyle_' + INVERT_MODEL_NAME + '_2.pt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Step 2 parameters
    ALPHA = 1 # Strength of the finetuned style
    ALPHA = 1 - ALPHA
    PRESERVE_COLOR = False # Whether to preserve the color of the reference image
    EPOCHS = 300

    seed = 42
    torch.random.manual_seed(seed)
    random.seed(seed)
    
    input_dataset_path = os.path.join('..', '..', 'Datasets', 'ffhq1k_random_slice_0.1')
    input_image_files = sorted(os.listdir(input_dataset_path))
    
    # reference_dataset_path = os.path.join('..', '..', 'Datasets', 'sketches', 'sketches_all_resized')
    reference_dataset_path = os.path.join('..', '..', 'Datasets', 'celeba_hq_lmdb', 'raw_images', 'test', 'images')
    reference_image_files = os.listdir(reference_dataset_path)
    random.shuffle(reference_image_files)
    reference_image_files = reference_image_files * 4
    
    # output_dataset_path = os.path.join('..', '..', 'Results', 'JojoGAN_ffhq_sketches_1k_random_slice_0.3')
    output_dataset_path = os.path.join('..', '..', 'Results', 'JojoGAN_ffhq_celeba_1k_random_slice_0.1')
    
    if not os.path.exists(output_dataset_path):
        os.makedirs(output_dataset_path)
    
    # Load models
    stylegan_checkpoint = torch.load(os.path.join('..', '..', 'Models', 'stylegan2-ffhq-config-f.pt'))
    generator = Generator(1024, 512, 8, 2).to(device)
    discriminator = Discriminator(1024, 2).to(device)
    discriminator.load_state_dict(stylegan_checkpoint['d'], strict=False)
    discriminator.eval()
    
    transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )
        
    for i, (input_image_file, reference_image_file) in enumerate(zip(input_image_files, reference_image_files)):
        print(f"{i + 1} / 1000 ", input_image_file, reference_image_file)
        
        start = time.time()
        
        INPUT_IMAGE_PATH = os.path.join(input_dataset_path, input_image_file)
        REFERENCE_IMAGE_PATH = os.path.join(reference_dataset_path, reference_image_file)
        
        # Step 4 parameters
        OUTPUT_IMAGE_NAME = input_image_file.split('.')[0] + '+' + reference_image_file.split('.')[0] + '_ALPHA=' + str(ALPHA) + '_PC=' + str(PRESERVE_COLOR) + '.png'
        OUTPUT_IMAGE_PATH = os.path.join(output_dataset_path, OUTPUT_IMAGE_NAME)
        
        ### Step 1: GAN Invert
        torch.cuda.empty_cache()

        try:
            reference_aligned = align_face(os.path.join('..', '..', 'Models', 'face_lendmarks.dat'), REFERENCE_IMAGE_PATH)
        except AssertionError:
            reference_aligned = Image.open(REFERENCE_IMAGE_PATH).resize((1024, 1024))    

        style_target, style_latent = invert_image2(reference_aligned, INVERT_GAN_PATH)    
        
        # style_target.save(os.path.join(output_dataset_path, f'blabla_{i+1}.png'))
        
        ### Step 2: Create training set
        # Prepare the data
        style_target = transform(style_target)
        style_target = style_target.unsqueeze(0).to(device)
        
        style_latent = torch.from_numpy(style_latent).unsqueeze(0).to(device)
        
        # Load the generator
        generator.load_state_dict(stylegan_checkpoint['g_ema'], strict=False)
        
        # Optimizer
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0, 0.99))
        
        # generator = torch.compile(generator)
        
        # Which layers to swap to generate a set of plausible real images
        if PRESERVE_COLOR:
            id_swap = [9, 11, 15, 16, 17]
        else:
            id_swap = list(range(7, generator.n_latent))
            
        # Finetune the generator
        generator.train()
        writer = SummaryWriter(log_dir=os.path.join('..', '..', 'Logs', OUTPUT_IMAGE_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        
        for epoch in range(EPOCHS):
            mean_latent_code = generator.get_latent(torch.randn((style_latent.shape[0], style_latent.shape[-1])).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
            input_latent = style_latent.clone()
            input_latent[:, id_swap] = ALPHA * style_latent[:, id_swap] + (1 - ALPHA) * mean_latent_code[:, id_swap]
            
            new_style_image = generator(input_latent.to(device), input_is_latent=True)
            
            if epoch % 10 == 0:
                new_style_pil = tensor2im(new_style_image.squeeze(0))
            
            with torch.no_grad():
                real_features = discriminator(style_target)
            fake_features = discriminator(new_style_image)
            
            
            loss = sum([F.l1_loss(x, y) for x, y in zip(fake_features, real_features)]) / len(real_features)
            
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()
            
            writer.add_scalar('Loss', loss.item(), epoch)
        
        ### Step 4: Generate the output image
        try:
            input_aligned = align_face(os.path.join('..', '..', 'Models', 'face_lendmarks.dat'), INPUT_IMAGE_PATH)
        except AssertionError:
            input_aligned = Image.open(INPUT_IMAGE_PATH).resize((1024, 1024))   
        
        input_target, input_latent = invert_image2(input_aligned, INVERT_GAN_PATH)    
        input_latent = torch.from_numpy(input_latent).unsqueeze(0).to(device)

        generator.eval()
        with torch.no_grad():
            new_style_image = generator(input_latent, input_is_latent=True)
        
        new_style_image = tensor2im(new_style_image.squeeze(0))
        new_style_image = new_style_image.resize((input_target.width, input_target.height))
        new_style_image.save(OUTPUT_IMAGE_PATH)
        
        del style_latent
        del style_target
        del input_latent
        
        end = time.time()
        print(f"Time: {end - start:.2f} seconds")
