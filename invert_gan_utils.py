import matplotlib.pyplot as plt
import torch
import gc

import numpy as np

from argparse import Namespace
from torchvision import transforms

from restyle.models.e4e import e4e
from restyle.models.psp import pSp
from restyle.utils.inference_utils import run_on_batch, get_average_image
from restyle.utils.model_utils import ENCODER_TYPES
from restyle.utils.common import tensor2im


def invert_image(image, invert_gan_path, reps):
    # Read checkpoint
    checkpoint = torch.load(invert_gan_path, map_location='cpu')
    invert_gan_state_dict = checkpoint['state_dict']
    invert_gan_opts = checkpoint['opts']
    
    # Override some options
    invert_gan_opts['checkpoint_path'] = invert_gan_path
    invert_gan_opts['batch_size'] = 1
    invert_gan_opts['test_batch_size'] = 1
    invert_gan_opts['n_iters_per_batch'] = reps
    invert_gan_opts['output_size'] = 1024
    invert_gan_opts = Namespace(**invert_gan_opts)
            
    # Load Invert GAN model
    if invert_gan_opts.encoder_type in ENCODER_TYPES['e4e']:
        invert_gan = e4e(invert_gan_opts)
    else:
        invert_gan = pSp(invert_gan_opts)
    invert_gan.load_state_dict(invert_gan_state_dict)
    invert_gan.eval()
    invert_gan.to(invert_gan_opts.device)
    
    # Prepare image
    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    image = transform(image)
    image = image.unsqueeze(0).to(invert_gan_opts.device)
        
    average_image = get_average_image(invert_gan, invert_gan_opts)
    resize_amount = (invert_gan_opts.output_size, invert_gan_opts.output_size)
        
    with torch.no_grad():
        result, result_latent = run_on_batch(image, invert_gan, invert_gan_opts, average_image)
        results = [tensor2im(result[0][i]).resize(resize_amount) for i in range(invert_gan_opts.n_iters_per_batch)]
        
    del invert_gan
    torch.cuda.empty_cache()
    gc.collect()
    
    return results[-1], result_latent[0][-1]