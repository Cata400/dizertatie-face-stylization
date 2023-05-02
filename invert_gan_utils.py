import matplotlib.pyplot as plt
import torch

from argparse import Namespace
from PIL import Image

from restyle.configs import data_configs
from restyle.models.e4e import e4e
from restyle.utils.inference_utils import run_on_batch, get_average_image
from restyle.utils.common import tensor2im


def invert_image(image, invert_gan_path):
    # Read checkpoint
    checkpoint = torch.load(invert_gan_path, map_location='cpu')
    invert_gan_state_dict = checkpoint['state_dict']
    invert_gan_opts = checkpoint['opts']
    
    # Override some options
    invert_gan_opts['checkpoint_path'] = invert_gan_path
    invert_gan_opts['batch_size'] = 1
    invert_gan_opts['n_iters_per_batch'] = 1
    invert_gan_opts = Namespace(**invert_gan_opts)
    
    # Load Invert GAN model
    invert_gan = e4e(invert_gan_opts)
    invert_gan.load_state_dict(invert_gan_state_dict)
    invert_gan.eval()
    invert_gan.to(invert_gan_opts.device)
    
    # Prepare image
    dataset_args = data_configs.DATASETS[invert_gan_opts.dataset_type]
    transforms_dict = dataset_args['transforms'](invert_gan_opts).get_transforms()
    image = transforms_dict['transform_inference'](image)
    image = image.unsqueeze(0)
    
    average_image = get_average_image(invert_gan, invert_gan_opts)
    
    resize_amount = (invert_gan_opts.output_size, invert_gan_opts.output_size)
        
    with torch.no_grad():
        input_cuda = image.cuda().float()
        result, result_latent = run_on_batch(input_cuda, invert_gan, invert_gan_opts, average_image)
                
        result = tensor2im(result[0][0])
        result.resize(resize_amount)

    return result, result_latent[0][0]