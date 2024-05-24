import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from diffusion_latent import Asyrp

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == '__main__':
    gpu = 0

    config = "celeba.yml"     # Other option: afhq.yml celeba.yml metfaces.yml ffhq.yml lsun_bedroom.yml ...
    save_dir = "./saved"   # output directory
    content_dir = "./content"
    style_dir = "./style"
    model_path = "../../Models/celeba_hq.ckpt"
    
    content_dir = '/home/catalin/Desktop/Disertatie/dizertatie-face-stylization/injectfusion/test/content'
    style_dir = '/home/catalin/Desktop/Disertatie/dizertatie-face-stylization/injectfusion/test/style'
    save_dir = '/home/catalin/Desktop/Disertatie/Results/VGG_projector_test2'
    
    
    h_gamma = 0.6
    dt_lambda = 0.9985      # 1.0 for out-of-domain style transfer.
    t_boost = 200           # 0 for out-of-domain style transfer.
    t_edit = 400
    omega = 0.3

    n_gen_step = 1000
    n_inv_step = 1000

    seed = 1234
    
    args = argparse.Namespace(
        gpu=gpu,
        config=config,
        save_dir=save_dir,
        content_dir=content_dir,
        style_dir=style_dir,
        hs_coeff=h_gamma,
        dt_lambda=dt_lambda,
        t_noise=t_boost,
        user_defined_t_edit=t_edit,
        n_gen_step=n_gen_step,
        n_inv_step=n_inv_step,
        omega=omega,
        seed=seed,
        model_path=model_path,
        use_mask=False,
        # Unimportant parameters
        edit_attr=None,
        src_txts="append",
        trg_txts="append",
        t_0=999,
        content_replace_step=50,
        sample_type='ddim',
        dt_end=950,
    )
    
    # parse config file
    with open(os.path.join('configs', config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    print(new_config)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    
    runner = Asyrp(args, new_config)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    runner.diff_style()
