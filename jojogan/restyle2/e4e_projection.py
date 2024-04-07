import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from restyle2.e4e.models.psp import pSp
from restyle2.e4e.utils.common import tensor2im


@ torch.no_grad()
def invert_image(img, model_path, device='cuda'):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    return tensor2im(images[0]).resize((1024, 1024)), w_plus[0].cpu().numpy()
