from audioop import reverse
from genericpath import isfile
import time
from glob import glob
from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.nn import normalization
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from losses.clip_loss import CLIPLoss
import random
import copy
from matplotlib import pyplot as plt
from PIL import Image

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from utils.text_dic import SRC_TRG_TXT_DIC
from losses import id_loss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from datasets.imagenet_dic import IMAGENET_DIC

class Asyrp(object):
    def __init__(self, args, config, device=None):
        # ----------- predefined parameters -----------#
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.learn_sigma = False # it will be changed in load_pretrained_model()

        # ----------- Editing txt -----------#
        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        elif self.args.edit_attr == "attribute":
            pass
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]


    def load_pretrained_model(self):

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset in ["CelebA_HQ", "CUSTOM", "CelebA_HQ_Dialog"]:
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MetFACE"]:
            # get the model ["FFHQ", "AFHQ", "MetFACE"] from 
            # https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            # reference : ILVR (https://arxiv.org/abs/2108.02938), P2 weighting (https://arxiv.org/abs/2204.00227)
            # reference github : https://github.com/jychoi118/ilvr_adm , https://github.com/jychoi118/P2-weighting 

            # get the model "IMAGENET" from
            # https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
            # reference : ADM (https://arxiv.org/abs/2105.05233)
            pass
        else:
            # if you want to use LSUN-horse, LSUN-cat -> https://github.com/openai/guided-diffusion
            # if you want to use CUB, Flowers -> https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN", "CelebA_HQ_Dialog"]:
            model = DDPM(self.config) 
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            self.learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset) #Get_h(self.config, model="i_DDPM", layer_num=self.args.get_h_num) #
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
            print("Improved diffusion Model loaded.")
        elif self.config.data.dataset in ["MetFACE"]:
            model = guided_Diffusion(self.config.data.dataset)
            init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt, strict=False)

        return model


    @torch.no_grad()
    def save_image(self, model, x_lat_tensor, seq_inv, seq_inv_next,
                    save_x0 = False, save_x_origin = False,
                    save_process_delta_h = False, save_process_origin = False,
                    x0_tensor = None, delta_h_dict=None, get_delta_hs=False,
                    folder_dir="", file_name="", hs_coeff=(1.0,1.0),
                    image_space_noise_dict=None):
        
        if save_process_origin or save_process_delta_h:
            os.makedirs(os.path.join(folder_dir,file_name), exist_ok=True)

        process_num = int(save_x_origin) + (len(hs_coeff) if isinstance(hs_coeff, list) else 1)
        

        with tqdm(total=len(seq_inv)*(process_num), desc=f"Generative process") as progress_bar:
            time_s = time.time()

            x_list = []

            if save_x0:
                if x0_tensor is not None:
                    x_list.append(x0_tensor.to(self.device))
            
            if save_x_origin:
            # No delta h
                x = x_lat_tensor.clone().to(self.device)

                for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    t = (torch.ones(self.args.bs_train) * i).to(self.device)
                    t_next = (torch.ones(self.args.bs_train) * j).to(self.device)

                    x, x0_t, _, _  = denoising_step(x, t=t, t_next=t_next, models=model,
                                    logvars=self.logvar,
                                    sampling_type= self.args.sample_type,
                                    b=self.betas,
                                    learn_sigma=self.learn_sigma,
                                    eta=1.0 if (self.args.origin_process_addnoise and t[0]<self.t_addnoise) else 0.0,
                                    )
                    progress_bar.update(1)
                    
                    if save_process_origin:
                        output = torch.cat([x, x0_t], dim=0)
                        output = (output + 1) * 0.5
                        grid = tvu.make_grid(output, nrow=self.args.bs_train, padding=1)
                        tvu.save_image(grid, os.path.join(folder_dir, file_name, f'origin_{int(t[0].item())}.png'), normalization=True)

                x_list.append(x)

            if self.args.pass_editing:
                pass
            else:
                if not isinstance(hs_coeff, list):
                    hs_coeff = [hs_coeff]
                
                for hs_coeff_tuple in hs_coeff:

                    x = x_lat_tensor.clone().to(self.device)

                    for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                        t = (torch.ones(self.args.bs_train) * i).to(self.device)
                        t_next = (torch.ones(self.args.bs_train) * j).to(self.device)

                        x, x0_t, delta_h, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                        logvars=self.logvar,                                
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        learn_sigma=self.learn_sigma,
                                        index=self.args.get_h_num-1 if not (self.args.image_space_noise_optim or self.args.image_space_noise_optim_delta_block) else None,
                                        eta=1.0 if t[0]<self.t_addnoise else 0.0,
                                        t_edit= self.t_edit,
                                        hs_coeff=hs_coeff_tuple,
                                        delta_h=None if get_delta_hs else delta_h_dict[0] if (self.args.ignore_timesteps and self.args.train_delta_h) else delta_h_dict[int(t[0].item())] if t[0]>= self.t_edit else None,
                                        ignore_timestep=self.args.ignore_timesteps,
                                        dt_lambda=self.args.dt_lambda,
                                        warigari=self.args.warigari,
                                        )
                        progress_bar.update(1)

                        if save_process_delta_h:
                            output = torch.cat([x, x0_t], dim=0)
                            output = (output + 1) * 0.5
                            grid = tvu.make_grid(output, nrow=self.args.bs_train, padding=1)
                            tvu.save_image(grid, os.path.join(folder_dir, file_name, f'delta_h_{int(t[0].item())}.png'), normalization=True)
                        if get_delta_hs and t[0]>= self.t_edit:
                            if delta_h_dict[t[0].item()] is None:
                                delta_h_dict[t[0].item()] = delta_h
                            else:
                                delta_h_dict[int(t[0].item())] = delta_h_dict[int(t[0].item())] + delta_h

                    x_list.append(x)

        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=self.args.bs_train, padding=1)

        tvu.save_image(grid, os.path.join(folder_dir, f'{file_name}_ngen{self.args.n_train_step}.png'), normalization=True)

        time_e = time.time()
        print(f'{time_e - time_s} seconds, {file_name}_ngen{self.args.n_train_step}.png is saved')


    # ----------- DiffStyle -----------#
    @torch.no_grad()
    def diff_style(self):

        print("Style transfer starts....")

        # ------------ Model ------------ #
        model = self.load_pretrained_model()
        model = model.to(self.device)


        # ----------- Pre-compute ----------- #
        content_lat_pairs = []
        style_lat_pairs = []

        # check if content and style dir are valid
        if not os.path.isdir(self.args.content_dir):
            raise ValueError("content_dir is not a valid directory")
        if not os.path.isdir(self.args.style_dir):
            raise ValueError("style_dir is not a valid directory")
        
        
        if os.path.isfile(self.args.content_dir) and os.path.isfile(self.args.style_dir):
            content_img_paths = [self.args.content_dir]
            style_img_paths = [self.args.style_dir]
        else:
            # list all file path in self.args.content_dir
            content_img_paths = [os.path.join(self.args.content_dir, f) for f in sorted(os.listdir(self.args.content_dir)) if os.path.isfile(os.path.join(self.args.content_dir, f)) and not os.path.isdir(os.path.join(self.args.content_dir, f))]
            style_img_paths = [os.path.join(self.args.style_dir, f) for f in os.listdir(self.args.style_dir) if os.path.isfile(os.path.join(self.args.style_dir, f)) and not os.path.isdir(os.path.join(self.args.style_dir, f))]

            random.shuffle(style_img_paths)
        
            if len(content_img_paths) > len(style_img_paths):
                style_img_paths = style_img_paths * (len(content_img_paths) // len(style_img_paths) + 1)
                
            if len(content_img_paths) == 1:
                content_img_paths = content_img_paths * len(style_img_paths)
            else:
                style_img_paths = style_img_paths[:len(content_img_paths)]
            
            content_img_paths = content_img_paths[750:]
            style_img_paths = style_img_paths[750:]

        # precompute content latent pairs
        for img_path in content_img_paths:
            print(f"- Content inversion {img_path.split(os.path.sep)[-1]}")
            start = time.time()
            content_lat_pairs.append(self.precompute_pairs_with_h(model, img_path))
            end = time.time()
            print(f"-- Time: {end - start:.2f} seconds")
            
        # precompute style latent pairs
        for img_path in style_img_paths:
            print(f"- Style inversion {img_path.split(os.path.sep)[-1]}")
            start = time.time()
            style_lat_pairs.append(self.precompute_pairs_with_h(model, img_path))
            end = time.time()
            print(f"-- Time: {end - start:.2f} seconds")

        # ----------- Get seq -----------#    
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        seq_train = np.linspace(0, 1, self.args.n_gen_step) * self.args.t_0
        # seq_train = seq_train[seq_train >= self.args.maintain]
        seq_train = [int(s+1e-6) for s in list(seq_train)]
        print('Uniform skip type')

        seq_train_next = [-1] + list(seq_train[:-1])

        if self.args.content_replace_step and self.args.content_replace_step != -1:
            seq_replace = np.linspace(0,1, self.args.content_replace_step) * self.args.t_0
            seq_replace = [int(s+1e-6) for s in list(seq_replace)]
            print('seq_replace: ', seq_replace)
        else:
            seq_replace = seq_train         

        if self.args.user_defined_t_edit is None:
            self.args.user_defined_t_edit = 400


        # ----------- save__dir ----------- #
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
            print("save_dir created")
        else:
            print("save_dir exists")
        

        # ----------- Diff style ----------- #


        # results_list = [style_lat_pair[0].cpu() for style_lat_pair in style_lat_pairs]
        # results_list.insert(0,torch.zeros_like(results_list[0]))


        def take_closest(input_list, value):
            #return closest value among input_list
            return min(input_list, key=lambda x: abs(x-value))

        inv_timesteps = sorted(list(content_lat_pairs[0][3].keys()),reverse=True)

        for i, (content_lat_pair, style_lat_pair) in enumerate(zip(content_lat_pairs, style_lat_pairs)):
            
            # results_list.append(content_lat_pair[0].cpu())
            
            print(f"{i + 1} / {len(content_lat_pairs)}")
            start = time.time()
            
            style_i = i

            content_path = content_img_paths[i]
            style_path = style_img_paths[style_i]

            # save_path = 'content_' + content_path.split(os.path.sep)[-1].split('.')[0] + '_style_' + style_path.split(os.path.sep)[-1].split('.')[0] + '.png'
            save_path = content_path.split(os.path.sep)[-1].split('.')[0] + "+" + style_path.split(os.path.sep)[-1].split('.')[0] + '.png'
            save_path = os.path.join(self.args.save_dir, save_path)

            
            X_T = None
            target_img_lat_pairs = []

            x_origin = [style_lat_pair[0].to('cpu')]
            X_T = [style_lat_pair[2].to(self.device)]

            target_img_lat_pairs = [content_lat_pair[3]]
            target_img_x0_pairs = [content_lat_pair[0]]


            X_T = torch.cat(X_T,dim=0)
            x_origin = torch.cat(x_origin,dim=0)
            target_img_x0_pairs = torch.cat(target_img_x0_pairs,dim=0)

            x = X_T.clone().to(self.device)
            noise_lat = X_T.clone().to(self.device)

            xt_original = x
            for t_it, (i,j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                # progress_bar.set_description(f"step_{t_it}")

                t = (torch.ones(X_T.shape[0]) * i).to(self.device)
                t_next = (torch.ones(X_T.shape[0]) * j).to(self.device)

                closest_t = take_closest(inv_timesteps, i)

                delta_hs = []
                for target_img_lat_pair in target_img_lat_pairs:
                    h_tmp = target_img_lat_pair[closest_t].to(self.device)
                    delta_hs.append(h_tmp)

                delta_hs = torch.cat(delta_hs,dim=0)
                if i in seq_replace:
                    x, _, coeff_before, xt_original = denoising_step(x, t=t, t_next=t_next, models=model,
                                                                    logvars=self.logvar,
                                                                    sampling_type=self.args.sample_type,
                                                                    b=self.betas,
                                                                    eta=0 if t[0].item()> self.args.t_noise else 1,#self.args.eta,
                                                                    learn_sigma=self.learn_sigma,
                                                                    index=0,
                                                                    hs_coeff = [1 - self.args.hs_coeff],
                                                                    delta_h=delta_hs,
                                                                    use_mask=self.args.use_mask,
                                                                    dt_lambda=self.args.dt_lambda,
                                                                    dt_end = self.args.dt_end,
                                                                    t_edit = self.args.user_defined_t_edit,
                                                                    omega = self.args.omega,
                                                                    )
                else:
                    x, x0_t, _, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                                logvars=self.logvar,
                                                sampling_type=self.args.sample_type,
                                                b=self.betas,
                                                learn_sigma=self.learn_sigma)
                        
                    # if t_it % 91 == 0:
                    #     resize_transform = transforms.Resize(1024)
                    #     x_save = resize_transform((x.detach().cpu() + 1) * 0.5)
                        
                    #     tvu.save_image(x_save, f'./reverse/arnold_arcane_jinx{t_it}.png')
                    
                    # progress_bar.update(1)
                    
                    
            # resize_transform = transforms.Resize(1024)
            # x_save = resize_transform((x.detach().cpu() + 1) * 0.5)
            # tvu.save_image(x_save, f'./reverse2/arnold_arcane_jinx{t_it}.png')

                    
            # x_list = [x_origin, target_img_x0_pairs.detach().cpu(), x.detach().cpu()]
            x_list = [x.detach().cpu()]

            x_list = torch.cat(x_list, dim=0)
            x_list = (x_list + 1) * 0.5
            
            grid = tvu.make_grid(x_list, nrow=x_origin.shape[0], padding=1)            
            resize = transforms.Resize((1024, 1024))
            grid = resize(grid)
            
            tvu.save_image(grid, save_path, normalize=False)
            print(f"Image saved to {save_path}")
            
            end = time.time()
            print(f"Time: {end - start:.2f} seconds")

            # results_list.append(x.detach().cpu())


        # results_list = torch.cat(results_list, dim=0)
        # results_list = (results_list + 1) * 0.5

        # grid_total = tvu.make_grid(results_list, nrow=(len(style_img_paths)+1), padding=1)
        # tvu.save_image(grid_total, os.path.join(self.args.save_dir, 'grid.png'), normalize=False)
        # print("Grid saved to {}".format(os.path.join(self.args.save_dir, 'grid.png')))

        print("done")



    @torch.no_grad()
    def precompute_pairs_with_h(self, model, img_path):


        if not os.path.exists('./precomputed'):
            os.mkdir('./precomputed')

        save_path = "_".join(img_path.split(".")[-2].split('/')[-2:])
        save_path = self.config.data.category + '_inv' + str(self.args.n_inv_step) + '_' + save_path + '.pt'
        save_path = os.path.join('precomputed', save_path)
        
        if not os.path.exists(os.path.sep.join(save_path.split(os.path.sep)[:-1])):
            os.makedirs(os.path.sep.join(save_path.split(os.path.sep)[:-1]))

        n = 1

        print("Precompute multiple h and x_T")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        if os.path.exists(save_path):
            print("Precomputed pairs already exist")
            img_lat_pair = torch.load(save_path)
            return img_lat_pair
        else:
            tmp_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            image = Image.open(img_path).convert('RGB')

            width, height = image.size
            if width > height:
                image = transforms.CenterCrop(height)(image)
            else:
                image = transforms.CenterCrop(width)(image)
            
            image = tmp_transform(image)

            h_dic = {}

            x0 = image.unsqueeze(0).to(self.device)

            x = x0.clone()
            model.eval()
            time_s = time.time()

            with torch.no_grad():
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x, _, _, h = denoising_step(x, t=t, t_next=t_prev, models=model,
                                        logvars=self.logvar,
                                        sampling_type='ddim',
                                        b=self.betas,
                                        eta=0,
                                        learn_sigma=self.learn_sigma,
                                        )
                    # progress_bar.update(1)
                    h_dic[i] = h.detach().clone().cpu()
                    
            #         if "style" in img_path and it % 91 == 0:
            #             resize_transform = transforms.Resize(1024)
            #             x_save = resize_transform((x.detach().cpu() + 1) * 0.5)
            #             tvu.save_image(x_save, f'./forward/arnold_arcane_jinx{it}.png')
                            
                # if "style" in img_path:
                #     resize_transform = transforms.Resize(1024)
                #     x_save = resize_transform((x.detach().cpu() + 1) * 0.5)
                #     tvu.save_image(x_save, f'./forward/arnold_arcane_jinx{it}.png')
                                    

                time_e = time.time()
                # progress_bar.set_description(f"Inversion processing time: {time_e - time_s:.2f}s")
                x_lat = x.clone()
            print("Generative process is skipped")

            img_lat_pairs = [x0, 0 , x_lat.detach().clone().cpu(), h_dic]
            print("Image latent pairs shape: ", len(img_lat_pairs), img_lat_pairs[0].shape, img_lat_pairs[2].shape, len(img_lat_pairs[3]), img_lat_pairs[3][0].shape) 

            torch.save(img_lat_pairs,save_path)
            # print("Precomputed pairs are saved to ", save_path)

            return img_lat_pairs


    # ----------- Pre-compute -----------#
    @torch.no_grad()
    def precompute_pairs(self, model, save_imgs=False):
    
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        for mode in ['train', 'test']:
            img_lat_pairs = []
            if self.config.data.dataset == "IMAGENET":
                if self.args.target_class_num is not None:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            else:
                if mode == 'train':
                    pairs_path = os.path.join('precomputed/',
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_train_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_test_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path) and not self.args.re_precompute:
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=torch.device('cpu'))
                if save_imgs:
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                        tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                        tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                    f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                        if step == self.args.n_precomp_img - 1:
                            break
                continue
            else:

                exist_num = 0
                for exist_precompute_num in reversed(range(self.args.n_train_img if mode == 'train' else self.args.n_test_img)):
                    tmp_path = os.path.join('precomputed/',
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{exist_precompute_num}_ninv{self.args.n_inv_step}_pairs.pth')
                    if os.path.exists(tmp_path):
                        print(f'latest {mode} pairs are exist. Continue precomputing...')
                        img_lat_pairs = img_lat_pairs + torch.load(tmp_path, map_location=torch.device('cpu'))
                        exist_num = exist_precompute_num
                        break

                if self.config.data.category == 'CUSTOM':
                    DATASET_PATHS["custom_train"] = self.args.custom_train_dataset_dir
                    DATASET_PATHS["custom_test"] = self.args.custom_test_dataset_dir

                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num)

                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=1,#self.args.bs_train,
                                            num_workers=self.config.data.num_workers, shuffle=self.args.shuffle_train_dataloader)
                loader = loader_dic[mode]

                if self.args.save_process_origin:
                    save_process_folder = os.path.join(self.args.image_folder, f'inversion_process')
                    if not os.path.exists(save_process_folder):
                        os.makedirs(save_process_folder)

            for step, img in enumerate(loader):
                if (mode == "train" and step == self.args.n_train_img) or (mode == "test" and step == self.args.n_test_img):
                    break
                if exist_num != 0:
                    exist_num = exist_num - 1
                    continue
                x0 = img.to(self.config.device)
                if save_imgs:
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, _, _, _ = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=self.learn_sigma,
                                               )
                            progress_bar.update(1)
                    
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                    x_lat = x.clone()
                    if save_imgs:
                        tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                    f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        time_s = time.time()
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, x0t, _, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=self.learn_sigma)
                            progress_bar.update(1)
                            if self.args.save_process_origin:
                                tvu.save_image((x + 1) * 0.5, os.path.join(save_process_folder, f'xt_{step}_{it}_{t[0]}.png'))
                                tvu.save_image((x0t + 1) * 0.5, os.path.join(save_process_folder, f'x0t_{step}_{it}_{t[0]}.png'))
                        time_e = time.time()
                        print(f'{time_e - time_s} seconds')

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                
                if save_imgs:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                            f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                

            img_lat_pairs_dic[mode] = img_lat_pairs
            # pairs_path = os.path.join('precomputed/',
            #                           f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        return img_lat_pairs_dic