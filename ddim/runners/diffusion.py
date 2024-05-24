import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


from models.diffusion_vgg import DiffusionVGG, get_content_model_vgg
import random
import torchvision.transforms as tfs
from PIL import Image

from losses import id_loss

from minlora import get_lora_state_dict, add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        
            # states = torch.load(os.path.join(self.args.exp, "logs", "celeba_hq.ckpt"))
            # for key in list(states.keys()):
            #     states["module." + key] = states.pop(key)
            # model.load_state_dict(states)
        

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
                
        states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))


    ############################ Projector #############################

    def train_projector(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = DiffusionVGG(config)
        states = torch.load(os.path.join(self.args.exp, "logs", "celeba_hq.ckpt"))
        
        # for key in list(states.keys()):
        #     states["module." + key] = states.pop(key)
        
        model.load_state_dict(states)
        model.setattr_layers()
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        
        content_vgg_model = get_content_model_vgg("block5_conv2")
        content_vgg_model = content_vgg_model.to(self.device)
        content_vgg_model = torch.nn.DataParallel(content_vgg_model)
        content_vgg_model.eval()
        
        optim_param_list = []
        for i in range(1):
            projector = getattr(model.module, f"projector_{i}")
            optim_param_list = optim_param_list + list(projector.parameters())

        optimizer = get_optimizer(self.config, optim_param_list)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        
        start_epoch = 0
        step = 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        
        for p in model.module.parameters():
            p.requires_grad = False
        for i in range(1):
            projector = getattr(model.module, f"projector_{i}")
            for p in projector.parameters():
                p.requires_grad = True
                
                
        if config.model.type == "vgg_id":
            id_loss_func = id_loss.IDLoss().to(self.device).eval()

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, x_vgg) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x_vgg = x_vgg.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas
                
                content_vgg = content_vgg_model(x_vgg)
                content_vgg_detached = content_vgg.detach()
                
                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                
                # if t[0] < self.config.diffusion.projector.t_edit:
                #     loss = loss_registry[config.model.type](model, x, t, e, b)
                # else:
                #     loss = loss_registry[config.model.type](model, x, t, e, b, content_vgg_detached)
                
                loss = loss_registry[config.model.type](model, x, t, e, b, content_vgg_detached)
                    
                # loss = loss_registry[config.model.type](model, id_loss_func, x, t, e, b, self.args.eta, content_vgg_detached)
                

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    
                    projector_states = []
                    for i in range(1):
                        projector = getattr(model.module, f"projector_{i}")
                        projector_states.append(projector.state_dict())
                        
                    torch.save(
                        projector_states,
                        os.path.join(self.args.log_path, "projector_ckpt_{}.pth".format(step)),
                    )
                    torch.save(projector_states, os.path.join(self.args.log_path, "projector_ckpt.pth"))

                data_start = time.time()
                
        states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        
        
        projector_states = []
        for i in range(1):
            projector = getattr(model.module, f"projector_{i}")
            projector_states.append(projector.state_dict())
            
        torch.save(
            projector_states,
            os.path.join(self.args.log_path, "projector_ckpt_{}.pth".format(step)),
        )
        torch.save(projector_states, os.path.join(self.args.log_path, "projector_ckpt.pth"))
        
    
    def sample_projector(self):
        model = DiffusionVGG(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                print("Loading checkpoint {}".format(os.path.join(self.args.log_path, "ckpt.pth")))
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            
            content_vgg_model = get_content_model_vgg("block5_conv2")
            content_vgg_model = content_vgg_model.to(self.device)
            content_vgg_model = torch.nn.DataParallel(content_vgg_model)
            content_vgg_model.eval()    
                
                
            model.setattr_layers()    
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)
        else:
            raise ValueError

        model.eval()
        
        _, test_dataset = get_dataset(self.args, self.config)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        
        for i, (x, x_vgg) in tqdm.tqdm(enumerate(test_loader), desc="Generating image samples for FID evaluation"):
            x = x.to(self.device)
            x_vgg = x_vgg.to(self.device)
            
            x0 = x.clone()
            x = data_transform(self.config, x)
            content_vgg = content_vgg_model(x_vgg)
            
            y = self.resample_image(x, model, content_vgg)
            y = inverse_data_transform(self.config, y)

            for j in range(self.config.sampling.batch_size):
                tvu.save_image(
                    y[j], os.path.join(self.args.image_folder, f"pred_{i * self.config.sampling.batch_size + j}_projector_slerp_id_mse_small.png")
                )   
                
                tvu.save_image(
                    x0[j], os.path.join(self.args.image_folder, f"orig_{i * self.config.sampling.batch_size + j}.png")
                )
                
            break            
                
                
                    
    def resample_image(self, x, model, content_vgg=None, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
            
        from functions.denoising import generalized_steps_vgg, reverse_generalized_steps, generalized_steps_vgg_diffstyle

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            else:
                raise NotImplementedError
            
            xs = reverse_generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs[0][-1]
            x = x.to(self.device)
            # xs = generalized_steps_vgg(x, seq, model, self.betas, eta=self.args.eta, content_vgg=content_vgg, 
            #                         t_edit=self.config.diffusion.projector.t_edit,
            #                         t_boost=self.config.diffusion.projector.t_boost)
            xs = generalized_steps_vgg_diffstyle(x, seq, model, self.betas, content_vgg=content_vgg, 
                                                t_edit=self.config.diffusion.projector.t_edit,
                                                t_boost=self.config.diffusion.projector.t_boost,
                                                omega=self.config.diffusion.projector.omega)
            x = xs
            
        elif self.args.sample_type == "ddpm_noisy":
            raise NotImplementedError
        
        if last:
            x = x[0][-1]
            
        return x
    
    
    def diffstyle_vgg(self):
        random.seed(42)
        model = DiffusionVGG(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                print("Loading checkpoint {}".format(os.path.join(self.args.log_path, "ckpt.pth")))
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            
            content_vgg_model = get_content_model_vgg("block5_conv2")
            content_vgg_model = content_vgg_model.to(self.device)
            content_vgg_model = torch.nn.DataParallel(content_vgg_model)
            content_vgg_model.eval()    
                
                
            model.setattr_layers()    
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)
        else:
            raise ValueError

        model.eval()
        
        if not os.path.exists(self.args.output):
            os.makedirs(self.args.output)
        
        if os.path.isfile(self.args.content):
            content_img_paths = [self.args.content]
        else:
            content_img_paths = [os.path.join(self.args.content, f) for f in sorted(os.listdir(self.args.content)) if os.path.isfile(os.path.join(self.args.content, f)) and not os.path.isdir(os.path.join(self.args.content, f))]
            
        if os.path.isfile(self.args.style):
            style_img_paths = [self.args.style]
        else:
            style_img_paths = [os.path.join(self.args.style, f) for f in os.listdir(self.args.style) if os.path.isfile(os.path.join(self.args.style, f)) and not os.path.isdir(os.path.join(self.args.style, f))]

        random.shuffle(style_img_paths)
        if len(content_img_paths) > len(style_img_paths):
            style_img_paths = style_img_paths * (len(content_img_paths) // len(style_img_paths) + 1)
        style_img_paths = style_img_paths[:len(content_img_paths)]
        
        
        ddim_transforms = tfs.Compose([
            tfs.Resize((256, 256)),
            tfs.ToTensor(),
        ])
        vgg_transform = tfs.Compose([
            tfs.Resize((224, 224)),
            tfs.ToTensor()
        ])
        
        
        for i, (content_img_path, style_img_path) in tqdm.tqdm(enumerate(zip(content_img_paths, style_img_paths)), desc="Generating image samples for DiffStyle VGG evaluation"):
            content_img = Image.open(content_img_path)
            style_img = Image.open(style_img_path)
            
            # x_content = ddim_transforms(content_img).unsqueeze(0).to(self.device)
            x_style = ddim_transforms(style_img).unsqueeze(0).to(self.device)
            x_content_vgg = vgg_transform(content_img).unsqueeze(0).to(self.device)
            
            # x_content = data_transform(self.config, x_content)
            x_style = data_transform(self.config, x_style)
            
            # if self.args.ood:
            #     x_style = self.reconstruct_image(x_style, model)
            #     x_style = inverse_data_transform(self.config, x_style)
            #     tvu.save_image(tfs.Resize((1024, 1024))(x_style[0]), os.path.join(self.args.output, style_img_path.split(os.path.sep)[-1].split('.')[0] + '.png'))
                
            #     x_style = ddim_transforms(style_img).unsqueeze(0).to(self.device)
            #     x_style = data_transform(self.config, x_style)            

            content_vgg = content_vgg_model(x_content_vgg)
            
            y = self.resample_image(x_style, model, content_vgg)
            y = inverse_data_transform(self.config, y)
            
            save_path = content_img_path.split(os.path.sep)[-1].split('.')[0] + "+" + style_img_path.split(os.path.sep)[-1].split('.')[0] + '.png'
            save_path = os.path.join(self.args.output, save_path)
            
            tvu.save_image(tfs.Resize((1024, 1024))(y[0]), save_path)
            
            
            
    def reconstruct_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
            
        from functions.denoising import generalized_steps_vgg, reverse_generalized_steps, generalized_steps_vgg_diffstyle

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            else:
                raise NotImplementedError
            
            from functions.denoising import generalized_steps

            
            xs = reverse_generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs[0][-1]
            x = x.to(self.device)
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
            
        elif self.args.sample_type == "ddpm_noisy":
            raise NotImplementedError
        
        if last:
            x = x[0][-1]
            
        return x
            
            
            
    ############## LORA ############
    def train_projector_lora(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        
        model = DiffusionVGG(config)
        states = torch.load(os.path.join(self.args.exp, "logs", "celeba_hq.ckpt"))
        model.load_state_dict(states)
        
        for p in model.parameters():
            p.requires_grad = False
        
        add_lora(model)
        lora_parameters = list(get_lora_params(model))

        model.setattr_layers()
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        
        content_vgg_model = get_content_model_vgg("block5_conv2")
        content_vgg_model = content_vgg_model.to(self.device)
        content_vgg_model = torch.nn.DataParallel(content_vgg_model)
        content_vgg_model.eval()
        
        optim_param_list = []
        for i in range(1):
            projector = getattr(model.module, f"projector_{i}")
            optim_param_list = optim_param_list + list(projector.parameters())
            
        optim_param_list = optim_param_list + lora_parameters

        optimizer = get_optimizer(self.config, optim_param_list)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        
        start_epoch = 0
        step = 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        
        # for p in model.module.parameters():
        #     p.requires_grad = False
        for i in range(1):
            projector = getattr(model.module, f"projector_{i}")
            for p in projector.parameters():
                p.requires_grad = True
                
                
        if config.model.type == "vgg_id":
            id_loss_func = id_loss.IDLoss().to(self.device).eval()

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, x_vgg) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x_vgg = x_vgg.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas
                
                content_vgg = content_vgg_model(x_vgg)
                content_vgg_detached = content_vgg.detach()
                
                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                loss = loss_registry[config.model.type](model, x, t, e, b, content_vgg_detached)                

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                        get_lora_state_dict(model)
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    
                    projector_states = []
                    for i in range(1):
                        projector = getattr(model.module, f"projector_{i}")
                        projector_states.append(projector.state_dict())
                        
                    torch.save(
                        projector_states,
                        os.path.join(self.args.log_path, "projector_ckpt_{}.pth".format(step)),
                    )
                    torch.save(projector_states, os.path.join(self.args.log_path, "projector_ckpt.pth"))

                data_start = time.time()
                
        states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
                get_lora_state_dict(model)
            ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                
        projector_states = []
        for i in range(1):
            projector = getattr(model.module, f"projector_{i}")
            projector_states.append(projector.state_dict())
            
        torch.save(
            projector_states,
            os.path.join(self.args.log_path, "projector_ckpt_{}.pth".format(step)),
        )
        torch.save(projector_states, os.path.join(self.args.log_path, "projector_ckpt.pth"))        
            
        
    def sample_projector_lora(self):
        model = DiffusionVGG(self.config)
        states = torch.load(os.path.join(self.args.exp, "logs", "celeba_hq.ckpt"))
        model.load_state_dict(states)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                print("Loading checkpoint {}".format(os.path.join(self.args.log_path, "ckpt.pth")))
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
                projector_states = torch.load(
                    os.path.join(self.args.log_path, "projector_ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
                projector_states = torch.load(
                    os.path.join(self.args.log_path, f"projector_ckpt_{self.config.sampling.ckpt_id}.pth"),
                    map_location=self.config.device,
                )
            
            content_vgg_model = get_content_model_vgg("block5_conv2")
            content_vgg_model = content_vgg_model.to(self.device)
            content_vgg_model = torch.nn.DataParallel(content_vgg_model)
            content_vgg_model.eval()    
                
            add_lora(model)
            _ = model.load_state_dict(states[-1], strict=False)
            merge_lora(model)
                
            model.setattr_layers()
            for i in range(1):
                projector = getattr(model, f"projector_{i}")
                projector.load_state_dict(projector_states[i])
                    
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            # model.load_state_dict(states[0], strict=True)
        else:
            raise ValueError

        model.eval()
        
        _, test_dataset = get_dataset(self.args, self.config)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        
        for i, (x, x_vgg) in tqdm.tqdm(enumerate(test_loader), desc="Generating image samples for FID evaluation"):
            x = x.to(self.device)
            x_vgg = x_vgg.to(self.device)
            
            x0 = x.clone()
            x = data_transform(self.config, x)
            content_vgg = content_vgg_model(x_vgg)
            
            y = self.resample_image(x, model, content_vgg)
            y = inverse_data_transform(self.config, y)

            for j in range(self.config.sampling.batch_size):
                tvu.save_image(
                    y[j], os.path.join(self.args.image_folder, f"pred_{i * self.config.sampling.batch_size + j}_projector_slerp_lora_100kW.png")
                )   
                
                tvu.save_image(
                    x0[j], os.path.join(self.args.image_folder, f"orig_{i * self.config.sampling.batch_size + j}.png")
                )
                
            break    
            
    ############## LORA ############ 

    ############################ Projector #############################
                


    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
            
            # states = torch.load(os.path.join(self.args.exp, "logs", "celeba_hq.ckpt"))
        
            # for key in list(states.keys()):
            #     states["module." + key] = states.pop(key)
            
            # model.load_state_dict(states)

        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 1000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))



    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
