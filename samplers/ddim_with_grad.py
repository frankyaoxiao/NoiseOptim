# Adapted from Bansal's Universal Guided Diffusion https://github.com/arpitbansal297/Universal-Guided-Diffusion
import torch
import numpy as np
from tqdm import tqdm
from torchvision import utils
from util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import os


class DDIMSamplerWithGrad(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.model.module.num_timesteps, verbose=verbose)

        alphas_cumprod = self.model.module.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.module.device)

        self.register_buffer('betas', to_torch(self.model.module.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.module.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               S,
               batch_size,
               shape,
               operated_image=None,
               operation=None,
               conditioning=None,
               eta=0.,
               temperature=1.,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               start_zt=None,
               identifier="",
               ):


        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)
        min_loss = 1e9


        device = self.model.module.betas.device
        b = shape[0]

        img = start_zt = torch.randn(shape, device=device) if start_zt is None else start_zt

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False



        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            b, *_, device = *img.shape, img.device

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            beta_t = a_t / a_prev
            num_steps = operation.num_steps[0]

            operation_func = operation.operation_func
            other_guidance_func = operation.other_guidance_func
            criterion = operation.loss_func
            other_criterion = operation.other_criterion

            apply_guidance = i >= 20 

            for j in range(num_steps):
                if operation.guidance_3 and apply_guidance:  # Modified condition here
                    torch.set_grad_enabled(True)
                    img_in = img.detach().requires_grad_(True)

                    if operation.original_guidance:
                        if len(conditioning) == 3:  # If we have a negative prompt
                            x_in = torch.cat([img_in] * 3)  # Three copies of the input
                            t_in = torch.cat([ts] * 3)
                            e_t_uncond, e_t_neg, e_t = self.model.module.apply_model(x_in, t_in, conditioning).chunk(3)
                            # Modified guidance formula incorporating negative prompt
                            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) - unconditional_guidance_scale * (e_t_neg - e_t_uncond)
                        else:  # Original behavior
                            x_in = torch.cat([img_in] * 2)
                            t_in = torch.cat([ts] * 2)
                            e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, conditioning).chunk(2)
                            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                    else:
                        e_t = self.model.module.apply_model(img_in, ts, conditioning)

                    pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = self.model.module.decode_first_stage_with_grad(pred_x0)
                    # utils.save_image(recons_image, f'{operation.folder}/recons_image_at_{ts[0]}.png')

                    if other_guidance_func != None:
                        op_im = other_guidance_func(recons_image)
                    elif operation_func != None:
                        op_im = operation_func(recons_image)
                    else:
                        op_im = recons_image

                    if op_im is not None:
                        if hasattr(operation_func, 'cal_loss'):
                            loss = operation_func.cal_loss(recons_image, total_steps - i - 1, identifier, operation.folder)
                            selected = -1 * loss.sum()
                            print(f"Step {i}: loss: {loss.sum().item()}")
                            if loss.sum() < min_loss and i > 400:
                                min_loss = loss.sum()
                                normalized_image = torch.clamp((recons_image + 1.0) / 2.0, min=0.0, max=1.0)
                                utils.save_image(normalized_image, f'{operation.folder}/min_loss_image_{identifier}.png')
                                # Log the minimum loss
                                log_path = os.path.join(operation.folder, 'generation_log.txt')
                                with open(log_path, 'a') as f:
                                    f.write(f"Image {identifier} - Minimum Loss: {min_loss.item():.6f} at step {i}\n")
                        elif other_criterion != None:
                            selected = -1 * other_criterion(op_im, operated_image)
                            print(f"Step {i}: other_criterion was used")
                        else:
                            selected = -1 * criterion(op_im, operated_image)
                            print(f"Step {i}: default criterion was used")

                        # print(ts)
                        # print(selected)

                        grad = torch.autograd.grad(selected.sum(), img_in)[0]
                        grad = grad * operation.optim_guidance_3_wt

                        e_t = e_t - sqrt_one_minus_at * grad.detach()

                        img_in = img_in.requires_grad_(False)

                        if operation.print:
                            if j == 0:
                                temp = (recons_image + 1) * 0.5
                                utils.save_image(temp, f'{operation.folder}/img_at_{ts[0]}.png')

                        del img_in, pred_x0, recons_image, op_im, selected, grad
                        if operation.original_guidance:
                            del x_in

                    else:
                        e_t = e_t
                        img_in = img_in.requires_grad_(False)

                        if operation.print:
                            if j == 0:
                                temp = (recons_image + 1) * 0.5
                                utils.save_image(temp, f'{operation.folder}/img_at_{ts[0]}.png')

                        del img_in, pred_x0, recons_image, op_im
                        if operation.original_guidance:
                            del x_in


                    torch.set_grad_enabled(False)

                else:
                    if operation.original_guidance:
                        if len(conditioning) == 3:  # If we have a negative prompt
                            x_in = torch.cat([img] * 3)  # Three copies of the input
                            t_in = torch.cat([ts] * 3)
                            e_t_uncond, e_t_neg, e_t = self.model.module.apply_model(x_in, t_in, conditioning).chunk(3)
                            # Modified guidance formula incorporating negative prompt
                            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) - unconditional_guidance_scale * (e_t_neg - e_t_uncond)
                        else:  # Original behavior
                            x_in = torch.cat([img] * 2)
                            t_in = torch.cat([ts] * 2)
                            e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, conditioning).chunk(2)
                            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                    else:
                        e_t = self.model.module.apply_model(img, ts, conditioning)

                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device, False) * temperature

                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    img = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(img.shape, device, False)

                    del pred_x0, dir_xt, noise

            img = x_prev


        return img, start_zt


