import torch
import os
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import Canny_Edge_Detection as CED
from typing import List, Optional, Tuple, Union

class ImageProcessor:
    @staticmethod
    def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        # Ensure images are in RGB mode
        images = [image.convert('RGB') for image in images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)  # Shape: (N, H, W, C)
        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        if images.ndim == 3:
            images = images[None, ...]  # Add batch dimension
        images = images.transpose(0, 3, 1, 2)  # Shape: (N, C, H, W)
        images = torch.from_numpy(images)
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = images.cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # Shape: (N, H, W, C)
        images = np.clip(images, 0, 1)  # Ensure values are in [0, 1]
        # Remove multiplication by 255 here
        return images  # Values remain in [0, 1]

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> Union[List[Image.Image], Image.Image]:
        if images.ndim == 3:
            images = images[None, ...]  # Add batch dimension
        # Multiply by 255 only once here
        images = (images * 255).astype(np.uint8)
        # Ensure images have the correct color mode
        pil_images = [Image.fromarray(image, mode='RGB') for image in images]
        if len(pil_images) == 1:
            return pil_images[0]
        else:
            return pil_images

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        print("1")
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        print("2")
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        print("3")
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def return_img(local_latent, pipe):
    local_latent = local_latent.clone().detach()
    local_latent.requires_grad_(False)
    image_local = pipe.decode_latents(local_latent)
    return ImageProcessor.numpy_to_pil(image_local)

def return_img(local_latent, pipe):
    local_latent = local_latent.clone().detach()
    local_latent.requires_grad_(False)
    image_local = pipe.vae.decode(local_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image_local = image_local.detach()
    return pipe.image_processor.postprocess(image_local)[0]

def one_step_sds_orange(image, depth, total_epochs, pipe, view_cut):
    depth = depth.to(pipe.device)
    cur_t = [0.02, 0.98]
    clip = 1
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(pipe.device)
    init_latents = pipe.vae.encode(image_tensor)
    init_latents = retrieve_latents(init_latents)
    init_latents = pipe.vae.config.scaling_factor * init_latents
    init_latents = init_latents.detach().clone().requires_grad_(True)
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.1)  # Choose a learning rate
    for e in range(total_epochs):
        optimizer.zero_grad()
        step_ratio = min(1, e / total_epochs)
        grad = pipe.get_sds_latent(
            f"a photo of the {view_cut} cross section of an orange, detailed",
            image=image_tensor,
            depth_map=depth,
            strength=0.1,
            num_inference_steps=100,
            init_latents=init_latents,
            t_range = cur_t,
            guidance_scale=10,
            step_ratio=None
        )
        grad.clamp(-clip, clip)
        target = (init_latents - grad).detach()
        loss = 0.5 * F.mse_loss(init_latents.float(), target, reduction=' sum') / init_latents.shape[0]
        loss.backward()
        optimizer.step()
    return return_img(init_latents, pipe)


def get_controlnet_sds(pipe, prompt, control_image, init_latents, t_range, guidance_scale=10.0):
    device = pipe.device

    text_input = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = pipe.tokenizer(
        [""], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 2. 随机采样时间步 t
    t_min, t_max = int(t_range[0] * 1000), int(t_range[1] * 1000)
    t = torch.randint(t_min, t_max, (1,), device=device).long()

    noise = torch.randn_like(init_latents)
    latents_noisy = pipe.scheduler.add_noise(init_latents, noise, t)

    latent_model_input = torch.cat([latents_noisy] * 2)
    control_image_input = torch.cat([control_image] * 2)

    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = pipe.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=control_image_input,
            return_dict=False,
        )

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    grad = (noise_pred - noise)
    return grad


def one_step_c_orange(image, image_path, total_epochs, pipe, view_cut):
    cur_t = [0.02, 0.98]
    clip = 1

    # 获取边缘线稿
    canny_image_path = CED.generate_with_controlnet(image_path)
    canny_image = Image.open(canny_image_path)

    control_image = pipe.image_processor.preprocess(canny_image).to(device=pipe.device, dtype=pipe.dtype)
    image_tensor = pipe.image_processor.preprocess(image).to(device=pipe.device, dtype=pipe.dtype)

    # VAE 编码
    with torch.no_grad():
        init_latents = pipe.vae.encode(image_tensor)
        init_latents = retrieve_latents(init_latents)
        init_latents = pipe.vae.config.scaling_factor * init_latents

    init_latents = init_latents.detach().clone().to(torch.float32).requires_grad_(True)

    optimizer = optim.Adam([init_latents], lr=0.1)

    for e in range(total_epochs):
        optimizer.zero_grad()

        grad = get_controlnet_sds(
            pipe=pipe,
            prompt=f"a photo of the {view_cut} cross section of an orange, detailed",
            control_image=control_image,
            init_latents=init_latents.to(pipe.dtype),
            t_range=cur_t,
            guidance_scale=10.0
        )

        grad = grad.to(torch.float32)
        grad.clamp_(-clip, clip)

        target = (init_latents - grad).detach()
        loss = 0.5 * F.mse_loss(init_latents, target, reduction='sum') / init_latents.shape[0]

        loss.backward()
        optimizer.step()

    return return_img(init_latents.to(pipe.dtype), pipe)
