import os, sys
import random
import shutil
import logging
from dataclasses import dataclass, field
import numpy as np 
import matplotlib.pyplot as plt 
import triangle as tr
import igl
import matplotlib.tri as mtri
import scipy.sparse as sp
from typing import Any, Callable, Dict, List, Optional, Union
import torchvision.utils as vutils
from sklearn.decomposition import PCA
from einops import rearrange

import torch
import torch.nn.functional as F
from torchvision import transforms
from contextlib import contextmanager
import PIL
from PIL import Image, ImageDraw
from math import sqrt
from skimage.transform import resize

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    SchedulerMixin,
    EulerDiscreteScheduler,
    DDPMPipeline,
)
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    randn_tensor,
    replace_example_docstring,
)

from transformers import AutoTokenizer, CLIPTextModel
from loguru import logger as logger 
from diffusers.utils import randn_tensor
from tqdm import tqdm 
import argparse
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default="leaning tower of pisa on a field")
parser.add_argument('--visualization_at_t', type=float, default=0.3)
args = parser.parse_args()

class StableDiffusion:
    @dataclass
    class Config:
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        seed: int = 0
        num_inference_steps: int = 50
        infer_batch_size: int = 1
        half_precision_weights: bool = False 
        height: int = 512 
        width: int = 512

    cfg: Config 
    def __init__(self):
        self.cfg = self.Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        self.device = torch.device("cuda")
        self.pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                torch_dtype=self.weights_dtype, 
            ).to(self.device)
        
        self.generator = torch.Generator(device=torch.device("cpu")).manual_seed(self.cfg.seed)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
    @property
    def unet(self):
        return self.pipe.unet
    
    @property
    def scheduler(self):
        return self.pipe.scheduler
    
    @property
    def vae(self):
        return self.pipe.vae 
    
    @property
    def tokenizer(self):
        return self.pipe.tokenizer
    
    @property
    def text_encoder(self):
        return self.pipe.text_encoder
    
    @torch.no_grad()
    def _sample(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np.array",
        return_dict: bool = False,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        ):
        
        # 0. Default height and width to unet
        device = self.device
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_inference_steps = num_inference_steps or self.cfg.num_inference_steps
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.visualization_at_t = \
                self.scheduler.timesteps[int((1 - args.visualization_at_t) * self.cfg.num_inference_steps)]
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.register_attention()
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                self.register_time(t)
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image  = torch.clamp(image / 2 + 0.5, 0., 1.)
        grid_image = vutils.make_grid(image, nrow=2)
        grid_image = grid_image.permute([1, 2, 0]).detach().cpu().numpy()
        
        return grid_image

    def __call__(self,
            prompt):
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        img = self._sample(prompt,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    num_inference_steps=self.cfg.num_inference_steps)
        return img 

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        ):
        
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
           
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

           
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def register_time(self, t):
        conv_module = self.unet.up_blocks[1].resnets[1]
        
        setattr(conv_module, 't', t)
        down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
        up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
        for res in up_res_dict:
            for block in up_res_dict[res]:
                module = self.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                setattr(module, 't', t)

        for res in down_res_dict:
            for block in down_res_dict[res]:
                module = self.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
                setattr(module, 't', t)
        module = self.unet.mid_block.attentions[0].transformer_blocks[0].attn1
        setattr(module, 't', t)


    def register_attention(self, 
                            ):
        def sa_forward(module):
            to_out = module.to_out
            if type(to_out) is torch.nn.modules.container.ModuleList:
                to_out = module.to_out[0]
            else:
                to_out = module.to_out
            
            def forward(x, encoder_hidden_states=None, attention_mask=None):
                batch_size, sequence_length, dim = x.shape
                h = module.heads
    
                is_cross = encoder_hidden_states is not None
                
                y = module.head_to_batch_dim(x)
                
                encoder_hidden_states = encoder_hidden_states if is_cross else x
                
                q = module.to_q(x)
                k = module.to_k(encoder_hidden_states)
                q = module.head_to_batch_dim(q)
                k = module.head_to_batch_dim(k)

                v = module.to_v(encoder_hidden_states)
                v = module.head_to_batch_dim(v)
                
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * module.scale
                # attn [batch, h, w]
                self_attn_map = sim.softmax(dim=-1)
                
                if module.t == self.visualization_at_t:
                    #self_attn_map [B, 256, 256]
                    
                    
                    # rid = vutils.make_grid(out[:int(batch_size / 2), :3], nrow=2, normalize=True, value_range=(0, 1))
                    # vutils.save_image(rid, "image_grid2.png")
                    # exit()
                    self.visualize(self_attn_map[:int(self_attn_map.shape[0] / 2)])
                    
                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~attention_mask, max_neg_value)

                
                
                
                out = torch.einsum("b i j, b j d -> b i d", self_attn_map, v)
                
                out = module.batch_to_head_dim(out)
                
                return to_out(out)  

            return forward
        
        res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        #res_dict = {1: [1]}
        #res_dict = {2: [2]}
        
        for res in res_dict:
            for block in res_dict[res]:
                module = self.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                
                module.forward = sa_forward(module)

    def visualize(self, self_attn_map):
        #feature_maps [b, h * m]
        feature_maps = rearrange(self_attn_map,'h n m -> n (h m)').detach().cpu().numpy()
        
        pca = PCA(n_components=3)
        feature_maps_pca = pca.fit_transform(feature_maps)
        h = w = int(sqrt(feature_maps_pca.shape[0]))
        feature_maps_pca = feature_maps_pca.reshape(h, w, 3)
        feature_maps_pca_min = feature_maps_pca.min(axis=(0, 1))
        feature_maps_pca_max = feature_maps_pca.max(axis=(0, 1))
        feature_maps_pca = (feature_maps_pca - feature_maps_pca_min) / (feature_maps_pca_max - feature_maps_pca_min)
        #feature_maps_pca = resize(feature_maps_pca, (128, 128, 3), order=0, anti_aliasing=False)
        feature_maps_pca = np.clip(feature_maps_pca, 0., 1.0)
        plt.imsave("./test.jpg", feature_maps_pca)
        
@logger.catch
def main():
    model = StableDiffusion()
    images = model([args.prompt] * 1)
    plt.imsave("./out.jpg", images)
    print(images.shape)



if __name__ == "__main__":
    main()