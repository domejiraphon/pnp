import os
import sys
import random
import shutil
from loguru import logger 
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from math import sqrt

import numpy as np
import scipy.sparse as sp
import triangle as tr
import igl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import re 

import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image, ImageDraw
from contextlib import contextmanager

from einops import rearrange
from sklearn.decomposition import PCA

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    SchedulerMixin,
    EulerDiscreteScheduler,
    DDPMPipeline,
)

from transformers import AutoTokenizer, CLIPTextModel

from tqdm import tqdm 
import argparse

# Configure the logger with a custom format
logger.remove()  # Remove the default handler

# Add a new handler with green-colored log text
green_color_start = "\033[92m"
green_color_end = "\033[0m"
logger.add(
    sys.stdout,
    format=f"{green_color_start}{{message}}{green_color_end}",
    level="INFO",
)

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default="leaning tower of pisa on moon")
parser.add_argument('--layer', type=str, default="block1-attn2-trans1")
parser.add_argument('--output_dir', type=str, default="./results")
parser.add_argument('--visualization_at_t', type=float, default=0.3)
parser.add_argument('--visualization_every', type=int, default=-1)
parser.add_argument('--deformed', action='store_true')
args = parser.parse_args()

class MeshDeformator:
    def __init__(self, 
                height: int, 
                width: int,
                device: str = torch.device("cuda")) -> None:
        self.height = height
        self.width = width
        self.device = device
        self.build_mesh_from_image()
        
        
    def build_mesh_from_image(self) -> None:
        logger.info("Build mesh for an image")
        points = np.array([(i, j) for i in range(self.height) 
                            for j in range(self.width)])
        
        # Create the delaunay mesh
        delaunay = tr.triangulate({'vertices': points}, 'q')
        self.vertices = delaunay['vertices'].astype(np.float32)
        self.faces = delaunay['triangles']
        

    def transform_coordinates(self, 
                              coordinates: list,  
                              img_shape: int = 1024) -> list:
        return [(int(x / img_shape * self.height), 
                    int(y / img_shape * self.width)) for x, y in coordinates]

    def find_weights(self, 
                    handles: List[Tuple[int, int]],
                    targets: List[Tuple[int, int]],) -> np.ndarray:
        logger.info("Compute weights!")
        top_boundary = [(0, j) for j in range(self.width)]
        bottom_boundary = [(self.height-1, j) 
                            for j in range(self.width)]
        left_boundary = [(i, 0) for i in range(1, self.height-1)]
        right_boundary = [(i, self.width-1) 
                            for i in range(1, self.height-1)]
        self.handles = self.transform_coordinates(handles) + top_boundary + bottom_boundary + left_boundary + right_boundary
        self.targets = self.transform_coordinates(targets) + top_boundary + bottom_boundary + left_boundary + right_boundary
        
        idx = [y * self.width + x for (y, x) in self.handles]
        b = np.array(idx).astype(self.faces.dtype)
        bc = np.eye(len(self.handles)).astype(self.vertices.dtype)
        
        weights = igl.harmonic(self.vertices, self.faces, b, bc, 2)
        
        self.weights = torch.from_numpy(weights).to(self.device).float()
        
    def find_deformed_coords(self) -> torch.Tensor:
        delta_C = - np.array(self.targets) + np.array(self.handles)
        delta_C = torch.from_numpy(delta_C).to(self.device).float() 
        
        torch_vertices = torch.from_numpy(self.vertices).to(self.device).float()
        vertices_deformed = torch_vertices + (self.weights) @ delta_C
        coords = vertices_deformed.reshape(self.height, 
                                            self.width, 
                                            2)[None]
        coords /= torch.tensor([self.height - 1, 
                            self.width - 1], 
                        dtype=torch.float32,
                        device=self.device)
        coords = 2 * coords - 1
        coords = torch.flip(coords, [-1])
        
        return coords
  
class StableDiffusion:
    @dataclass
    class Config:
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        seed: int = 1
        num_inference_steps: int = 50
        infer_batch_size: int = 1
        half_precision_weights: bool = False 
        height: int = 1024
        width: int = 1024

    cfg: Config 
    def __init__(self):
        self.cfg = self.Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        self.device = torch.device("cuda")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                torch_dtype=self.weights_dtype, 
            ).to(self.device)
       
        self.self_attn_map_t = []
        self.np_to_torch = transforms.Compose([transforms.ToTensor()])

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
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
            self,
            prompt: Union[str, List[str]] = None,
            handles: List[Tuple[int, int]] = [], 
            targets: List[Tuple[int, int]] = [],
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
        ):
        # 0. Default height and width to unet
        self.default_sample_size = self.unet.config.sample_size
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
       
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        
        

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
       
        # 
        #
        # num_channels_latents = self.unet.config.in_channels
        
       # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        if args.visualization_every > 0:
            self.visualization_at_t = self.scheduler.timesteps[::args.visualization_every]
        else:
            self.visualization_at_t = \
                self.scheduler.timesteps[int((1 - args.visualization_at_t) * self.cfg.num_inference_steps)]
        
        
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
       
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
        self.register_attention(handles, targets)
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            self.register_time(t)
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
           
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
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
            
        img = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
       
        img = (img / 2 + 0.5).clamp(0, 1).float()
        
        grid_self_attn = None 
        if self.self_attn_map_t:
            grid_self_attn = vutils.make_grid(
                torch.stack(self.self_attn_map_t, 0),
                nrow=len(self.self_attn_map_t),
                padding=0,
            )
            
        
        return {"img": img,
               "self_attn_map_t": grid_self_attn}
        

    def __call__(self,
            prompt: str = "",
            handles: List[Tuple[int, int]] = [], 
            targets: List[Tuple[int, int]] = [],):
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        img = self._sample(prompt=prompt,
                    handles=handles,
                    targets=targets,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    num_inference_steps=self.cfg.num_inference_steps,
                    generator=generator)
        return img 

    
    def register_time(self, t):
        
        # conv_module = self.unet.up_blocks[1].resnets[1]
        # setattr(conv_module, 't', t)
        # down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
        
        up_blocks_dict = {
                            0: {
                                0: list(range(10)),
                                1: list(range(10)),
                                2: list(range(10)),
                            },
                            1: {
                                0: list(range(2)),
                                1: list(range(2)),
                                2: list(range(2)),
                            },
                        }

        for block in up_blocks_dict :
            for attent in up_blocks_dict[block]:
                for tran in up_blocks_dict[block][attent]:
                    module = self.unet.up_blocks[block].attentions[attent].transformer_blocks[tran].attn1
                    setattr(module, 't', t)

        # for res in down_res_dict:
        #     for block in down_res_dict[res]:
        #         module = self.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
        #         setattr(module, 't', t)
        # module = self.unet.mid_block.attentions[0].transformer_blocks[0].attn1
        # setattr(module, 't', t)

    def register_attention(self, 
                            handles, 
                            targets,):
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
                
                if module.t in self.visualization_at_t:
                    #self_attn_map [B, 256, 256]
                    
                    
                    # rid = vutils.make_grid(out[:int(batch_size / 2), :3], nrow=2, normalize=True, value_range=(0, 1))
                    # vutils.save_image(rid, "image_grid2.png")
                    # exit()
                    #self.visualize(self_attn_map[:int(self_attn_map.shape[0] / 2)])
                    attn_no_deformed = self.visualize(self_attn_map[:int(self_attn_map.shape[0] / 2)])
                    self.self_attn_map_t.append(attn_no_deformed)
                    if args.deformed:
                        h = w = int(sqrt(self_attn_map.shape[1]))
                        deformer = MeshDeformator(h, w)
                        
                        deformer.find_weights(handles, targets)
                        coords = deformer.find_deformed_coords()
                        # reshaped_attn [B, h, w, 256]
                        
                        reshaped_attn = rearrange(self_attn_map, 'b (h w) d -> b h w d', h=h, w=w)
                        reshaped_attn = reshaped_attn.permute([0, 3, 1, 2])
                        
                        
                        #self_attn_map [B, h * w, h, w] -> [B, h * w, w]
                        
                        self_attn_map = F.grid_sample(reshaped_attn,
                                                coords.repeat([reshaped_attn.shape[0], 1, 1, 1]),
                                                mode='bilinear',
                                                align_corners=False)
                        self_attn_map = rearrange(self_attn_map, 'b d h w-> b (h w) d')
                        
                        after = self.visualize(self_attn_map[:int(self_attn_map.shape[0] / 2)])
                        # b_torch = torch.from_numpy(before).to(self.device).float()[None].permute([0, 3, 1, 2])
                        # after = F.grid_sample(b_torch,
                        #                         coords,
                        #                         mode='bilinear',
                        #                         align_corners=False)
                        
                        # after = after[0].permute([1, 2, 0]).cpu().numpy()
                        out = np.concatenate([before, after], 1)
                    
                    
                    
                    
                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~attention_mask, max_neg_value)

                out = torch.einsum("b i j, b j d -> b i d", self_attn_map, v)
                
                out = module.batch_to_head_dim(out)
                
                return to_out(out)  

            return forward
        
        
        numbers = [int(i) for i in re.findall(r'\d+', args.layer)]
        up_blocks_dict = {numbers[0]: {numbers[1]: [numbers[2]]}}
        
        for block in up_blocks_dict:
            for attent in up_blocks_dict[block]:
                for tran in up_blocks_dict[block][attent]:
                    
                    module = self.unet.up_blocks[block].attentions[attent].transformer_blocks[tran].attn1
                    module.forward = sa_forward(module)
                    setattr(module, 'handles', handles)
                    setattr(module, 'targets', targets)

    def visualize(self, self_attn_map):
        feature_maps = rearrange(self_attn_map, 'h n m -> n (h m)').detach().cpu().numpy()
        
        pca_result = PCA(n_components=3).fit_transform(feature_maps)
        dim = int(sqrt(pca_result.shape[0]))
        feature_maps_pca = pca_result.reshape(dim, dim, 3)
        
        min_vals = feature_maps_pca.min(axis=(0, 1))
        max_vals = feature_maps_pca.max(axis=(0, 1))
        normalized_maps = (feature_maps_pca - min_vals) / (max_vals - min_vals)
        
        torch_maps = self.np_to_torch(np.clip(normalized_maps, 0., 1.0))
        return torch_maps

    @staticmethod
    def save_images(inputs: Dict[str, np.ndarray], 
                    output_dir: str):
        for name, img in inputs.items():
            if img is None: continue
            img = img[0] if len(img.shape) == 4 and img.shape[0] == 1 else img
            if len(img.shape) == 4 and img.shape[0] != 1:
                raise ValueError("The first dimension of a 4-dimensional tensor is not 1.")
            img = img.permute([1, 2, 0]).detach().cpu().numpy()

            # Convert the image data to uint8
            img = (img * 255).clip(0, 255).astype(np.uint8)
            if "self" in name:
                name = name.split('_')[0] + "_" + args.layer
            output_path = os.path.join(output_dir, args.prompt, name + ".jpg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            io.imsave(output_path, img)
        logger.info(f"Images have been saved to {os.path.dirname(output_path)}")

@logger.catch
def main():
    model = StableDiffusion()
    handles = [(135, 255), (135, 475), (750, 280), (730, 530)]
    targets = [(435, 255), (435, 475), (750, 280), (730, 530)]
    out = model([args.prompt] * 1,
                   handles,
                   targets)
    StableDiffusion.save_images(out, args.output_dir)
    



if __name__ == "__main__":
    main()