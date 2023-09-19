import torch
import os
import random
import numpy as np
from loguru import logger 
from typing import Tuple, List
import triangle as tr
import igl
import matplotlib.tri as mtri
import scipy.sparse as sp
import torch.nn.functional as F

_CACHE = {}
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
                              img_shape: int = 512) -> list:
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
        _CACHE[f"{self.height}-{self.width}"] = coords
        return coords
  
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    #logger.info(f"Load latents from this: {latents_t_path}")
    latents = torch.load(latents_t_path)
    return latents

def register_attention(model, 
                        injection_schedule,
                        handles,
                        targets):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
           
            y = self.head_to_batch_dim(x)
            
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            source_batch_size = int(x.shape[0] // 3)

            # if not is_cross and self.injection_schedule is not None and (
            #         self.t in self.injection_schedule or self.t == 1000):
            if False:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                
                
                # inject unconditional 
                # q, k [batch, 256, 1280]
                
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]
                
                q = self.head_to_batch_dim(q)
                
                k = self.head_to_batch_dim(k)
                
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)
            
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
           
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of

            #sim [batch, 256, 256]
            
            
            
            if self.t in self.injection_schedule:
                
                B = int(sim.shape[0] / 3)
                key = f"{sim.shape[1]}-{sim.shape[2]}"
                if key not in _CACHE:
                    deformer = MeshDeformator(sim.shape[1], 
                                    sim.shape[2],
                                    torch.device("cuda"))
                    deformer.find_weights(handles, targets)
                    coords = deformer.find_deformed_coords()
                else:
                    coords = _CACHE[key]
                
                deformed_sim = F.grid_sample(sim[None, :B], 
                                coords, 
                                mode='bilinear', 
                                align_corners=False)[0]
                
                sim[B:2 * B] = deformed_sim
                # inject conditional
                sim[2 * B:] = deformed_sim

            attn = sim.softmax(dim=-1)
           
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            
            out = self.batch_to_head_dim(out)
            
            return to_out(out)  

        return forward
    
    #res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    res_dict = {1: [1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(model, 
                                    injection_schedule,
                                    handles,
                                    targets):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)