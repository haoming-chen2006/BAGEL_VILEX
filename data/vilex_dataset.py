import torch
import torch.nn.functional as F
import torchvision.transforms as T
import webdataset as wds
import random
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from data.data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    add_special_tokens
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
from transformers import HfArgumentParser, set_seed
from data.prompts import variations
# Todos:
# 1. Multiple queries, multiple batches
# 2. load the "sampled instruciton text" -- generate an image of
# 3. check the generation code to be correct and use it
# 4. have right training loop so output one text only, one vilex only reconsturciton, and one with both

padding = False


from train.dataconf import (DataArguments,ModelArguments,TrainingArguments)


parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


class VilexDataset(torch.utils.data.IterableDataset):
    """
    Loads data and set basic configs
    #todo: make config load from yaml file so more convertable
    use create_webdataset to load samples, and handover to process sample ot do preprocessing
    """
    
    def __init__(
        self,
        shards,
        tokenizer,
        special_tokens,
        vae_model=None,
        max_image_size=512,  # Changed from image_size to max_image_size
        vae_image_downsample=16,
        vit_patch_size=14,
        max_sequence_length=4096,
        shuffle_buffer_size=1000,
        tail_drop_prob = model_args.tail_drop_prob,
        tail_drop_max = model_args.tail_drop_max,
    ):
        self.tokenizer = tokenizer
        self.vae_model = vae_model
        self.max_image_size = max_image_size
        self.vae_image_downsample = vae_image_downsample
        self.vit_patch_size = vit_patch_size
        self.max_sequence_length = max_sequence_length
        self.tail_drop_max = tail_drop_max
        self.tail_drop_prob = tail_drop_prob
        
        # Set special tokens as attributes
        for k, v in special_tokens.items():
            setattr(self, k, v)
            
        # Calculate actual image size (divisible by patch_size) -- originally just resizes to vae or vit, now resize to the lcm of both
        lcm_size = math.lcm(vit_patch_size, vae_image_downsample)
        self.actual_image_size = (max_image_size // lcm_size) * lcm_size
        print(f"Adjusted image size from {max_image_size} to {self.actual_image_size} (divisible by LCM({vit_patch_size}, {vae_image_downsample}) = {lcm_size})")
        
        # Image transform with dynamic sizing
        self.transform = T.Compose([
            T.Resize(self.actual_image_size),
            T.CenterCrop(self.actual_image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
        
        self.dataset = self._create_webdataset(shards, shuffle_buffer_size)
    
    def _create_webdataset(self, shards, shuffle_buffer_size):
        if shuffle_buffer_size > 0:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(shards),
                wds.shuffle(100),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(shuffle_buffer_size),
                wds.decode("pil"),
                wds.to_tuple("jpg", "txt"),
                wds.map(self._process_sample),
            )
        else:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(shards),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("pil"),
                wds.to_tuple("jpg", "txt"),
                wds.map(self._process_sample),
            )
        return dataset
    
    def _process_sample(self, sample):
        """Process single sample into BAGEL format with fixed sequence
        tokenize text, pachify vit, and keep vae the same
        pass the components and calculated num tokens into pack simple sequence for packing things together
        """
        image, text = sample

        # pick a random 'variation prompt' from the variations
        import random
        choice = random.randint(0,len(variations)-1)
        # for now use same length text token --  so no padding involved
        text = variations[choice]
        print(f"selected {text}")

        
        
        # Preprocess
        image_tensor = self.transform(image)
        text_tokens = self.tokenizer.encode(text.strip())

        target_length = 20
        text_cut = 0  # Initialize text_cut to 0 for both cases
        
        # padding text tokens to the current max length -- 20
        if padding:
            if len(text_tokens) < target_length:
                text_cut = target_length - len(text_tokens)
                text_tokens.extend([self.tokenizer.pad_token_id for _ in range(text_cut)])
        
        
        # Patchify for ViT
        from data.data_utils import patchify
        vit_patches = patchify(image_tensor, self.vit_patch_size)
        num_vit_patches = vit_patches.shape[0]
        
        # VAE dimensions  
        vae_h = image_tensor.shape[1] // self.vae_image_downsample
        vae_w = image_tensor.shape[2] // self.vae_image_downsample
        num_vae_tokens = vae_h * vae_w
        
        # FIXED SEQUENCE: text + vit + vae
        return self._pack_sequence(
            text_tokens, image_tensor, vit_patches, 
            vae_h, vae_w, num_vit_patches, num_vae_tokens, 32, text_cut # numqueries for testing, automatic 1, maybe use 32 at inference time 
        )
    
    def _pack_sequence(self, text_tokens, image_tensor, vit_patches, 
                            vae_h, vae_w, num_vit_patches, num_vae_tokens,num_queries,text_cut):
        """Pack sequence in fixed order: text + vit + vae
        also includes meta data of the indexes (where these tokens appear in the sequence)
        length of sequence, position ids of different indexes, loss indexes (which part is got to compute which loss)
        and finally mse and ce loss weights
        #to do: look into controlling attn_modes and positional ids
        """
        
        # Initialize all arrays
        packed_text_ids = []
        packed_text_indexes = []
        packed_position_ids = []
        packed_vit_tokens = []
        packed_vit_position_ids = []
        packed_vit_token_indexes = []
        packed_vae_token_indexes = []
        packed_timesteps = []
        mse_loss_indexes = []
        packed_label_ids = []
        ce_loss_indexes = []
        ce_loss_weights = []
        tail_drop_prob = self.tail_drop_prob
        tail_drop_max = self.tail_drop_max
        
        curr = 0  # Absolute position tracker
        rope_id = 0  # RoPE position tracker
        split_lens = []
        attn_modes = []
        
        # 1. TEXT BLOCK
        text_split_len = 0
        
        # Add BOS + text tokens + Eos
        shifted_text = [self.bos_token_id] + text_tokens
        packed_text_ids.extend(shifted_text)
        packed_text_indexes.extend(range(curr, curr + len(shifted_text)))
        
        # Text loss
        ce_loss_indexes.extend(range(curr, curr + len(shifted_text)))
        weight_value = len2weight(len(shifted_text))
        if padding and text_cut > 0:
            # Only apply zero weights for padding tokens when padding is enabled
            ce_loss_weights.extend([weight_value] * (len(shifted_text) - text_cut) + [0.0] * text_cut)
        else:
            # No padding or padding disabled - all tokens get weight
            ce_loss_weights.extend([weight_value] * len(shifted_text))
        
        curr += len(shifted_text)
        text_split_len += len(shifted_text)
        packed_label_ids.extend([shifted_text])
        
        # Text uses sequential RoPE positions
        packed_position_ids.extend(range(rope_id, rope_id + text_split_len))
        rope_id += text_split_len
        attn_modes.append("causal")
        
        # 2. VIT BLOCK  
        vit_split_len = 0
        
        # Start of image -  not adding image tokens to vit for now
        # packed_text_ids.append(self.start_of_image)
        # packed_text_indexes.append(curr)
        # curr += 1
        # vit_split_len += 1
        
        # ViT tokens -- calculate and apply taildorp right here!
        k = 0
        if tail_drop_prob > 0 and tail_drop_max > 0 and num_queries > 1: # turn tail drop off for  now
            k = random.randint(0, tail_drop_max) if random.random() < tail_drop_prob else 0
            if k > 0:
                num_queries -=k



        packed_vit_token_indexes.extend(range(curr, curr + num_queries))
        packed_vit_tokens.append(vit_patches)
        curr += num_queries
        vit_split_len += num_queries

        packed_text_ids.extend([self.eos_token_id])
        packed_text_indexes.append(curr)

        curr += 1
        packed_position_ids.extend([rope_id])
        rope_id += 1
        text_split_len +=1
        

        # ViT position IDs (2D flattened)
        h, w = image_tensor.shape[1:]
        vit_pos_ids = get_flattened_position_ids_extrapolate(
            h, w, self.vit_patch_size, max_num_patches_per_side = 70
        )
        packed_vit_position_ids.extend([vit_pos_ids])

        packed_text_ids.append(self.start_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        packed_position_ids.extend([rope_id])
        rope_id += 1
        text_split_len +=1
        
        # End of image -- currently not applied
        # packed_text_ids.append(self.end_of_image)

        # packed_text_indexes.append(curr)
        # curr += 1
        # vit_split_len += 1
        
        # All ViT tokens get same RoPE position as text

        
        packed_position_ids.extend(range(rope_id, rope_id + vit_split_len))
        rope_id += vit_split_len
        split_lens.append(text_split_len)
        split_lens.append(vit_split_len)
        attn_modes.append("causal")
        
        # 3. VAE BLOCK
        vae_split_len = 0
        
        # Start of image
        
        # vae_split_len += 1 -- instead append a small split lens of size 2
        
        # VAE tokens
        packed_vae_token_indexes.extend(range(curr, curr + num_vae_tokens))
        mse_loss_indexes.extend(range(curr, curr + num_vae_tokens))
        
        # Random timestep for diffusion
        timestep = np.random.randint(0, 1000) # no bound?
        packed_timesteps.extend([timestep] * num_vae_tokens)
        
        curr += num_vae_tokens
        vae_split_len += num_vae_tokens
        
        # End of image
        packed_text_ids.append(self.end_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        vae_split_len += 1 
        packed_position_ids.extend([rope_id])
        rope_id += 1

        
        # All VAE tokens get same RoPE position
        packed_latent_position_ids = []
        packed_position_ids.extend([rope_id] * vae_split_len)
        packed_latent_position_ids.append(
                    get_flattened_position_ids_extrapolate(
                        image_tensor.size(1), image_tensor.size(2),
                        self.vae_image_downsample, 
                        max_num_patches_per_side=70
                    )
                )
        rope_id += 1
        split_lens.append(vae_split_len)
        attn_modes.append("noise")  # For diffusion training

        nested_attention_masks = [self._prepare_attention_mask(split_lens, attn_modes)]
        
        # Only apply padding mask modifications when padding is enabled
        if padding and text_cut > 0:
            text_len = len(shifted_text)
            padd_mask_list = [float("-inf")] * len(nested_attention_masks[0][0])
            padd_mask = torch.tensor(padd_mask_list, dtype=nested_attention_masks[0].dtype, device=nested_attention_masks[0].device)
            
            for i in range(text_cut):
                # Padding tokens are at positions: text_len - text_cut, text_len - text_cut + 1, ..., text_len - 1
                padding_token_pos = text_len - text_cut + i
                nested_attention_masks[0][padding_token_pos] = padd_mask

            for i in range(text_len, curr):
                for j in range(text_cut):
                    nested_attention_masks[0][i][text_len - j] = float("-inf")

        # Convert to tensor format
        return {
            'sequence_length': curr,
            'sample_lens': [curr],  # Single sample
            'packed_text_ids': torch.tensor(packed_text_ids),
            'packed_text_indexes': torch.tensor(packed_text_indexes), 
            'packed_position_ids': torch.tensor(packed_position_ids[:-1]),
            'text_cut': text_cut, # the length of the padding tokens (0 when padding=False)
            
            # ViT data
            'packed_vit_tokens': torch.cat([vit_patches], dim=0) if packed_vit_tokens else torch.empty(0),
            'packed_vit_position_ids': torch.cat(packed_vit_position_ids, dim=0) if packed_vit_position_ids else torch.empty(0),
            'packed_vit_token_indexes': torch.tensor(packed_vit_token_indexes),
            'vit_token_seqlens': torch.tensor([num_vit_patches]),
            'packed_latent_position_ids':torch.cat(packed_latent_position_ids, dim=0),
            'k': k,
            
            # VAE data  
            'padded_images': image_tensor.unsqueeze(0),  # Add batch dim
            'patchified_vae_latent_shapes': [(vae_h, vae_w)],
            'packed_vae_token_indexes': torch.tensor(packed_vae_token_indexes),
            'packed_timesteps': torch.tensor(packed_timesteps),
            'mse_loss_indexes': torch.tensor(mse_loss_indexes),
            
            # Text loss data
            'packed_label_ids': torch.tensor(packed_label_ids),
            'ce_loss_indexes': torch.tensor(ce_loss_indexes), 
            'ce_loss_weights': torch.tensor(ce_loss_weights),
            
            # Attention
            "num_tokens": curr,
            'split_lens': split_lens,
            'attn_modes': attn_modes,
            'nested_attention_masks': nested_attention_masks,
            
            # Metadata
            'batch_data_indexes': [{'data_indexes': [0, 0, 0], 'worker_id': 0, 'dataset_name': 'simple'}],
            'data_indexes': {'data_indexes': [0, 0, 0], 'worker_id': 0, 'dataset_name': 'simple'},   
        } 
    
    def _prepare_attention_mask(self, split_lens, attn_modes):
        """Create attention mask for the sequence"""
        from data.data_utils import prepare_attention_mask_per_sample
        return prepare_attention_mask_per_sample(split_lens, attn_modes)
    
    def __iter__(self):
        for sample in self.dataset:
            if sample['num_tokens'] <= self.max_sequence_length:
                print("yielding sample with length" + str(sample["num_tokens"]))
                yield sample
                


def create_loader(
    shards,
    tokenizer, 
    special_tokens,
    vae_model=None,
    batch_size=1,
    num_workers=1,
    **kwargs
):
    """Create simplified BAGEL-compatible dataloader"""
    
    dataset = VilexDataset(
        shards=shards,
        tokenizer=tokenizer,
        special_tokens=special_tokens, 
        vae_model=vae_model,
        **kwargs
    )
    
    def collate_fn(batch):
        """Custom collate function for BAGEL compatibility"""
        # If padding is disabled, return single sample without batch dimension
        if not padding:
            return batch[0]
        
        # With padding enabled, stack tensors to create batch dimension
        batch_dict = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                batch_dict[key] = torch.stack([b[key] for b in batch])
            else:
                batch_dict[key] = [b[key] for b in batch]
        return batch_dict
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    from collections import Counter


def get_loader(split,tokenizer = None,special_tokens = None,vae_model = None, cfg_path = "/home/haoming/Bagel/data/configs/datacomp.yaml"):


    if tokenizer == None:
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    if special_tokens == None:
        special_tokens = new_token_ids

    if vae_model == None:
        vae_model, vae_config = load_ae("/home/haoming/Bagel/models/BAGEL-7B-MoT/ae.safetensors")
        vae_model.eval()

    dataset_cfg = OmegaConf.load(cfg_path)

    # Pick the right config block
    if split.lower() == "train":
        split_cfg = dataset_cfg.train
    elif split.lower() == "val":
        split_cfg = dataset_cfg.validation
    else:
        raise ValueError(f"Unknown split: {split}")
    
    return create_loader(shards = split_cfg.shards, tokenizer = tokenizer, special_tokens = special_tokens, vae_model=vae_model)


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_attention_mask(mask, text_indexes, vit_indexes, vae_indexes, 
                         special_tokens=None, title="Attention Mask", save_path=None, batch_idx=0):
    """
    Visualize attention mask for a single batch element (batch_idx)
    """
    # Handle both batched and unbatched cases based on padding setting
    if padding:
        # With padding, we have batch dimensions
        if isinstance(mask, list):
            mask = mask[0]
        if hasattr(mask, 'dim') and mask.dim() == 3:
            mask = mask[batch_idx]
        if hasattr(text_indexes, 'dim') and text_indexes.dim() == 2:
            text_indexes = text_indexes[batch_idx]
        if hasattr(vit_indexes, 'dim') and vit_indexes.dim() == 2:
            vit_indexes = vit_indexes[batch_idx]
        if hasattr(vae_indexes, 'dim') and vae_indexes.dim() == 2:
            vae_indexes = vae_indexes[batch_idx]
    else:
        # Without padding, no batch dimension to handle
        if isinstance(mask, list):
            mask = mask[0]
        # text_indexes, vit_indexes, vae_indexes are already 1D
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sequence_length = mask.shape[0]
    
    # Convert mask to numpy for visualization (0 = white, -inf = black)
    mask_vis = mask.clone()
    mask_vis[mask_vis == float('-inf')] = -1
    mask_vis[mask_vis == 0] = 1
    
    # Create the heatmap
    im = ax.imshow(mask_vis.numpy(), cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    
    # Create token type mapping (0=text, 1=vit, 2=vae, -1=unknown)
    token_types = [-1] * sequence_length
    
    # Map all token positions to their type
    for idx in text_indexes.tolist():
        if 0 <= idx < sequence_length:
            token_types[idx] = 0  # Text
    
    for idx in vit_indexes.tolist():
        if 0 <= idx < sequence_length:
            token_types[idx] = 1  # ViT
    
    for idx in vae_indexes.tolist():
        if 0 <= idx < sequence_length:
            token_types[idx] = 2  # VAE
    
    # Check for any overlap (should not happen)
    text_set = set(text_indexes.tolist())
    vit_set = set(vit_indexes.tolist())
    vae_set = set(vae_indexes.tolist())
    
    overlap_tv = text_set.intersection(vit_set)
    overlap_ta = text_set.intersection(vae_set)
    overlap_va = vit_set.intersection(vae_set)
    
    if overlap_tv or overlap_ta or overlap_va:
        print(f"WARNING: Token index overlap detected!")
        if overlap_tv: print(f"Text-ViT overlap: {overlap_tv}")
        if overlap_ta: print(f"Text-VAE overlap: {overlap_ta}")
        if overlap_va: print(f"ViT-VAE overlap: {overlap_va}")
    
    # Add boundary lines between different token types
    for i in range(1, sequence_length):
        if token_types[i] != token_types[i-1]:
            ax.axvline(x=i-0.5, color='yellow', linewidth=2, alpha=0.8)
            ax.axhline(y=i-0.5, color='yellow', linewidth=2, alpha=0.8)
    
    # Add section labels
    text_positions = sorted(text_indexes.tolist())
    vit_positions = sorted(vit_indexes.tolist())
    vae_positions = sorted(vae_indexes.tolist())
    
    # Add section labels using the median position of each token type
    if text_positions:
        ax.text(np.median(text_positions), -2, 'TEXT', ha='center', fontweight='bold', fontsize=12)
    if vit_positions:
        ax.text(np.median(vit_positions), -2, 'VIT', ha='center', fontweight='bold', fontsize=12)
    if vae_positions:
        ax.text(np.median(vae_positions), -2, 'VAE', ha='center', fontweight='bold', fontsize=12)
    
    # Add tick labels for key positions
    major_ticks = [0]
    major_labels = ['0']
    
    # Add first and last position of each token type
    for positions, name in [(text_positions, 'T'), (vit_positions, 'V'), (vae_positions, 'A')]:
        if positions:
            major_ticks.extend([positions[0], positions[-1]])
            major_labels.extend([f'{positions[0]}{name}', f'{positions[-1]}{name}'])
    
    major_ticks.append(sequence_length-1)
    major_labels.append(f'{sequence_length-1}')
    
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(major_labels)
    
    # Add minor ticks for every 10th position if sequence is long
    if sequence_length > 20:
        minor_ticks = list(range(0, sequence_length, max(1, sequence_length//10)))
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True, which='minor', alpha=0.3)
    
    ax.set_title(f'{title}\nWhite=Attend (0), Blue=Masked (-inf)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['-inf (Masked)', '0 (Attend)'])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f'attention_mask_{batch_idx}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention mask saved to: {save_path}")
    plt.close()
    return save_path


def visualize_loss_sequence(sequence_length, ce_loss_indexes, mse_loss_indexes, 
                        text_indexes, vit_indexes, vae_indexes, 
                        ce_loss_weights=None,  # Add ce_loss_weights parameter
                        title="Loss Index Sequence", save_path=None, batch_idx=0):
    """
    Visualize loss sequence for a single batch element (batch_idx)
    """
    # Handle both batched and unbatched cases based on padding setting
    if padding:
        # With padding, we have batch dimensions
        if isinstance(ce_loss_indexes, list):
            ce_loss_indexes = ce_loss_indexes[0]
        if isinstance(mse_loss_indexes, list):
            mse_loss_indexes = mse_loss_indexes[0]
        if isinstance(text_indexes, list):
            text_indexes = text_indexes[0]
        if isinstance(vit_indexes, list):
            vit_indexes = vit_indexes[0]
        if isinstance(vae_indexes, list):
            vae_indexes = vae_indexes[0]
        
        # Ensure sequence_length is an integer
        if isinstance(sequence_length, list):
            sequence_length = int(sequence_length[batch_idx])
        elif isinstance(sequence_length, torch.Tensor) and sequence_length.dim() > 0:
            sequence_length = int(sequence_length[batch_idx].item())
        
        if hasattr(ce_loss_indexes, 'dim') and ce_loss_indexes.dim() == 2:
            ce_loss_indexes = ce_loss_indexes[batch_idx]
        if hasattr(mse_loss_indexes, 'dim') and mse_loss_indexes.dim() == 2:
            mse_loss_indexes = mse_loss_indexes[batch_idx]
        if hasattr(text_indexes, 'dim') and text_indexes.dim() == 2:
            text_indexes = text_indexes[batch_idx]
        if hasattr(vit_indexes, 'dim') and vit_indexes.dim() == 2:
            vit_indexes = vit_indexes[batch_idx]
        if hasattr(vae_indexes, 'dim') and vae_indexes.dim() == 2:
            vae_indexes = vae_indexes[batch_idx]
        if ce_loss_weights is not None and hasattr(ce_loss_weights, 'dim') and ce_loss_weights.dim() == 2:
            ce_loss_weights = ce_loss_weights[batch_idx]
    else:
        # Without padding, no batch dimension to handle
        if isinstance(ce_loss_indexes, list):
            ce_loss_indexes = ce_loss_indexes[0]
        if isinstance(mse_loss_indexes, list):
            mse_loss_indexes = mse_loss_indexes[0]
        if isinstance(text_indexes, list):
            text_indexes = text_indexes[0]
        if isinstance(vit_indexes, list):
            vit_indexes = vit_indexes[0]
        if isinstance(vae_indexes, list):
            vae_indexes = vae_indexes[0]
        
        # Ensure sequence_length is an integer
        if isinstance(sequence_length, list):
            sequence_length = int(sequence_length[0])
        elif isinstance(sequence_length, torch.Tensor):
            sequence_length = int(sequence_length.item())
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Create base sequence
    sequence = list(range(sequence_length))
    colors = ['lightgray'] * sequence_length  # Default: no loss
    token_types = ['Unknown'] * sequence_length  # Default: unknown token type
    
    # Map all token positions to their type
    for idx in text_indexes.tolist():
        if 0 <= idx < sequence_length:
            token_types[idx] = 'Text'
    
    for idx in vit_indexes.tolist():
        if 0 <= idx < sequence_length:
            token_types[idx] = 'ViT'
    
    for idx in vae_indexes.tolist():
        if 0 <= idx < sequence_length:
            token_types[idx] = 'VAE'
    
    # Background colors for token types
    for i, token_type in enumerate(token_types):
        if token_type == 'Text':
            ax.axvspan(i-0.5, i+0.5, alpha=0.1, color='green')
        elif token_type == 'ViT':
            ax.axvspan(i-0.5, i+0.5, alpha=0.1, color='orange')
        elif token_type == 'VAE':
            ax.axvspan(i-0.5, i+0.5, alpha=0.1, color='purple')
    
    # Mark CE loss positions with non-zero weights (purple)
    if ce_loss_indexes is not None and ce_loss_weights is not None:
        ce_indexes_list = ce_loss_indexes.tolist()
        ce_weights_list = ce_loss_weights.tolist()
        for i, idx in enumerate(ce_indexes_list):
            if idx < sequence_length and ce_weights_list[i] > 0:
                colors[idx] = 'purple'

    # Mark MSE loss positions (red) - may override CE
    if mse_loss_indexes is not None:
        for idx in mse_loss_indexes:
            if idx < sequence_length:
                colors[idx] = 'red'

    # Create bar chart
    bars = ax.bar(sequence, [1]*sequence_length, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add boundary lines between different token types
    for i in range(1, sequence_length):
        if token_types[i] != token_types[i-1]:
            ax.axvline(x=i-0.5, color='yellow', linewidth=2, alpha=0.8)
    
    # Add position numbers for key locations
    text_positions = sorted(text_indexes.tolist())
    vit_positions = sorted(vit_indexes.tolist())
    vae_positions = sorted(vae_indexes.tolist())
    
    key_positions = [0]
    # Add first and last position of each token type
    for positions in [text_positions, vit_positions, vae_positions]:
        if positions:
            key_positions.extend([positions[0], positions[-1]])
    key_positions.append(sequence_length-1)
    
    for pos in sorted(set(key_positions)):
        if pos < sequence_length:
            ax.text(pos, -0.15, str(pos), ha='center', fontweight='bold', fontsize=10)
    
    # Add tick marks
    if sequence_length > 20:
        tick_positions = list(range(0, sequence_length, max(1, sequence_length//20)))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_positions], fontsize=8)
    else:
        ax.set_xticks(sequence)
        ax.set_xticklabels([str(i) for i in sequence], fontsize=8)
    
    # Add section labels at top for contiguous blocks
    def find_blocks(positions):
        if not positions:
            return []
        blocks = []
        current_block = [positions[0]]
        for pos in positions[1:]:
            if pos == current_block[-1] + 1:
                current_block.append(pos)
            else:
                blocks.append(current_block)
                current_block = [pos]
        blocks.append(current_block)
        return blocks
    
    # Add labels for blocks of tokens
    for token_type, positions, color in [
        ('TEXT', text_positions, 'green'),
        ('VIT', vit_positions, 'orange'),
        ('VAE', vae_positions, 'purple')
    ]:
        blocks = find_blocks(positions)
        for block in blocks:
            if len(block) > 2:  # Only label blocks with at least 3 tokens
                ax.text(np.mean(block), 1.1, token_type, ha='center', 
                        fontweight='bold', fontsize=10, color=color)
    
    ax.set_ylim(-0.2, 1.3)
    ax.set_xlim(-0.5, sequence_length - 0.5)
    ax.set_ylabel('Loss Type')
    ax.set_xlabel('Sequence Position')
    ax.set_title(f'{title}\nPurple=CE Loss (Weight>0), Red=MSE Loss, Gray=No Loss', fontsize=14, fontweight='bold')
    
    # Custom legend
    legend_elements = [
        patches.Patch(color='purple', alpha=0.7, label='CE Loss (Text, Weight > 0)'),
        patches.Patch(color='red', alpha=0.7, label='MSE Loss (VAE)'),
        patches.Patch(color='lightgray', alpha=0.7, label='No Loss'),
        patches.Patch(color='green', alpha=0.1, label='Text Tokens'),
        patches.Patch(color='orange', alpha=0.1, label='ViT Tokens'),
        patches.Patch(color='purple', alpha=0.1, label='VAE Tokens')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f'loss_sequence_{batch_idx}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss sequence plot saved to: {save_path}")
    plt.close()
    return save_path

def analyze_bagel_batch(batch, tokenizer=None, detailed=True, save_dir="./plots"):
    """
    Analyze a BAGEL batch, print shapes including batch size, visualize first two elements, print text cut and decoded text for each batch element
    """
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 80)
    print("BAGEL BATCH ANALYSIS")
    print("=" * 80)
    
    # Print batch shapes
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, list):
            print(f"  {k}: list of len {len(v)}")
        else:
            print(f"  {k}: {type(v)}")
    
    # Determine if we have batched data based on padding setting
    if padding:
        # With padding, we have batch dimensions
        batch_size = 2  # Visualize first two elements
        attn_masks = batch.get('nested_attention_masks')
        if attn_masks and isinstance(attn_masks[0], torch.Tensor) and attn_masks[0].dim() == 3:
            batch_size = min(batch_size, attn_masks[0].shape[0])
    else:
        # Without padding, only one element (no batch dimension)
        batch_size = 1
    
    # Visualize elements in batch
    saved_plots = []
    attn_masks = batch.get('nested_attention_masks')
    text_indexes = batch.get('packed_text_indexes')
    vit_indexes = batch.get('packed_vit_token_indexes')
    vae_indexes = batch.get('packed_vae_token_indexes')
    sequence_length = batch.get('sequence_length')
    ce_loss_indexes = batch.get('ce_loss_indexes')
    mse_loss_indexes = batch.get('mse_loss_indexes')
    ce_loss_weights = batch.get('ce_loss_weights')
    
    for i in range(batch_size):
        print(f"\nVisualizing batch element {i}")
        attention_save_path = os.path.join(save_dir, f'attention_mask_{i}.png')
        saved_path = visualize_attention_mask(
            attn_masks, text_indexes, vit_indexes, vae_indexes,
            title=f"Attention Mask Structure (Batch {i})",
            save_path=attention_save_path,
            batch_idx=i
        )
        saved_plots.append(saved_path)
        loss_save_path = os.path.join(save_dir, f'loss_sequence_{i}.png')
        saved_path = visualize_loss_sequence(
            sequence_length, ce_loss_indexes, mse_loss_indexes,
            text_indexes, vit_indexes, vae_indexes,
            ce_loss_weights=ce_loss_weights,
            title=f"Loss Computation Sequence (Batch {i})",
            save_path=loss_save_path,
            batch_idx=i
        )
        saved_plots.append(saved_path)
    
    print(f"Visualizations saved to: {saved_plots}")
    
    # Print text cut and decoded text for each batch element
    text_cut = batch.get('text_cut')
    text_ids = batch.get('packed_text_ids')
    print("\nText cut and decoded text for each batch element:")
    
    if padding:
        # With padding, we have batch dimensions
        batch_size = text_ids.shape[0] if text_ids is not None and text_ids.dim() > 1 else 1
        for i in range(batch_size):
            cut_val = text_cut[i] if isinstance(text_cut, list) or (isinstance(text_cut, torch.Tensor) and text_cut.dim() > 0) else text_cut
            ids = text_ids[i] if text_ids.dim() > 1 else text_ids
            print(f"  Batch {i}: text_cut={cut_val}")
            if tokenizer:
                decoded = tokenizer.decode(ids.tolist())
                print(f"    Decoded text: '{decoded}'")
    else:
        # Without padding, no batch dimension
        cut_val = text_cut
        ids = text_ids
        print(f"  Sample: text_cut={cut_val}")
        if tokenizer:
            decoded = tokenizer.decode(ids.tolist())
            print(f"    Decoded text: '{decoded}'")

    # 1. BASIC BATCH INFO
    print("\n1. BASIC BATCH STRUCTURE:")
    print(f"   Batch type: {type(batch)}")
    print(f"   Batch keys: {list(batch.keys())}")
    
    if hasattr(batch, 'sequence_length'):
        print(f"   Total sequence length: {batch.sequence_length}")
    elif 'sequence_length' in batch:
        print(f"   Total sequence length: {batch['sequence_length']}")
    
    # 2. SEQUENCE STRUCTURE ANALYSIS
    print("\n2. SEQUENCE STRUCTURE:")
    sample_lens = batch.get('sample_lens', batch.get('sample_lens', []))
    # Flatten sample_lens if it contains nested lists
    if isinstance(sample_lens, list):
        flat_sample_lens = []
        for item in sample_lens:
            if isinstance(item, list):
                flat_sample_lens.extend(item)
            else:
                flat_sample_lens.append(item)
        sample_lens = flat_sample_lens
    if sample_lens:
        print(f"   Number of samples in batch: {len(sample_lens)}")
        print(f"   Sample lengths: {sample_lens}")
        print(f"   Total tokens across samples: {sum(sample_lens)}")
        print(f"   Average sample length: {np.mean(sample_lens):.1f}")
        print(f"   Min/Max sample length: {min(sample_lens)}/{max(sample_lens)}")
    
    # 3. TEXT TOKEN ANALYSIS
    print("\n3. TEXT TOKENS:")
    
    # Print text_cut for each batch element
    text_cut = batch.get('text_cut')
    if text_cut is not None:
        print(f"   Text cut shape/info: {text_cut if not hasattr(text_cut, 'shape') else text_cut.shape}")
        if isinstance(text_cut, list):
            for i, cut in enumerate(text_cut):
                print(f"   Text cut batch {i}: {cut}")
        else:
            print(f"   Text cut: {text_cut}")
    
    text_ids = batch.get('packed_text_ids')
    if text_ids is not None:
        print(f"   Text IDs shape: {text_ids.shape}")
        print(f"   Text IDs dtype: {text_ids.dtype}")
        print(f"   Vocab range: {text_ids.min().item()} - {text_ids.max().item()}")
        
        # Token frequency analysis
        unique_tokens, counts = torch.unique(text_ids, return_counts=True)
        print(f"   Unique tokens: {len(unique_tokens)}")
        print(f"   Most frequent tokens: {torch.topk(counts, 5).values.tolist()}")
        
        if tokenizer:
            # Decode tokens for each batch element
            try:
                if text_ids.dim() == 2:  # batched
                    batch_size = text_ids.shape[0]
                    print(f"   Decoded text for each batch element:")
                    for i in range(min(batch_size, 3)):  # Show first 3 batch elements
                        sample_tokens = text_ids[i, :100].tolist()  # Take first 100 tokens
                        
                        # Ensure sample_tokens is a flat list of integers
                        flat_tokens = []
                        for token in sample_tokens:
                            if isinstance(token, list):
                                flat_tokens.extend([int(t) for t in token])
                            else:
                                flat_tokens.append(int(token))
                        
                        decoded = tokenizer.decode(flat_tokens)
                        print(f"     Batch {i}: '{decoded[:200]}...'")
                else:  # single sequence
                    sample_tokens = text_ids[:100].tolist()
                    
                    # Ensure sample_tokens is a flat list of integers
                    flat_tokens = []
                    for token in sample_tokens:
                        if isinstance(token, list):
                            flat_tokens.extend([int(t) for t in token])
                        else:
                            flat_tokens.append(int(token))
                    
                    decoded = tokenizer.decode(flat_tokens)
                    print(f"   First 100 tokens decoded: '{decoded[:200]}...'")
            except Exception as e:
                print(f"   Could not decode tokens: {e}")
    
    text_indexes = batch.get('packed_text_indexes')
    if text_indexes is not None:
        # Handle batched text_indexes
        if isinstance(text_indexes, list):
            text_indexes_tensor = text_indexes[0] if text_indexes else torch.tensor([])
        else:
            text_indexes_tensor = text_indexes[0] if text_indexes.dim() == 2 else text_indexes
            
        if text_indexes_tensor.numel() > 0:
            print(f"   Text indexes shape: {text_indexes_tensor.shape}")
            print(f"   Text index range: {text_indexes_tensor.min().item()} - {text_indexes_tensor.max().item()}")
            print(f"   Text positions: {text_indexes_tensor.tolist()[:20]}...")
        else:
            print(f"   Text indexes: empty")
    
    # 4. VIT TOKEN ANALYSIS  
    print("\n4. VIT TOKENS:")
    
    vit_tokens = batch.get('packed_vit_tokens')
    if vit_tokens is not None:
        print(f"   ViT tokens shape: {vit_tokens.shape}")
        print(f"   ViT tokens dtype: {vit_tokens.dtype}")
        print(f"   Value range: [{vit_tokens.min().item():.4f}, {vit_tokens.max().item():.4f}]")
        print(f"   Mean: {vit_tokens.mean().item():.4f}, Std: {vit_tokens.std().item():.4f}")
        
    
    vit_indexes = batch.get('packed_vit_token_indexes')
    if vit_indexes is not None:
        # Handle batched vit_indexes
        if isinstance(vit_indexes, list):
            vit_indexes_tensor = vit_indexes[0] if vit_indexes else torch.tensor([])
        else:
            vit_indexes_tensor = vit_indexes[0] if vit_indexes.dim() == 2 else vit_indexes
            
        if vit_indexes_tensor.numel() > 0:
            print(f"   ViT token indexes shape: {vit_indexes_tensor.shape}")
            print(f"   ViT index range: {vit_indexes_tensor.min().item()} - {vit_indexes_tensor.max().item()}")
            print(f"   ViT positions: {vit_indexes_tensor.tolist()[:20]}...")
        else:
            print(f"   ViT token indexes: empty")
    
    vit_pos_ids = batch.get('packed_vit_position_ids')
    if vit_pos_ids is not None:
        # Handle batched position IDs
        if isinstance(vit_pos_ids, list):
            vit_pos_tensor = vit_pos_ids[0] if vit_pos_ids else torch.tensor([])
        else:
            vit_pos_tensor = vit_pos_ids[0] if vit_pos_ids.dim() == 2 else vit_pos_ids
            
        if vit_pos_tensor.numel() > 0:
            print(f"   ViT position IDs shape: {vit_pos_tensor.shape}")
            print(f"   ViT position range: {vit_pos_tensor.min().item()} - {vit_pos_tensor.max().item()}")
        else:
            print(f"   ViT position IDs: empty")
        
    vit_seqlens = batch.get('vit_token_seqlens')
    if vit_seqlens is not None:
        # Handle batched sequence lengths
        if isinstance(vit_seqlens, list):
            print(f"   ViT sequence lengths: {vit_seqlens}")
        else:
            print(f"   ViT sequence lengths: {vit_seqlens.tolist()}")
    
    # 5. VAE TOKEN ANALYSIS
    print("\n5. VAE TOKENS:")
    
    vae_images = batch.get('padded_images')
    if vae_images is not None:
        print(f"   VAE images shape: {vae_images.shape}")
        print(f"   VAE images dtype: {vae_images.dtype}")
        print(f"   Value range: [{vae_images.min().item():.4f}, {vae_images.max().item():.4f}]")
        print(f"   Mean: {vae_images.mean().item():.4f}, Std: {vae_images.std().item():.4f}")
    
    vae_indexes = batch.get('packed_vae_token_indexes') 
    if vae_indexes is not None:
        # Handle batched vae_indexes
        if isinstance(vae_indexes, list):
            vae_indexes_tensor = vae_indexes[0] if vae_indexes else torch.tensor([])
        else:
            vae_indexes_tensor = vae_indexes[0] if vae_indexes.dim() == 2 else vae_indexes
            
        if vae_indexes_tensor.numel() > 0:
            print(f"   VAE token indexes shape: {vae_indexes_tensor.shape}")
            print(f"   VAE index range: {vae_indexes_tensor.min().item()} - {vae_indexes_tensor.max().item()}")
            print(f"   VAE positions: {vae_indexes_tensor.tolist()[:20]}...")
        else:
            print(f"   VAE token indexes: empty")
    
    vae_latent_shapes = batch.get('patchified_vae_latent_shapes')
    if vae_latent_shapes:
        print(f"   VAE latent shapes: {vae_latent_shapes}")
        # Handle both single and batched latent shapes
        if isinstance(vae_latent_shapes[0], (list, tuple)) and len(vae_latent_shapes[0]) == 2:
            # Single batch element case
            total_latent_tokens = sum(h * w for h, w in vae_latent_shapes)
        else:
            # Batched case - take first element
            first_shapes = vae_latent_shapes[0] if isinstance(vae_latent_shapes[0], list) else [vae_latent_shapes[0]]
            total_latent_tokens = sum(h * w for h, w in first_shapes if isinstance((h, w), (list, tuple)) and len((h, w)) == 2)
        print(f"   Total VAE tokens: {total_latent_tokens}")
    
    vae_pos_ids = batch.get('packed_latent_position_ids')
    if vae_pos_ids is not None:
        # Handle batched position IDs
        if isinstance(vae_pos_ids, list):
            vae_pos_tensor = vae_pos_ids[0] if vae_pos_ids else torch.tensor([])
        else:
            vae_pos_tensor = vae_pos_ids[0] if vae_pos_ids.dim() == 2 else vae_pos_ids
            
        if vae_pos_tensor.numel() > 0:
            print(f"   VAE position IDs shape: {vae_pos_tensor.shape}")
            print(f"   VAE position range: {vae_pos_tensor.min().item()} - {vae_pos_tensor.max().item()}")
        else:
            print(f"   VAE position IDs: empty")
    
    # 6. POSITION AND ROPE ANALYSIS
    print("\n6. ROPE POSITIONS:")
    
    pos_ids = batch.get('packed_position_ids')
    if pos_ids is not None:
        # Handle batched position IDs
        if isinstance(pos_ids, list):
            pos_tensor = pos_ids[0] if pos_ids else torch.tensor([])
        else:
            pos_tensor = pos_ids[0] if pos_ids.dim() == 2 else pos_ids
            
        if pos_tensor.numel() > 0:
            print(f"   Position IDs shape: {pos_tensor.shape}")
            print(f"   Position range: {pos_tensor.min().item()} - {pos_tensor.max().item()}")
            
            # Analyze RoPE pattern
            unique_positions = torch.unique(pos_tensor)
            print(f"   Unique RoPE positions: {len(unique_positions)}")
            print(f"   RoPE positions: {unique_positions.tolist()}")
            
            # Count tokens per RoPE position
            rope_counts = [(pos.item(), (pos_tensor == pos).sum().item()) for pos in unique_positions]
            print(f"   Tokens per RoPE position: {rope_counts}")
        else:
            print(f"   Position IDs: empty")
    
    # 7. ATTENTION MASK ANALYSIS
    print("\n7. ATTENTION MASKS:")
    
    attn_masks = batch.get('nested_attention_masks')
    if attn_masks:
        print(f"   Number of attention masks: {len(attn_masks)}")
        for i, mask in enumerate(attn_masks):
            if isinstance(mask, list):
                mask_tensor = mask[0] if mask else torch.tensor([])
            else:
                mask_tensor = mask[0] if mask.dim() == 3 else mask
            
            if mask_tensor.numel() > 0:
                print(f"   Mask {i} shape: {mask_tensor.shape}")
            else:
                print(f"   Mask {i}: empty")

        # Only access text/vit/vae indexes if they exist and are properly formatted
        text_indexes = batch.get('packed_text_indexes')
        vit_indexes = batch.get('packed_vit_token_indexes')  
        vae_indexes = batch.get('packed_vae_token_indexes')
        
        if text_indexes is not None and vit_indexes is not None and vae_indexes is not None:
            # Handle batched indexes
            if isinstance(text_indexes, list):
                text_tensor = text_indexes[0] if text_indexes else torch.tensor([])
            else:
                text_tensor = text_indexes[0] if text_indexes.dim() == 2 else text_indexes
                
            if isinstance(vit_indexes, list):
                vit_tensor = vit_indexes[0] if vit_indexes else torch.tensor([])
            else:
                vit_tensor = vit_indexes[0] if vit_indexes.dim() == 2 else vit_indexes
                
            if isinstance(vae_indexes, list):
                vae_tensor = vae_indexes[0] if vae_indexes else torch.tensor([])
            else:
                vae_tensor = vae_indexes[0] if vae_indexes.dim() == 2 else vae_indexes

            text_len = text_tensor.shape[0] if text_tensor.numel() > 0 else 0
            vit_len = vit_tensor.shape[0] if vit_tensor.numel() > 0 else 0
            vae_len = vae_tensor.shape[0] if vae_tensor.numel() > 0 else 0
            
            total = text_len + vit_len + vae_len
            print(f"   Token breakdown: text={text_len}, vit={vit_len}, vae={vae_len}, total={total}")
    
    # 8. LOSS COMPUTATION ANALYSIS
    print("\n8. LOSS COMPUTATION:")
    
    # Text loss
    ce_indexes = batch.get('ce_loss_indexes')
    ce_weights = batch.get('ce_loss_weights')
    label_ids = batch.get('packed_label_ids')
    
    if ce_indexes is not None:
        # Handle batched ce_indexes
        if isinstance(ce_indexes, list):
            ce_tensor = ce_indexes[0] if ce_indexes else torch.tensor([])
        else:
            ce_tensor = ce_indexes[0] if ce_indexes.dim() == 2 else ce_indexes
            
        if ce_tensor.numel() > 0:
            print(f"   CE loss indexes shape: {ce_tensor.shape}")
            print(f"   CE loss range: {ce_tensor.min().item()} - {ce_tensor.max().item()}")
            print(f"   Number of text tokens with loss: {len(ce_tensor)}")
        else:
            print(f"   CE loss indexes: empty")
    
    if ce_weights is not None:
        # Handle batched weights
        if isinstance(ce_weights, list):
            weights_tensor = ce_weights[0] if ce_weights else torch.tensor([])
        else:
            weights_tensor = ce_weights[0] if ce_weights.dim() == 2 else ce_weights
            
        if weights_tensor.numel() > 0:
            print(f"   CE loss weights shape: {weights_tensor.shape}")
            print(f"   Weight range: {weights_tensor.min().item():.4f} - {weights_tensor.max().item():.4f}")
            print(ce_weights)
        else:
            print(f"   CE loss weights: empty")
    
    if label_ids is not None:
        # Handle batched label IDs
        if isinstance(label_ids, list):
            labels_tensor = label_ids[0] if label_ids else torch.tensor([])
        else:
            labels_tensor = label_ids[0] if label_ids.dim() == 2 else label_ids
            
        if labels_tensor.numel() > 0:
            print(f"   Label IDs shape: {labels_tensor.shape}")
            print(f"   Label range: {labels_tensor.min().item()} - {labels_tensor.max().item()}")
        else:
            print(f"   Label IDs: empty")
    
    # VAE loss
    mse_indexes = batch.get('mse_loss_indexes')
    timesteps = batch.get('packed_timesteps')
    
    if mse_indexes is not None:
        # Handle batched MSE indexes
        if isinstance(mse_indexes, list):
            mse_tensor = mse_indexes[0] if mse_indexes else torch.tensor([])
        else:
            mse_tensor = mse_indexes[0] if mse_indexes.dim() == 2 else mse_indexes
            
        if mse_tensor.numel() > 0:
            print(f"   MSE loss indexes shape: {mse_tensor.shape}")
            print(f"   MSE loss range: {mse_tensor.min().item()} - {mse_tensor.max().item()}")
            print(f"   Number of VAE tokens with loss: {len(mse_tensor)}")
        else:
            print(f"   MSE loss indexes: empty")
    
    if timesteps is not None:
        # Handle batched timesteps
        if isinstance(timesteps, list):
            timesteps_tensor = timesteps[0] if timesteps else torch.tensor([])
        else:
            timesteps_tensor = timesteps[0] if timesteps.dim() == 2 else timesteps
            
        if timesteps_tensor.numel() > 0:
            print(f"   Timesteps shape: {timesteps_tensor.shape}")
            valid_timesteps = timesteps_tensor[timesteps_tensor != float('-inf')]
            if len(valid_timesteps) > 0:
                print(f"   Valid timesteps: {len(valid_timesteps)}")
                print(f"   Timestep range: {valid_timesteps.min().item():.1f} - {valid_timesteps.max().item():.1f}")
                print(f"   Average timestep: {valid_timesteps.float().mean().item():.1f}")
            else:
                print("   No valid timesteps found")
        else:
            print(f"   Timesteps: empty")
    return None


def test_dataloader_statistics(dataloader, tokenizer=None, num_batches=3):
    """
    Test multiple batches and provide aggregate statistics
    """
    
    print("=" * 80)
    print("MULTI-BATCH DATALOADER TESTING")
    print("=" * 80)
    
    batch_stats = []
    
    try:
        for i, batch in enumerate(dataloader):
            print(f"\n{'='*20} BATCH {i+1} {'='*20}")
            
            # Analyze this batch
            stats = analyze_bagel_batch(batch, tokenizer, detailed=(i==0))
            batch_stats.append(stats)
            
            if i >= num_batches - 1:
                break
                
    except Exception as e:
        print(f"Error during batch iteration: {e}")
        import traceback
        traceback.print_exc()
        return

# USAGE EXAMPLE:
if __name__ == "__main__":
    # Example usage for diagnosing a BAGEL batch and dataloader
    if True:
        # Set up tokenizer and special tokens
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        special_tokens = new_token_ids

        # Load VAE model
        vae_model, vae_config = load_ae("/home/haoming/Bagel/models/BAGEL-7B-MoT/ae.safetensors")
        vae_model.eval()

        # Get loader for train split
        dataloader = get_loader(
            split="train",
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            vae_model=vae_model,
            cfg_path="/home/haoming/Bagel/data/configs/datacomp.yaml"
        )

        # Test and analyze a few batches
        test_dataloader_statistics(dataloader, tokenizer=tokenizer, num_batches=1)