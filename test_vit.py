import torch
import os
from data.vilex_dataset import get_loader
from modeling.bagel import SiglipVisionConfig, SiglipVisionModel

def debug_vit_mismatch():
    print("=== DEBUG VIT MISMATCH ===")
    
    # Load data
    print("Loading data...")
    train_loader = get_loader(split="train")
    
    # Get first batch
    print("Getting first batch...")
    for i, data in enumerate(train_loader):
        print(f"Batch {i} data keys:", list(data.keys()))
        
        # Check VIT tokens
        if 'packed_vit_tokens' in data:
            vit_tokens = data['packed_vit_tokens']
            print(f"VIT tokens shape: {vit_tokens.shape}")
            print(f"VIT tokens dtype: {vit_tokens.dtype}")
            print(f"VIT tokens min/max: {vit_tokens.min():.3f} / {vit_tokens.max():.3f}")
        else:
            print("No packed_vit_tokens found in data")
            return
        
        break
    
    # Create VIT model with default config
    print("\nCreating VIT model...")
    vit_config = SiglipVisionConfig.from_json_file(os.path.join("/home/haoming/Bagel/models/BAGEL-7B-MoT", "vit_config.json"))
    print(f"VIT config - hidden_size: {vit_config.hidden_size}")
    print(f"VIT config - patch_size: {vit_config.patch_size}")
    print(f"VIT config - num_channels: {vit_config.num_channels}")
    print(f"VIT config - image_size: {vit_config.image_size}")
    
    vit_model = SiglipVisionModel.from_pretrained("/home/haoming/Bagel/models/BAGEL-7B-MoT/ema.safetensors")
    
    # Check patch embedding layer
    patch_embedding = vit_model.vision_model.embeddings.patch_embedding
    print(f"\nPatch embedding layer:")
    print(f"  Type: {type(patch_embedding)}")
    print(f"  Weight shape: {patch_embedding.weight.shape}")
    if hasattr(patch_embedding, 'in_channels'):
        print(f"  in_channels: {patch_embedding.in_channels}")
        print(f"  out_channels: {patch_embedding.out_channels}")
        print(f"  kernel_size: {patch_embedding.kernel_size}")
    
    # Convert to linear like in training
    print(f"\nConverting Conv2d to Linear...")
    vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    
    patch_embedding_linear = vit_model.vision_model.embeddings.patch_embedding
    print(f"Linear patch embedding:")
    print(f"  Type: {type(patch_embedding_linear)}")
    print(f"  Weight shape: {patch_embedding_linear.weight.shape}")
    print(f"  Expected input size: {patch_embedding_linear.in_features}")
    print(f"  Output size: {patch_embedding_linear.out_features}")
    
    # Try to apply the data
    print(f"\nTrying to apply VIT tokens to model...")
    try:
        # Check if we need other inputs
        if 'packed_vit_position_ids' in data:
            position_ids = data['packed_vit_position_ids']
            print(f"Position IDs shape: {position_ids.shape}")
            print(f"Position IDs min/max: {position_ids.min().item()} / {position_ids.max().item()}")
            print(f"Position IDs first 10: {position_ids}")
            print(f"Position IDs last 10: {position_ids[-10:]}")
            
            # Check RoPE tensor size
            rope_size = vit_model.vision_model.rope.cos_h.shape[0]
            print(f"RoPE tensor size: {rope_size}")
            print(f"Position IDs exceed RoPE size: {position_ids.max().item() >= rope_size}")
        else:
            print("Creating dummy position IDs")
            position_ids = torch.arange(vit_tokens.shape[0])
            
        if 'vit_token_seqlens' in data:
            seqlens = data['vit_token_seqlens']
            print(f"Seqlens: {seqlens}")
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(seqlens).item()
        else:
            print("Creating dummy seqlens")
            cu_seqlens = torch.tensor([0, vit_tokens.shape[0]], dtype=torch.int32)
            max_seqlen = vit_tokens.shape[0]
            
        print(f"cu_seqlens: {cu_seqlens}")
        print(f"max_seqlen: {max_seqlen}")
        
        # Try the forward pass
        output = vit_model(
            packed_pixel_values=vit_tokens,
            packed_flattened_position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        print(f"SUCCESS! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e)}")
        
        # Debug the mismatch
        print(f"\nDEBUG INFO:")
        print(f"VIT tokens shape: {vit_tokens.shape}")
        print(f"Expected input features (linear): {patch_embedding_linear.in_features}")
        print(f"Expected input calculation: {vit_config.num_channels} * {vit_config.patch_size}^2 = {vit_config.num_channels * vit_config.patch_size**2}")
        
        if len(vit_tokens.shape) == 4:
            print(f"VIT tokens looks like image format: [B, C, H, W] = {vit_tokens.shape}")
            expected_patches = (vit_tokens.shape[2] // vit_config.patch_size) * (vit_tokens.shape[3] // vit_config.patch_size)
            expected_features = vit_config.num_channels * vit_config.patch_size * vit_config.patch_size
            print(f"Expected after patchify: [{expected_patches}, {expected_features}]")
        elif len(vit_tokens.shape) == 2:
            print(f"VIT tokens is already flattened: [N, D] = {vit_tokens.shape}")
            print(f"But linear expects D = {patch_embedding_linear.in_features}")
            
    print("\n=== END DEBUG ===")

if __name__ == "__main__":
    debug_vit_mismatch()
