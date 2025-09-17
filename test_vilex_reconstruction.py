#!/usr/bin/env python3
"""
Test script for VILEX reconstruction pipeline.

This script tests the new VILEX reconstruction functionality:
1. Takes text prompt + input image
2. Processes image through VILEX projector to get 32 VILEX tokens
3. Uses VILEX tokens to guide image generation (no VAE encoding)
4. Follows training sequence: BOS + text + EOS + <img_start> + VILEX_tokens + <img_end> + generated_image
"""

import os
import torch
import argparse
import gc
from PIL import Image

# Import BAGEL components
from inferencer import InterleaveInferencer
from modeling.bagel.bagel_vilex import Bagel, BagelConfig
from modeling.bagel import Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights


def load_model(model_path, mode=1):
    """Load BAGEL model with VILEX configuration"""
    
    # Load configs
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # VILEX configuration - MUST be True for reconstruction
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
        use_vilex=True,  # use vilex and load the configs
        vilex_config={
            "num_layer": 4,
            "num_heads": 8,
            "num_output_tokens": 32,  # Fixed number of VILEX tokens -- in generation the k taildrop is 
            "taildrop_prob": 0.1,
            "taildrop_max": 10,
        },
    )

    # Initialize model
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Load tokenizer and special tokens
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Device mapping for multi-GPU
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    # Ensure key modules are on same device
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',  # VILEX projector
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load checkpoint
    ckpt_path = "/home/haoming/Bagel/experiments/results_tune_no_ce/checkpoints/0012000"
    
    if mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(ckpt_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
            strict=False
        ).eval()
    else:
        raise NotImplementedError("Only mode 1 supported for now")

    return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform


def test_vilex_reconstruction():
    """Test VILEX reconstruction with a sample image and prompt"""
    
    print("=== VILEX Reconstruction Test ===")
    
    # Configuration
    model_path = "models/BAGEL-7B-MoT"
    
    # Load model
    print("Loading BAGEL model with VILEX configuration...")
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = load_model(model_path)
    
    # Create inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    
    # Test inputs
    input_image_path = "/home/haoming/Bagel/puppy.jpg"
    
    print(f"Input image: {input_image_path}")
    
    # Load test image
    if not os.path.exists(input_image_path):
        print(f"Error: Test image not found at {input_image_path}")
        return
        
    input_image = Image.open(input_image_path)
    print(f"Loaded image size: {input_image.size}")
    
    # Test VILEX reconstruction
    print("\n=== Starting VILEX Reconstruction ===")
    
    result = inferencer(
        image=input_image,
        think=False,  # Can be set to True for thinking mode
        cfg_text_scale=4.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),  # Output size
        use_vilex=True,  # Enable VILEX reconstruction
    )
    
    # Save results
    output_dir = "/home/haoming/Bagel/results/vilex_reconstruction"
    os.makedirs(output_dir, exist_ok=True)
    
    if result['image'] is not None:
        output_path = os.path.join(output_dir, "vilex_reconstructed.png")
        result['image'].save(output_path)
        print(f"Reconstructed image saved to: {output_path}")
    
    if result['text'] is not None:
        text_path = os.path.join(output_dir, "thinking_process.txt")
        with open(text_path, 'w') as f:
            f.write(result['text'])
        print(f"Thinking process saved to: {text_path}")
    
    # Save original for comparison
    original_path = os.path.join(output_dir, "original_input.png")
    input_image.save(original_path)
    print(f"Original image saved to: {original_path}")
    
    print("\n=== VILEX Reconstruction Complete ===")
    print(f"Results saved in: {output_dir}")
    
    return result


def clear_memory():
    """Clear GPU memory and force garbage collection"""
    import gc
    
    # Clear Python cache
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_checkpoint_comparison():
    """Test VILEX reconstruction across multiple checkpoints"""
    
    print("=== Multi-Checkpoint VILEX Reconstruction Test ===")
    
    # Configuration
    model_path = "models/BAGEL-7B-MoT"
    
    # Define checkpoint paths from different experiments
    checkpoint_configs = [
        # results_tune checkpoints
        {
            "name": "results_tune_0000050",
            "path": "/home/haoming/Bagel/experiments/results_tune/checkpoints/0000050",
            "experiment": "results_tune"
        },
        {
            "name": "results_tune_0000100", 
            "path": "/home/haoming/Bagel/experiments/results_tune/checkpoints/0000100",
            "experiment": "results_tune"
        },
        {
            "name": "results_tune_0002000",
            "path": "/home/haoming/Bagel/experiments/results_tune/checkpoints/0002000", 
            "experiment": "results_tune"
        },
        {
            "name": "results_tune_0009000",
            "path": "/home/haoming/Bagel/experiments/results_tune/checkpoints/0009000",
            "experiment": "results_tune"
        },
        # result_tune_new checkpoints
        {
            "name": "result_tune_new_0002000",
            "path": "/home/haoming/Bagel/experiments/result_tune_new/checkpoints/0002000",
            "experiment": "result_tune_new"
        },
        {
            "name": "result_tune_new_0004000",
            "path": "/home/haoming/Bagel/experiments/result_tune_new/checkpoints/0004000",
            "experiment": "result_tune_new"
        },
        {
            "name": "result_tune_new_0008000",
            "path": "/home/haoming/Bagel/experiments/result_tune_new/checkpoints/0008000",
            "experiment": "result_tune_new"
        },
        {
            "name": "result_tune_new_0018000",
            "path": "/home/haoming/Bagel/experiments/result_tune_new/checkpoints/0018000",
            "experiment": "result_tune_new"
        },
        {
            "name": "result_tune_new_0028000",
            "path": "/home/haoming/Bagel/experiments/result_tune_new/checkpoints/0028000",
            "experiment": "result_tune_new"
        }
    ]
    
    # Test inputs
    input_image_path = "/home/haoming/Bagel/puppy.jpg"
    test_prompt = "I want you to render an image of"  # Simple prompt for consistency
    
    if not os.path.exists(input_image_path):
        print(f"Error: Test image not found at {input_image_path}")
        return
        
    input_image = Image.open(input_image_path)
    print(f"Using input image: {input_image_path} (size: {input_image.size})")
    
    # Create main output directory
    output_base_dir = "/home/haoming/Bagel/results/checkpoint_comparison"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Save original image once
    original_path = os.path.join(output_base_dir, "original_input.png")
    input_image.save(original_path)
    print(f"Original image saved to: {original_path}")
    
    successful_tests = 0
    failed_tests = []
    
    for i, checkpoint_config in enumerate(checkpoint_configs):
        checkpoint_name = checkpoint_config["name"]
        checkpoint_path = checkpoint_config["path"]
        
        print(f"\n{'='*60}")
        print(f"Testing checkpoint {i+1}/{len(checkpoint_configs)}: {checkpoint_name}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Check if checkpoint exists
        ema_path = os.path.join(checkpoint_path, "ema.safetensors")
        if not os.path.exists(ema_path):
            print(f"Warning: Checkpoint not found at {ema_path}, skipping...")
            failed_tests.append(f"{checkpoint_name}: Checkpoint not found")
            continue
        
        try:
            # Clear memory before loading new checkpoint
            clear_memory()
            
            # Create custom load function for this checkpoint
            def load_model_checkpoint(model_path, checkpoint_path):
                """Load model with specific checkpoint"""
                
                # Load configs
                llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
                llm_config.qk_norm = True
                llm_config.tie_word_embeddings = False
                llm_config.layer_module = "Qwen2MoTDecoderLayer"

                vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
                vit_config.rope = False
                vit_config.num_hidden_layers -= 1

                vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

                # VILEX configuration
                config = BagelConfig(
                    visual_gen=True,
                    visual_und=True,
                    llm_config=llm_config, 
                    vit_config=vit_config,
                    vae_config=vae_config,
                    vit_max_num_patch_per_side=70,
                    connector_act='gelu_pytorch_tanh',
                    latent_patch_size=2,
                    max_latent_size=64,
                    use_vilex=True,
                    vilex_config={
                        "num_layer": 4,
                        "num_heads": 8,
                        "num_output_tokens": 32,
                        "taildrop_prob": 0.1,
                        "taildrop_max": 10,
                    },
                )

                # Initialize model
                with init_empty_weights():
                    language_model = Qwen2ForCausalLM(llm_config)
                    vit_model = SiglipVisionModel(vit_config)
                    model = Bagel(language_model, vit_model, config)
                    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

                # Load tokenizer and special tokens
                tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
                tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

                # Image transforms
                vae_transform = ImageTransform(1024, 512, 16)
                vit_transform = ImageTransform(980, 224, 14)

                # Device mapping
                device_map = infer_auto_device_map(
                    model,
                    max_memory={i: "70GiB" for i in range(torch.cuda.device_count())},  # Reduced memory limit
                    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                )

                # Ensure key modules are on same device
                same_device_modules = [
                    'language_model.model.embed_tokens',
                    'time_embedder',
                    'latent_pos_embed', 
                    'vae2llm',
                    'llm2vae',
                    'connector',
                    'vit_pos_embed'
                ]

                if torch.cuda.device_count() == 1:
                    first_device = device_map.get(same_device_modules[0], "cuda:0")
                    for k in same_device_modules:
                        device_map[k] = first_device
                else:
                    first_device = device_map.get(same_device_modules[0])
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device

                # Load specific checkpoint
                model = load_checkpoint_and_dispatch(
                    model,
                    checkpoint=os.path.join(checkpoint_path, "ema.safetensors"),
                    device_map=device_map,
                    offload_buffers=True,
                    offload_folder="offload",
                    dtype=torch.bfloat16,
                    force_hooks=True,
                    strict=False
                ).eval()

                return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform
            
            # Load model with current checkpoint
            print(f"Loading model with checkpoint: {checkpoint_name}")
            model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = load_model_checkpoint(
                model_path, checkpoint_path
            )
            
            # Create inferencer
            inferencer = InterleaveInferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )
            
            print(f"Running VILEX reconstruction with prompt: '{test_prompt}'")
            
            # Run VILEX reconstruction
            result = inferencer(
                text=test_prompt,
                image=input_image,
                use_vilex=True,
                cfg_text_scale=4.0,
                cfg_interval=[0.4, 1.0],
                timestep_shift=3.0,
                num_timesteps=50,
                cfg_renorm_min=0.0,
                cfg_renorm_type="global",
                image_shapes=(1024, 1024),
            )
            
            # Save result
            if result['image'] is not None:
                output_path = os.path.join(output_base_dir, f"{checkpoint_name}_reconstruction.png")
                result['image'].save(output_path)
                print(f"✅ Success! Saved to: {output_path}")
                successful_tests += 1
            else:
                print("❌ Failed: No image generated")
                failed_tests.append(f"{checkpoint_name}: No image generated")
            
            # Clean up current model to free memory
            del model, inferencer
            clear_memory()
            
        except Exception as e:
            print(f"❌ Error with checkpoint {checkpoint_name}: {str(e)}")
            failed_tests.append(f"{checkpoint_name}: {str(e)}")
            # Clear memory even on failure
            clear_memory()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CHECKPOINT COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Total checkpoints tested: {len(checkpoint_configs)}")
    print(f"Successful reconstructions: {successful_tests}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    print(f"\nResults saved in: {output_base_dir}")
    print(f"Original input image: {original_path}")


def test_comparison():
    """Test both regular and VILEX reconstruction for comparison"""
    
    print("=== Comparison Test: Regular vs VILEX Reconstruction ===")
    
    # Configuration
    model_path = "models/BAGEL-7B-MoT"
    
    # Load model
    print("Loading BAGEL model...")
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = load_model(model_path)
    
    # Create inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    
    # Test inputs
    input_image_path = "/home/haoming/Bagel/puppy.jpg"
    test_prompt = "a cute dog in a beautiful garden"
    
    input_image = Image.open(input_image_path)
    output_dir = "/home/haoming/Bagel/results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Regular text-to-image generation
    print("\n1. Testing regular text-to-image generation...")
    regular_result = inferencer(
        text=test_prompt,
        cfg_text_scale=4.0,
        image_shapes=(1024, 1024)
    )
    
    if regular_result['image'] is not None:
        regular_path = os.path.join(output_dir, "regular_generation.png")
        regular_result['image'].save(regular_path)
        print(f"Regular generation saved to: {regular_path}")
    
    # 2. VILEX reconstruction
    print("\n2. Testing VILEX reconstruction...")
    vilex_result = inferencer(
        text=test_prompt,
        image=input_image,
        use_vilex=True,  # Enable VILEX mode
        cfg_text_scale=4.0,
        image_shapes=(1024, 1024)
    )
    
    if vilex_result['image'] is not None:
        vilex_path = os.path.join(output_dir, "vilex_reconstruction.png")
        vilex_result['image'].save(vilex_path)
        print(f"VILEX reconstruction saved to: {vilex_path}")
    
    # Save original
    original_path = os.path.join(output_dir, "original_reference.png")
    input_image.save(original_path)
    print(f"Original reference saved to: {original_path}")
    
    print(f"\nComparison results saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VILEX reconstruction pipeline")
    parser.add_argument("--mode", choices=["single", "comparison", "checkpoints"], default="single",
                       help="Test mode: 'single' for VILEX only, 'comparison' for both methods, 'checkpoints' for multi-checkpoint comparison")
    
    args = parser.parse_args()
    
    try:
        test_checkpoint_comparison()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final memory cleanup
        clear_memory()
