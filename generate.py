import gradio as gr
import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel.bagel_vilex import (
    BagelConfig, Bagel
)
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed


from modeling.bagel import (
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
import yaml
import argparse
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from omegaconf import OmegaConf
from train.fsdp_utils import FSDPCheckpoint
from train.train_utils import create_logger, get_latest_ckpt


@dataclass
class ModelArguments:
    model_path: str = field(
        default="models/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="google/siglip-so400m-patch14-384",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=16,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=32,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )
    num_layer: int = field(
        default= 4,
        metadata={"help": "number of layers for vilex's multilayer attention pooling projector."}
    )
    num_heads: int = field(
        default= 8,
        metadata={"help": "number of heads used during attention pooling."}
    )
    num_output_tokens: int = field(  # 32 for real, 1 for debugging
        default= 32,
        metadata={"help": "number of output tokens the project outputs."}
    )
    tail_drop_prob: float = field(  # ← Change from int to float
        default= 0.1,
        metadata={"help": "taildrop probability for vilex."}
    )
    tail_drop_max: int = field(
        default= 10,
        metadata={"help": "max number of tokens dropped ."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )



@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Enable or disable Weights & Biases logging."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

def load_config(config_path="config/test_inference.yaml"):
    print(f"Loading configuration from {config_path}...")
    config = OmegaConf.load(config_path)
    print("Configuration loaded successfully.")
    return config


def setup_device_mapping(model, config):
    print("Setting up device mapping...")
    device_config = config.get('device', {})
    
    # Create device map
    device_map = infer_auto_device_map(
        model,
        max_memory={i: device_config.get('max_memory_per_gpu', "80GiB") 
                   for i in range(torch.cuda.device_count())},
        no_split_module_classes=device_config.get('no_split_module_classes', 
                                                 ["Bagel", "Qwen2MoTDecoderLayer"]),
    )
    print("Device map created successfully.")

    # Handle same device modules
    same_device_modules = device_config.get('same_device_modules', [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ])

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    print("Device mapping setup completed.")
    return device_map


def load_model_weights(model, ckpt_path, config):
    print(f"Loading model weights from {ckpt_path}...")
    logger = create_logger("logs", rank=0)  # Create a logger for logging
    model, _ = FSDPCheckpoint.try_load_ckpt(
        ckpt_path,
        logger=logger,
        model=model,
        ema_model=None,
        resume_from_ema=True,  # Use EMA weights
    )
    print("Model weights loaded successfully.")
    return model


def load_modules(config):
    print("Loading modules...")
    model_path = config['model']['model_path']
    mode = config['model']['mode']

    # Load model configs with overrides
    print("Loading LLM and ViT configurations...")
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -=1

    print("Loading VAE model...")
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Arguments parsed successfully.")

    # Create BAGEL config from YAML
    print("Creating BAGEL configuration...")
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        use_vilex=True,  # Enable ViLex like test_whole.py
        vilex_config={
            "num_layer": model_args.num_layer,
            "num_heads": model_args.num_heads,
            "num_output_tokens": model_args.num_output_tokens,
            "taildrop_prob": model_args.tail_drop_prob,
            "taildrop_max": model_args.tail_drop_max,
        },
    )
    print("BAGEL configuration created successfully.")

    # Initialize model
    print("Initializing model...")
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.eval()
        language_model.eval()
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    print("Model initialized successfully.")

    # Setup device mapping and load weights
    print("Setting up device mapping...")
    device_map = setup_device_mapping(model, config)
    print("Loading model weights...")
    ckpt_path = config["model"]["ckpt_path"]
    model = load_model_weights(model, ckpt_path, config)

    # Load tokenizer and transforms
    print("Loading tokenizer and transforms...")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(
        config['transforms']['vae']['resolution'],
        config['transforms']['vae']['crop_size'], 
        config['transforms']['vae']['patch_size']
    )
    vit_transform = ImageTransform(
        config['transforms']['vit']['resolution'],
        config['transforms']['vit']['crop_size'],
        config['transforms']['vit']['patch_size']
    )
    print("Modules loaded successfully.")
    return model, tokenizer, new_token_ids, vae_model, vae_transform, vit_transform, config 


def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


# Text to Image function with thinking option and hyperparameters
def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4, 
                 timestep_shift=3.0, num_timesteps=50, 
                 cfg_renorm_min=0.0, cfg_renorm_type="global", 
                 max_think_token_n=1024, do_sample=False, text_temperature=0.3,
                 seed=0, image_ratio="1:1"):
    # Set seed for reproducibility
    set_seed(seed)

    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    elif image_ratio == "3:4":
        image_shapes = (1024, 768) 
    elif image_ratio == "16:9":
        image_shapes = (576, 1024)
    elif image_ratio == "9:16":
        image_shapes = (1024, 576) 
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )
    
    # Call inferencer with or without think parameter based on user choice
    print("calling inferencer")
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    return result["image"], result.get("text", None)


# Image Understanding function with thinking option and hyperparameters
def image_understanding(image: Image.Image, prompt: str, show_thinking=False, 
                        do_sample=False, text_temperature=0.3, max_new_tokens=512):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens, # Set max_length
    )
    
    # Use show_thinking parameter to control thinking process
    result = inferencer(image=image, text=prompt, think=show_thinking, 
                        understanding_output=True, **inference_hyper)
    return result["text"]


# Image Editing function with thinking option and hyperparameters
def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0, 
              cfg_img_scale=2.0, cfg_interval=0.0, 
              timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0, 
              cfg_renorm_type="text_channel", max_think_token_n=1024, 
              do_sample=False, text_temperature=0.3, seed=0):
    # Set seed for reproducibility
    set_seed(seed)
    
    if image is None:
        return "Please upload an image.", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    # Include thinking parameter based on user choice
    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
    return result["image"], result.get("text", "")


# Helper function to load example images
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None

if __name__ == "__main__":
    print("Starting tests...")
    
    # Load configuration
    print("Loading configuration...")
    configs = load_config()
    print("Configuration loaded successfully.")
    
    # Load modules
    print("Loading modules...")
    model, tokenizer, new_token_ids, vae_model, vae_transform, vit_transform, config = load_modules(configs)
    print("Modules loaded successfully.")
    
    # Debugging: Check loaded components
    
    # Create inferencer
    print("Creating inferencer...")
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model, 
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    print("Inferencer created successfully.")
    
    # Test text-to-image generation
    print("Testing text-to-image generation...")
    try:
        image, thinking = text_to_image("A beautiful sunset over mountains")
        if image:
            image.save("test_output.png")
            print("Image saved as test_output.png")
        else:
            print("No image generated.")
    except Exception as e:
        print(f"Error during text-to-image generation: {e}")
    
    print("Tests completed!")
