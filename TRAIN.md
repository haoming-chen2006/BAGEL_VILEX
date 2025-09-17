# Data prepration

We provide data examples for **T2I**, **Editing**, and **VLM** tasks. The T2I dataset is generated using [FLUX.1‑dev](https://huggingface.co/black-forest-labs/FLUX.1-dev); the editing examples are randomly sampled from [SEED‑Data‑Edit‑Part3](https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part2-3); and the VLM set is sourced from [LLaVA‑OneVision‑Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).

We offer examples in both raw-image folder and parquet shard formats. For other data formats, you can use our dataset code as a template and extend it as needed.


1. **Download the sample dataset**

   ```bash
   wget -O bagel_example.zip \
     https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
   unzip bagel_example.zip -d /data
   ```
2. **Expected hierarchy**

   ```text
   bagel_example
   ├── t2i/                           # text-to-image (parquet)
   ├── editing/                       # image editing (parquet)
   │   ├── seedxedit_multi/
   │   └── parquet_info/
   └── vlm/
       ├── images/                    # JPEG / PNG frames
       └── llava_ov_si.jsonl          # vision‑language SFT conversations
   ```
3. Edit every `your_data_path` placeholder in **`data/dataset_info.py`**.
4. *(Optional)*  Extend `DATASET_INFO` with your own parquet shards or JSONL files to mix extra data.

---

# Training

The baseline training recipe looks like this (replace environment variables with real paths or values):

```shell
# Pre-training
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --llm_path $llm_path \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --layer_module Qwen2MoTDecoderLayer \
  --use_flex True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --max_latent_size 64  # 32 for low-resolution pre-training

# Fine-tuning
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240
```

- **When fine-tuning BAGEL, set `max_latent_size=64` to ensure the correct pretrained weights are loaded.** If this is not set, an out-of-bounds error may occur.
- The total value of `num_used_data` should be greater than `NUM_GPUS × NUM_WORKERS`. (For toy data, use `num_worker=1`.)
- For T2I-only fine-tuning, set `visual_und=False`. For VLM-only fine-tuning, set `visual_gen=False`.
- For debugging purposes, use smaller values for `expected_num_tokens`, `max_num_tokens`, and `max_num_tokens_per_sample`.
- When fine-tuning on toy data, the loss behaves as follows:
    ```shell
    [2025-05-25 17:01:37] (step=0000000) Train Loss mse: 0.4063, Train Loss ce: 0.5504, Train Steps/Sec: 0.01, 
    [2025-05-25 17:01:40] (step=0000001) Train Loss mse: 0.4121, Train Loss ce: 0.8152, Train Steps/Sec: 0.44, 
    [2025-05-25 17:01:42] (step=0000002) Train Loss mse: 0.3876, Train Loss ce: 1.3411, Train Steps/Sec: 0.40, 
    [2025-05-25 17:01:45] (step=0000003) Train Loss mse: 0.3825, Train Loss ce: 0.7360, Train Steps/Sec: 0.44, 
    ```


You are encouraged to adjust any of these hyperparameters to fit your GPU budget and the scale of your dataset. If you encounter any issues, please open an issue for assistance. 🎉


## Model config


| Argument                     | Default                                     | Description                                                     |
| ---------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| `llm_path`                   | `hf/Qwen2.5-0.5B-Instruct`                  | Language‑model backbone (HuggingFace repo or local folder).     |
| `vae_path`                   | `flux/vae/ae.safetensors`                   | Pre‑trained VAE checkpoint for latent diffusion.                |
| `vit_path`                   | `hf/siglip-so400m-14-980-flash-attn2-navit` | SigLIP ViT used for image understanding.                        |
| `max_latent_size`            | `32`                                        | Maximum latent grid side; defines highest generable resolution. |
| `latent_patch_size`          | `2`                                         | VAE pixels represented by one latent patch.                     |
| `vit_max_num_patch_per_side` | `70`                                        | Max ViT patches per image side after resizing.                  |
| `text_cond_dropout_prob`     | `0.1`                                       | Probability to drop text conditioning while training.           |
| `vae_cond_dropout_prob`      | `0.3`                                       | Dropout on VAE latent inputs.                                   |
| `vit_cond_dropout_prob`      | `0.3`                                       | Dropout on visual features.                                     |

*(See `ModelArguments` for many more options.)*


## Data config


| Argument                    | Default                     | Description                                               |
| --------------------------- | --------------------------- | --------------------------------------------------------- |
| `dataset_config_file`       | `data/configs/example.yaml` | YAML that groups datasets and assigns sampling weights.   |
| `num_workers`               | `4`                         | Background workers per rank for the PyTorch `DataLoader`. |
| `prefetch_factor`           | `2`                         | Batches pre‑fetched by each worker.                       |
| `max_num_tokens_per_sample` | `16384`                     | Skip raw samples longer than this.                        |
| `max_num_tokens`            | `36864`                     | Hard cap for a packed batch (prevents OOM).               |
| `max_buffer_size`           | `50`                        | Overflow buffer length for oversized samples.             |
| `data_seed`                 | `42`                        | Seed for reproducible shuffling and sampling.             |


## Training config

| Argument                               | Default                | Description                                            |
| -------------------------------------- | ---------------------- | ------------------------------------------------------ |
| `total_steps`                          | `500_000`              | Optimiser steps to run.                                |
| `lr`                                   | `1e-4`                 | Peak learning rate after warm‑up.                      |
| `lr_scheduler`                         | `constant`             | Learning‑rate schedule (`constant` or `cosine`).       |
| `warmup_steps`                         | `2000`                 | Linear warm‑up duration.                               |
| `ema`                                  | `0.9999`               | Exponential moving‑average decay for model weights.    |
| `max_grad_norm`                        | `1.0`                  | Gradient‑clipping threshold.                           |
| `save_every`                           | `2000`                 | Checkpoint frequency (steps).                          |
| `visual_gen / visual_und`              | `True`                 | Enable image generation / understanding branches.      |
| `freeze_llm / freeze_vit / freeze_vae` | `False / False / True` | Freeze selected modules to save VRAM or for ablations. |
| `use_flex`                             | `True` (in example)    | Enable FLEX packing for higher GPU utilisation.        |
| `sharding_strategy`                    | `HYBRID_SHARD`         | FSDP sharding mode.                                    |
| `num_shard`                            | `8`                    | Parameter shards per rank in HYBRID mode.              |

**Distributed‑launch environment variables**

| Var                           | Meaning                           |
| ----------------------------- | --------------------------------- |
| `num_nodes` / `node_rank`     | Multi‑node orchestration indices. |
| `nproc_per_node`              | Number of GPUs per node.          |
| `master_addr` / `master_port` | NCCL rendezvous endpoint.         |


## Logging config


| Argument         | Default               | Description                                          |
| ---------------- | --------------------- | ---------------------------------------------------- |
| `results_dir`    | `results`             | Root directory for logs and metrics.                 |
| `checkpoint_dir` | `results/checkpoints` | Checkpoints are saved here.                          |
| `log_every`      | `10`                  | Steps between console / W\&B logs.                   |
| `wandb_project`  | `bagel`               | Weights & Biases project name.                       |
| `wandb_name`     | `run`                 | Run name inside the project.                         |
| `wandb_offline`  | `False`               | Switch to offline mode (logs locally, sync later).   |
| `wandb_resume`   | `allow`               | Resumption policy if an existing run ID is detected. |

> **Tip**  Export `WANDB_API_KEY` before launching if you want online dashboards.



(.venv) Apptainer> python app_vilex.py 
DEBUG: Initialized embedding table with size 4900
The safetensors archive passed at /home/haoming/Bagel/experiments/results_tune/checkpoints/0000100/ema.safetensors does not contain metadata. Make sure to save your model with the `save_pretrained` method. Defaulting to 'pt' metadata.
You shouldn't move a model that is dispatched using accelerate hooks.
inferencer called
inputlist of call [<PIL.Image.Image image mode=RGB size=2560x1706 at 0x7AD6A9570A90>, 'in a park']
use vilex status False
starting inference
nothinking 
<PIL.Image.Image image mode=RGB size=2560x1706 at 0x7AD6A9570A90>
generation input after vae image preparation {'padded_images': tensor([[[[-0.5608, -0.5686, -0.5765,  ..., -0.7176, -0.7176, -0.7176],
          [-0.5686, -0.5686, -0.5686,  ..., -0.7176, -0.7176, -0.7176],
          [-0.5765, -0.5686, -0.5608,  ..., -0.7176, -0.7176, -0.7176],
          ...,
          [ 0.5451,  0.5216,  0.4980,  ...,  0.1686,  0.1922,  0.2314],
          [ 0.5373,  0.5137,  0.4902,  ...,  0.2157,  0.1922,  0.2000],
          [ 0.5216,  0.5059,  0.4980,  ...,  0.2157,  0.2078,  0.2078]],

         [[-0.4353, -0.4431, -0.4510,  ..., -0.6627, -0.6627, -0.6627],
          [-0.4431, -0.4431, -0.4431,  ..., -0.6627, -0.6627, -0.6627],
          [-0.4510, -0.4431, -0.4353,  ..., -0.6627, -0.6627, -0.6627],
          ...,
          [ 0.7412,  0.7333,  0.7176,  ...,  0.3882,  0.3961,  0.4118],
          [ 0.7333,  0.7255,  0.7098,  ...,  0.4118,  0.4039,  0.4118],
          [ 0.7412,  0.7255,  0.7176,  ...,  0.4039,  0.4039,  0.4039]],

         [[-0.7412, -0.7490, -0.7569,  ..., -0.9294, -0.9294, -0.9294],
          [-0.7490, -0.7490, -0.7490,  ..., -0.9294, -0.9294, -0.9294],
          [-0.7569, -0.7490, -0.7412,  ..., -0.9294, -0.9294, -0.9294],
          ...,
          [ 0.8118,  0.8118,  0.8118,  ...,  0.5922,  0.6078,  0.6392],
          [ 0.8118,  0.8039,  0.8039,  ...,  0.6235,  0.6078,  0.6235],
          [ 0.8039,  0.8039,  0.8039,  ...,  0.6157,  0.6078,  0.6078]]]],
       device='cuda:0'), 'patchified_vae_latent_shapes': [(43, 64)], 'packed_vae_position_ids': tensor([   0,    1,    2,  ..., 2749, 2750, 2751], device='cuda:0'), 'packed_timesteps': tensor([0], device='cuda:0'), 'packed_vae_token_indexes': tensor([   1,    2,    3,  ..., 2750, 2751, 2752], device='cuda:0'), 'packed_text_ids': tensor([151652, 151653], device='cuda:0'), 'packed_text_indexes': tensor([   0, 2753], device='cuda:0'), 'packed_position_ids': tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), 'packed_seqlens': tensor([2754], device='cuda:0', dtype=torch.int32), 'packed_indexes': tensor([   0,    1,    2,  ..., 2751, 2752, 2753], device='cuda:0'), 'packed_key_value_indexes': tensor([], device='cuda:0', dtype=torch.int64), 'key_values_lens': tensor([0], device='cuda:0', dtype=torch.int32)}
generation input after vit {'packed_text_ids': tensor([151652, 151653], device='cuda:0'), 'packed_text_indexes': tensor([ 0, 33], device='cuda:0'), 'vit_token_seqlens': tensor([32], device='cuda:0', dtype=torch.int32), 'packed_vit_tokens': tensor([[-0.5608, -0.4353, -0.7412,  ..., -0.5373, -0.3961, -0.7255],
        [-0.5529, -0.3882, -0.6941,  ..., -0.5294, -0.3569, -0.6627],
        [-0.5529, -0.3647, -0.6314,  ..., -0.5294, -0.3725, -0.6471],
        ...,
        [ 0.3569,  0.5451,  0.7176,  ...,  0.2314,  0.4196,  0.6235],
        [ 0.3176,  0.4824,  0.6784,  ...,  0.2314,  0.4353,  0.6157],
        [ 0.2235,  0.4510,  0.6392,  ...,  0.2078,  0.4039,  0.6078]],
       device='cuda:0'), 'packed_vit_position_ids': tensor([   0,    1,    2,  ..., 3287, 3288, 3289], device='cuda:0'), 'packed_vit_token_indexes': tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
       device='cuda:0'), 'packed_position_ids': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0'), 'packed_seqlens': tensor([34], device='cuda:0', dtype=torch.int32), 'packed_indexes': tensor([2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765,
        2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777,
        2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787],
       device='cuda:0'), 'packed_key_value_indexes': tensor([   0,    1,    2,  ..., 2751, 2752, 2753], device='cuda:0'), 'key_values_lens': tensor([2754], device='cuda:0', dtype=torch.int32)}
all hidden stats debug ()
retuning all hidden states 3290
assigning packed vit tokens indexes to tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
       device='cuda:0') with shape of tensor([[-0.4199, -0.9492,  6.7812,  ...,  7.6875,  4.5625, -3.3438],
        [ 2.1406, -1.6250,  8.7500,  ...,  0.2090,  2.9844, -1.2656],
        [ 2.8750,  2.2656, -0.5078,  ...,  2.8750, -4.3125, -3.5469],
        ...,
        [ 2.0469,  0.9883,  9.5000,  ...,  5.0938, -0.8984,  2.2812],
        [ 0.5859, -0.3770, -0.6680,  ...,  5.8438, -3.9844, -0.0544],
        [ 5.6250,  4.2812,  5.4062,  ...,  3.2188, -1.1484, -2.8125]],
       device='cuda:0', dtype=torch.bfloat16)
the packed query sequence is torch.Size([34, 3584]) actually lookjs like tensor([[ 5.9814e-02,  2.0386e-02, -1.1292e-03,  ...,  1.0254e-02,
         -3.1494e-02,  4.0771e-02],
        [-4.1992e-01, -9.4922e-01,  6.7812e+00,  ...,  7.6875e+00,
          4.5625e+00, -3.3438e+00],
        [ 2.1406e+00, -1.6250e+00,  8.7500e+00,  ...,  2.0898e-01,
          2.9844e+00, -1.2656e+00],
        ...,
        [ 5.8594e-01, -3.7695e-01, -6.6797e-01,  ...,  5.8438e+00,
         -3.9844e+00, -5.4443e-02],
        [ 5.6250e+00,  4.2812e+00,  5.4062e+00,  ...,  3.2188e+00,
         -1.1484e+00, -2.8125e+00],
        [-4.0527e-02, -7.7209e-03, -1.9409e-02,  ...,  5.6763e-03,
          1.1353e-02,  2.4536e-02]], device='cuda:0', dtype=torch.bfloat16)
in a park
fdound string input term in a park
cfg text content {'kv_lens': [2788], 'ropes': [2], 'past_key_values': <modeling.bagel.qwen2_navit.NaiveCache object at 0x7ad6a9570b50>}
calling model to prepare prompts
encoded text ids in a park to [258, 264, 6118] 
generation input {'text_token_lens': tensor([5], device='cuda:0', dtype=torch.int32), 'packed_text_ids': tensor([151644,    258,    264,   6118, 151645], device='cuda:0'), 'packed_text_position_ids': tensor([2, 3, 4, 5, 6], device='cuda:0'), 'packed_text_indexes': tensor([2788, 2789, 2790, 2791, 2792], device='cuda:0'), 'packed_key_value_indexes': tensor([   0,    1,    2,  ..., 2785, 2786, 2787], device='cuda:0'), 'key_values_lens': tensor([2788], device='cuda:0', dtype=torch.int32)}
kv lens [2793]
ropes [7]
past key values <modeling.bagel.qwen2_navit.NaiveCache object at 0x7ad6a95e1450>
generation context {'kv_lens': [2793], 'ropes': [7], 'past_key_values': <modeling.bagel.qwen2_navit.NaiveCache object at 0x7ad6a95f1ed0>}
calling model to prepare prompts
encoded text ids in a park to [258, 264, 6118] 
generation input {'text_token_lens': tensor([5], device='cuda:0', dtype=torch.int32), 'packed_text_ids': tensor([151644,    258,    264,   6118, 151645], device='cuda:0'), 'packed_text_position_ids': tensor([0, 1, 2, 3, 4], device='cuda:0'), 'packed_text_indexes': tensor([0, 1, 2, 3, 4], device='cuda:0'), 'packed_key_value_indexes': tensor([], device='cuda:0', dtype=torch.int64), 'key_values_lens': tensor([0], device='cuda:0', dtype=torch.int32)}
kv lens [5]
ropes [5]
past key values <modeling.bagel.qwen2_navit.NaiveCache object at 0x7ad6a9570bb0>
cfg image context {'kv_lens': [5], 'ropes': [5], 'past_key_values': <modeling.bagel.qwen2_navit.NaiveCache object at 0x7ad6a95f53f0>}
calling gen image with context {'kv_lens': [2793], 'ropes': [7], 'past_key_values': <modeling.bagel.qwen2_navit.NaiveCache object at 0x7ad6a95f1ed0>}
vae latent prepared successfully
image generation input {'packed_text_ids': tensor([151652, 151653], device='cuda:0'), 'packed_text_indexes': tensor([   0, 2753], device='cuda:0'), 'packed_init_noises': tensor([[ 0.1391, -0.1082, -0.7174,  ...,  0.9300,  0.0463,  0.3486],
        [ 1.0300,  0.0132,  0.5492,  ..., -0.1305,  1.1959, -0.6755],
        [-0.5086,  0.9697,  0.1405,  ..., -0.4820, -1.1465,  1.6159],
        ...,
        [-0.6324,  0.0808,  1.2231,  ..., -0.7115,  0.1219, -0.5428],
        [ 1.7063, -0.4001,  0.1870,  ..., -0.7175,  0.8615, -0.4262],
        [-1.2329, -0.5312,  0.1857,  ..., -0.4459,  0.2437, -1.4525]],
       device='cuda:0'), 'packed_vae_position_ids': tensor([   0,    1,    2,  ..., 2749, 2750, 2751], device='cuda:0'), 'packed_vae_token_indexes': tensor([   1,    2,    3,  ..., 2750, 2751, 2752], device='cuda:0'), 'packed_seqlens': tensor([2754], device='cuda:0', dtype=torch.int32), 'packed_position_ids': tensor([7, 7, 7,  ..., 7, 7, 7], device='cuda:0'), 'key_values_lens': tensor([2793], device='cuda:0', dtype=torch.int32), 'packed_indexes': tensor([2793, 2794, 2795,  ..., 5544, 5545, 5546], device='cuda:0'), 'packed_key_value_indexes': tensor([   0,    1,    2,  ..., 2790, 2791, 2792], device='cuda:0')}
start model image generation
start model image generation

=== VISUALIZING PAST KEY VALUES ===
Main context (WITH text) - past_key_values type: <class 'modeling.bagel.qwen2_navit.NaiveCache'>
  Layer 0: Keys torch.Size([2793, 4, 128]), Values torch.Size([2793, 4, 128])
    -> Sequence length: 128 tokens cached
  Layer 1: Keys torch.Size([2793, 4, 128]), Values torch.Size([2793, 4, 128])
    -> Sequence length: 128 tokens cached
  Layer 2: Keys torch.Size([2793, 4, 128]), Values torch.Size([2793, 4, 128])
    -> Sequence length: 128 tokens cached

CFG Text context (WITHOUT current text) - cfg_text_past_key_values:
  Layer 0: Keys torch.Size([2788, 4, 128]) -> 128 tokens cached

CFG Image context - cfg_img_past_key_values:
  Layer 0: Keys torch.Size([5, 4, 128]) -> 128 tokens cached
=====================================

model called to generate image
  0%|                                                                                                                                                                              | 0/49 [00:00<?, ?it/s]time step 0
vae2llm_out: device=cuda:0, dtype=torch.bfloat16
x_t: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-5.593750, max=7.375000, mean=0.314453, std=0.988281
x_t sample values: [0.625, -0.390625, -0.84765625, -0.240234375, 0.57421875, -1.328125, -0.73828125, -0.21484375, -0.30078125, -0.435546875]
packed_sequence: shape=torch.Size([2754, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-5.593750, max=7.375000, mean=0.314453, std=0.988281
packed_text_embedding: shape=torch.Size([2, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-0.269531, max=0.492188, mean=0.000071, std=0.040527
packed_sequence[packed_text_indexes[0]]: shape=torch.Size([3584]), values=[0.059814453125, 0.0203857421875, -0.001129150390625, 0.0400390625, -0.006927490234375, -0.00909423828125, -0.078125, 0.017333984375, -0.04931640625, -0.048095703125]
packed_pos_embed: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-1.000000, max=1.000000, mean=0.392578, std=0.585938
packed_pos_embed sample values: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
packed_timestep_embeds: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-1.648438, max=2.796875, mean=-0.070801, std=0.210938
packed_timestep_embeds sample values: [-0.0015716552734375, -0.0308837890625, 0.12353515625, 0.0101318359375, -0.0169677734375, -0.0791015625, 0.04736328125, 0.08984375, 0.028564453125, 0.01214599609375]
vae2llm_out: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-5.500000, max=5.250000, mean=-0.008240, std=0.804688
packed_vae_token_indexes: shape=torch.Size([2752]), values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
packed_text_indexes: shape=torch.Size([2]), values=[0, 2753]
target_device: cuda:0, target_dtype: torch.bfloat16
  2%|███▍                                                                                                                                                                  | 1/49 [00:01<01:12,  1.51s/it]time step 1
vae2llm_out: device=cuda:0, dtype=torch.bfloat16
x_t: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-5.687500, max=7.343750, mean=0.316406, std=0.984375
x_t sample values: [0.62890625, -0.384765625, -0.8515625, -0.240234375, 0.5703125, -1.3203125, -0.7421875, -0.228515625, -0.27734375, -0.4375]
packed_sequence: shape=torch.Size([2754, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-5.687500, max=7.343750, mean=0.316406, std=0.984375
packed_text_embedding: shape=torch.Size([2, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-0.269531, max=0.492188, mean=0.000071, std=0.040527
packed_sequence[packed_text_indexes[0]]: shape=torch.Size([3584]), values=[0.059814453125, 0.0203857421875, -0.001129150390625, 0.0400390625, -0.006927490234375, -0.00909423828125, -0.078125, 0.017333984375, -0.04931640625, -0.048095703125]
packed_pos_embed: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-1.000000, max=1.000000, mean=0.392578, std=0.585938
packed_pos_embed sample values: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
packed_timestep_embeds: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-1.531250, max=2.656250, mean=-0.068848, std=0.199219
packed_timestep_embeds sample values: [-0.00142669677734375, -0.03076171875, 0.11669921875, 0.01141357421875, -0.0206298828125, -0.07666015625, 0.042724609375, 0.0859375, 0.028076171875, 0.008056640625]
vae2llm_out: shape=torch.Size([2752, 3584]), dtype=torch.bfloat16, device=cuda:0, min=-5.562500, max=5.281250, mean=-0.008850, std=0.800781
packed_vae_token_indexes: shape=torch.Size([2752]), values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
packed_text_indexes: shape=torch.Size([2]), values=[0, 2753]
target_device: cuda:0, target_dtype: torch.bfloat16



(.venv) haoming@dgx4:~/Bagel$ sudo rm -rf /home/haoming/Bagel/experiments/result_tune_new/checkpoints/first_batch
(.venv) haoming@dgx4:~/Bagel$ sudo rm -rf /home/haoming/Bagel/experiments/result_tune_new/checkpoints/debug
(.venv) haoming@dgx4:~/Bagel$ sudo rm -rf /home/haoming/Bagel/experiments/result_tune_new/checkpoints/attention_maps