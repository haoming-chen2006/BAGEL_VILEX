---
license: apache-2.0
base_model:
- Qwen/Qwen2.5-7B-Instruct
pipeline_tag: any-to-any
library_name: bagel-mot
---


<p align="left">
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="480"/>
</p>


# 🥯 BAGEL • Unified Model for Multimodal Understanding and Generation



<p align="left">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white" style="display: inline-block; vertical-align: middle;"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red" style="display: inline-block; vertical-align: middle;"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://github.com/bytedance-seed/BAGEL" target="_blank" style="margin: 2px;">
      <img 
        alt="Github" src="https://img.shields.io/badge/BAGEL-Codebase-536af5?color=536af5&logo=github" style="display: inline-block; vertical-align: middle;"
        alt="BAGEL Codebase"
      />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=white" style="display: inline-block; vertical-align: middle;"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://discord.com/invite/Z836xxzy">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-green?logo=discord&logoColor=white" style="display: inline-block; vertical-align: middle;"
      alt="BAGEL Discord"
    />
  </a>

  
</p>


> We present **BAGEL**, an open‑source multimodal foundation model with 7B active parameters (14B total) trained on large‑scale interleaved multimodal data. BAGEL outperforms the current top‑tier open‑source VLMs like Qwen2.5-VL and InternVL-2.5 on standard multimodal understanding leaderboards, and delivers text‑to‑image quality that is competitive with strong specialist generators such as SD3.
Moreover, BAGEL demonstrates superior qualitative results in classical image‑editing scenarios than the leading open-source models. More importantly, it extends to free-form visual manipulation, multiview synthesis, and world navigation, capabilities that constitute "world-modeling" tasks beyond the scope of previous image-editing models.


This repository hosts the model weights for **BAGEL**. For installation, usage instructions, and further documentation, please visit our [GitHub repository](https://github.com/bytedance-seed/BAGEL).



<p align="left"><img src="https://github.com/ByteDance-Seed/Bagel/raw/main/assets/teaser.webp" width="80%"></p>






## 🧠 Method
BAGEL adopts a Mixture-of-Transformer-Experts (MoT) architecture to maximize the model’s capacity to learn from richly diverse multimodal information. Following the same principle of capacity maximization, it utilizes two separate encoders to capture pixel-level and semantic-level features of an image. The overall framework follows a Next Group of Token Prediction paradigm, where the model is trained to predict the next group of language or visual tokens as a compression target.

BAGEL scales MoT’s capacity through Pre-training, Continued Training, and Supervised Finetuning on trillions of interleaved multimodal tokens spanning language, image, video, and web data. It surpasses open models on standard understanding and generation benchmarks and demonstrates advanced in-context multimodal abilities like free-form image editing, future frame prediction, 3D manipulation, world navigation, and sequential reasoning.

<p align="left"><img src="https://github.com/ByteDance-Seed/Bagel/raw/main/assets/arch.png" width="50%"></p>


## 🌱 Emerging Properties
<p align="left"><img src="https://github.com/ByteDance-Seed/Bagel/raw/main/assets/emerging_curves.png" width="50%"></p>

As we scale up BAGEL’s pretraining with more multimodal tokens, we observe consistent performance gains across understanding, generation, and editing tasks. Different capabilities emerge at distinct training stages—multimodal understanding and generation appear early, followed by basic editing, while complex, intelligent editing emerges later. This staged progression suggests an emergent pattern, where advanced multimodal reasoning builds on well-formed foundational skills. Ablation studies further show that combining VAE and ViT features significantly improves intelligent editing, underscoring the importance of visual-semantic context in enabling complex multimodal reasoning and further supporting its role in the emergence of advanced capabilities.



## 📊 Benchmarks
### 1. Visual Understanding
| Model | MME ↑ | MMBench ↑ |   MMMU ↑ | MM-Vet ↑ | MathVista ↑ |
| ------------------- | ----------: | ----------: | -------: | -------: | ----------: |
| Janus-Pro-7B        | -  |     79.2 |     41.0 |     50.0 |           – |
| Qwen2.5-VL-7B      | 2347    |   83.5 | **58.6** |     67.1 |           68.2 |
| **BAGEL**    | **2388**  |  **85.0** |     55.3 | **67.2** |    **73.1** |
### 2. Text-to-Image Generation · GenEval
| Model        | Overall ↑ |
| ------------ | --------- |
| FLUX-1-dev   | 0.82      |
| SD3-Medium   | 0.74      |
| Janus-Pro-7B | 0.80      |
| **BAGEL**    | **0.88**  |
### 3. Image Editing
| Model         | GEdit-Bench-EN (SC) ↑ | GEdit-Bench-EN (PQ) ↑ | GEdit-Bench-EN (O) ↑ | IntelligentBench ↑ |
| ------------- | --------------------- | --------------------- | ------------------- | ------------------ |
| Step1X-Edit   | 7.09                  | 6.76                  | **6.70**            | 14.9               |
| Gemini-2-exp. | 6.73                  | 6.61                  | 6.32                | **57.6**           |
| **BAGEL**     | **7.36**              | **6.83**              | 6.52                | 44.0               |
| **BAGEL+CoT** | –                   | –                     | –                   | 55.3               |

## License
BAGEL is licensed under the Apache 2.0 license. It is finetuned from [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [siglip-so400m-14-384-flash-attn2](https://huggingface.co/HuggingFaceM4/siglip-so400m-14-384-flash-attn2) model, and uses the [FLUX.1-schnell VAE model](https://huggingface.co/black-forest-labs/FLUX.1-schnell), all under Apache 2.0.

## ✍️ Citation
```bibtex
@article{deng2025bagel,
  title   = {Emerging Properties in Unified Multimodal Pretraining},
  author  = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
  journal = {arXiv preprint arXiv:2505.14683},
  year    = {2025}
}
```