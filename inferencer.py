# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

# Ensure we run on CUDA and expose a device constant for the inferencer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA is required for inference. No CUDA available.")

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        # keep references to inputs
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

        # bind device to this instance and try to place modules on CUDA
        self.device = device
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass
        try:
            self.vae_model = self.vae_model.to(self.device)
        except Exception:
            pass

    # helper to move nested input dict/list tensors to device
    def _move_inputs_to_device(self, inputs):
        if inputs is None:
            return inputs
        if isinstance(inputs, dict):
            for k, v in list(inputs.items()):
                if torch.is_tensor(v):
                    try:
                        inputs[k] = v.to(self.device)
                    except Exception:
                        pass
                elif isinstance(v, (list, tuple)):
                    new_list = []
                    for x in v:
                        if torch.is_tensor(x):
                            try:
                                new_list.append(x.to(self.device))
                            except Exception:
                                new_list.append(x)
                        else:
                            new_list.append(x)
                    inputs[k] = new_list
        return inputs

    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        print("calling model to prepare prompts")
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        # move any tensors in generation_input to the instance device
        generation_input = self._move_inputs_to_device(generation_input)
        print(f"generation input {generation_input}")
        print(f"kv lens {kv_lens}")
        print(f"ropes {ropes}")
        print(f"past key values {past_key_values}")
        # ensure past_key_values are on the correct device
        try:
            past_key_values = past_key_values.to(self.device)
        except Exception:
            pass
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_vilex(self, image, gen_context, text_ids=None):
        # Process image through VILEX (treated as text-like tokens)
        
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        print("calling model to prepare VILEX from image")
        generation_input, kv_lens, ropes = self.model.prepare_vilex_from_image(
            curr_kvlens=kv_lens[0],
            curr_position_id=ropes[0], 
            image=image,
            transforms=self.vit_transform,  # Use VIT transform for image processing
            new_token_ids=self.new_token_ids,
            text_ids=text_ids,
            tokenizer=self.tokenizer,
        )
        # move any tensors in generation_input to the instance device
        generation_input = self._move_inputs_to_device(generation_input)
        print(f"VILEX generation input {generation_input}")
        print(f"kv lens {kv_lens}")
        print(f"ropes {ropes}")
        print(f"past key values {past_key_values}")
        # ensure past_key_values are on the correct device
        try:
            past_key_values = past_key_values.to(self.device)
        except Exception:
            pass
        past_key_values = self.model.forward_cache_update_vilex(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._move_inputs_to_device(generation_input)
            print(f"generation input after vae image preparation {generation_input}")
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
            
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._move_inputs_to_device(generation_input)
            print(f"generation input after vit {generation_input}")
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        # move inputs to device
        generation_input = self._move_inputs_to_device(generation_input)
        #generation_input["packed_text_ids"] = gen_context["packed_text_ids"]
        #generation_input["packed_text_indexes"] = gen_context["packed_text_indexes"]
        print("vae latent prepared successfully")
        print(f"image generation input {generation_input}")
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_text = self._move_inputs_to_device(generation_input_cfg_text)

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_img = self._move_inputs_to_device(generation_input_cfg_img)

        # ensure past key values are on device
        try:
            past_key_values = past_key_values.to(self.device)
        except Exception:
            pass

        print("start model image generation")
        print("start model image generation")

        # Add this visualization code:
        print("\n=== VISUALIZING PAST KEY VALUES ===")
        print(f"Main context (WITH text) - past_key_values type: {type(past_key_values)}")
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            for layer_idx in range(min(3, len(past_key_values.key_cache))):  # Show first 3 layers
                if past_key_values.key_cache[layer_idx] is not None:
                    k_shape = past_key_values.key_cache[layer_idx].shape
                    v_shape = past_key_values.value_cache[layer_idx].shape
                    print(f"  Layer {layer_idx}: Keys {k_shape}, Values {v_shape}")
                    print(f"    -> Sequence length: {k_shape[2]} tokens cached")

        print(f"\nCFG Text context (WITHOUT current text) - cfg_text_past_key_values:")
        if hasattr(cfg_text_past_key_values, 'key_cache') and cfg_text_past_key_values.key_cache[0] is not None:
            cfg_k_shape = cfg_text_past_key_values.key_cache[0].shape
            print(f"  Layer 0: Keys {cfg_k_shape} -> {cfg_k_shape[2]} tokens cached")
        else:
            print("  Empty cache (no tokens)")

        print(f"\nCFG Image context - cfg_img_past_key_values:")
        if hasattr(cfg_img_past_key_values, 'key_cache') and cfg_img_past_key_values.key_cache[0] is not None:
            cfg_img_k_shape = cfg_img_past_key_values.key_cache[0].shape  
            print(f"  Layer 0: Keys {cfg_img_k_shape} -> {cfg_img_k_shape[2]} tokens cached")
        else:
            print("  Empty cache (no tokens)")

        print("=====================================\n")
        
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        # ensure latent is on device before decoding
        try:
            unpacked_latent = [l.to(self.device) if torch.is_tensor(l) else l for l in unpacked_latent]
        except Exception:
            pass
        print("start devoding image")
        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        # ensure latent is on the same device as VAE
        try:
            latent = latent.to(self.device)
        except Exception:
            pass

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = self._move_inputs_to_device(generation_input)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        use_vilex = False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:
        print(f"use vilex status {use_vilex}")

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        print("starting inference")


        if use_vilex:
            print("using vilex pathway")
            print(input_lists[0])
            cfg_vilex_context = deepcopy(gen_context)
            input_term = input_lists[0]
            text = input_lists[1]
            gen_context = self.update_context_vilex(input_term, gen_context,text_ids = self.tokenizer.encode(text))
            cfg_img_context = self.update_context_text(text, cfg_img_context)
            print(gen_context)
            img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

            output_list.append(img)

            return output_list
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                print("thinking")
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
                print("updated thinking context")
            else:
                print("nothinking ")

            for input_term in input_lists:
                print(input_term)
                if isinstance(input_term, str):
                    print(f"fdound string input term {input_term}")
                    cfg_text_context = deepcopy(gen_context)
                    print(f"cfg text content {cfg_text_context}")
                    gen_context = self.update_context_text(input_term, gen_context)
                    print(f"generation context {gen_context}")
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                    print(f"cfg image context {cfg_img_context}")

                elif isinstance(input_term, Image.Image):
                    if use_vilex:
                        # For VILEX: Process image to get VILEX tokens (treated as text)
                        gen_context = self.update_context_vilex(input_term, gen_context)
                        # Both CFG contexts should also include VILEX tokens
                        cfg_text_context = deepcopy(gen_context)
                        cfg_img_context = deepcopy(gen_context)
                        # Set image shapes for generation
                        input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                        image_shapes = input_term.size[::-1]
                    else:
                        # Original path: use VAE encoding
                        input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                        gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                        image_shapes = input_term.size[::-1]
                        cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                print("understanding task")
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)
                print(f"calling gen image with context {gen_context}")

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = "generate an image of", 
        use_vilex: Optional[str] = False, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}
        print("inferencer called")
        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)
        print(f"inputlist of call {input_list}")
        output_list = self.interleave_inference(input_list, use_vilex=use_vilex,**kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
                print("returned image")
            elif isinstance(i, str):
                output_dict['text'] = i
                print("returned text")
        return output_dict


