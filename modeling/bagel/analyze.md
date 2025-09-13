ict_keys(['sequence_length', 'sample_lens', 'packed_text_ids', 'packed_text_indexes', 'packed_position_ids', 'packed_vit_tokens', 'packed_vit_position_ids', 'packed_vit_token_indexes', 'vit_token_seqlens', 'packed_latent_position_ids', 'k', 'padded_images', 'patchified_vae_latent_shapes', 'packed_vae_token_indexes', 'packed_timesteps', 'mse_loss_indexes', 'packed_label_ids', 'ce_loss_indexes', 'num_tokens', 'split_lens', 'attn_modes', 'nested_attention_masks', 'data_indexes'])
========== DEBUG: Forward Pass Input Keys ==========
Key: sequence_length
  Value: 808
Key: packed_text_ids
this is a tensor
  Shape: (23,)
  Values: [151644     32   3691  20638    374   1787    389    264  22360  18010
    448    264   3691  20638   1142    389   1909    315    432     13] ...
Key: packed_text_indexes
this is a tensor
  Shape: (23,)
  Values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] ...
Key: sample_lens
this is a list
  Length: 1
  First 10: [808] ...
Key: packed_position_ids
this is a tensor
  Shape: (805,)
  Values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] ...
Key: nested_attention_masks
this is a list
  Length: 1
  First 10: [tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
        [0., 0., -inf,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])] ...
Key: split_lens
this is a list
  Length: 4
  First 10: [21, 1, 2, 784] ...
Key: attn_modes
this is a list
  Length: 4
  First 10: ['causal', 'causal', 'causal', 'noise'] ...
Key: ce_loss_indexes
this is a tensor
  Shape: (20,)
  Values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] ...
Key: packed_label_ids
this is a tensor
  Shape: (1, 20)
  First row: [1.51644e+05 3.20000e+01 3.69100e+03 2.06380e+04 3.74000e+02 1.78700e+03
 3.89000e+02 2.64000e+02 2.23600e+04 1.80100e+04 4.48000e+02 2.64000e+02
 3.69100e+03 2.06380e+04 1.14200e+03 3.89000e+02 1.90900e+03 3.15000e+02
 4.32000e+02 1.30000e+01] ...
Key: packed_vit_tokens
this is a tensor
  Shape: (1024, 588)
  First row: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] ...
Key: packed_vit_token_indexes
this is a tensor
  Shape: (1,)
  Values: [20] ...
Key: packed_vit_position_ids
this is a tensor
  Shape: (1024,)
  Values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] ...
Key: vit_token_seqlens
this is a tensor
  Shape: (1,)
  Values: [1024] ...
Key: padded_latent
  Value: None
Key: patchified_vae_latent_shapes
this is a list
  Length: 1
  First 10: [[28, 28]] ...
Key: packed_latent_position_ids
this is a tensor
  Shape: (784,)
  Values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] ...
Key: packed_vae_token_indexes
this is a tensor
  Shape: (784,)
  Values: [23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42] ...
Key: packed_timesteps
this is a tensor
  Shape: (784,)
  Values: [567 567 567 567 567 567 567 567 567 567 567 567 567 567 567 567 567 567
 567 567] ...
Key: mse_loss_indexes
this is a tensor
  Shape: (784,)
  Values: [23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42] ...
Key: num_tokens
  Value: 808
Key: k
  Value: 0
====================================================
the input -- text ids
torch.Size([23])
tensor([151644,     32,   3691,  20638,    374,   1787,    389,    264,  22360,
         18010], device='cuda:0')
initializing overall sequence with 
torch.Size([808, 896])
length of indexes
torch.Size([23])
using atttention mask
torch.Size([808, 808])
second step, vilex tokens
