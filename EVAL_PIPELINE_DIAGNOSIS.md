# Evaluation Pipeline Diagnosis for BAGEL + VILEX

## 1. Training Baseline

The dataset packer establishes the canonical ordering of modalities inside a training sample. Text is prefixed with `<|bos|>`, the ViT/VILEX block follows immediately, and only then do we append `<|eos|>` and `<|vision_start|>` before entering the diffusion block. This structure is visible in the packer:

```python
# data/vilex_dataset.py
packed_vit_token_indexes.extend(range(curr, curr + num_queries))
packed_vit_tokens.append(vit_patches)
curr += num_queries
...
packed_text_ids.extend([self.eos_token_id])
packed_text_indexes.append(curr)
...
packed_text_ids.append(self.start_of_image)
packed_text_indexes.append(curr)
```
【F:data/vilex_dataset.py†L233-L259】

The same routine later places diffusion tokens and the closing `<|vision_end|>` marker, ensuring RoPE positions and index buffers are synchronised across modalities.【F:data/vilex_dataset.py†L268-L316】

A second quirk to note is that training currently hard-clamps the number of VILEX queries to 32, which is consistent with the projector configuration but worth monitoring when experimenting with different projector heads.【F:data/vilex_dataset.py†L224-L236】

## 2. Problems Observed in Evaluation

### 2.1 Sequence assembly lost ordering information

The previous evaluation helper collapsed the entire (text + VILEX + control tokens) bundle into a single `combined_embeddings` tensor before handing it to the language model cache updater. Because no per-token indexes were tracked, VILEX outputs were appended after the control tokens, yielding a layout that diverged from the training packer described above. This manifested as duplicated text spans and an inconsistent `<|vision_start|>` boundary in the cache.

### 2.2 Cache update used causal attention for VILEX

Inference reused the text-only cache updater and forced `is_causal=True`, so the vision block could not attend bidirectionally to the prompt tokens even though the training masks allow it. This made the interleaved cache structurally different from the training-time sparse attention pattern.

### 2.3 Guidance contexts missed the text + VILEX alignment

`update_context_vilex` fed scalar `curr_kvlens`/`curr_position_id` values into the model, while the rest of the inference helpers operate on single-element lists. This mismatch caused the KV indices and RoPE offsets to fall out of sync once the context grew beyond the very first call.

## 3. Fixes Introduced

### 3.1 Deterministic token-by-token packing

`prepare_vilex_from_image` now mirrors the dataset routine: it records every text, VILEX, and control token index explicitly, and it emits the matching RoPE positions so the downstream cache updater reconstructs the exact training layout.

```python
packed_text_ids.append(token)
packed_text_indexes.append(query_cursor)
...
for _ in range(num_vilex_tokens):
    packed_vilex_token_indexes.append(query_cursor)
    packed_position_ids.append(rope_cursor)
...
packed_text_ids.append(start_image_id)
packed_text_indexes.append(query_cursor)
```
【F:modeling/bagel/bagel_vilex.py†L488-L586】

The function also keeps the projector output (`packed_vilex_embeddings`) on the model device with shape `(num_vilex_tokens, hidden_size)`, so the hidden-state layout matches the `num_output_tokens` expected by the VILEX connector.【F:modeling/bagel/bagel_vilex.py†L517-L579】

### 3.2 Non-causal cache update for the vision block

`forward_cache_update_vilex` now rebuilds the packed sequence with both the text embeddings and the projector outputs before calling the language model. Crucially, it switches to `is_causal=False`, aligning the inference attention pattern with the training sparse mask and preventing the VILEX queries from being masked out of the prompt history.

```python
text_embeddings = self.language_model.model.embed_tokens(packed_text_ids)
packed_sequence = text_embeddings.new_zeros((sequence_length, self.hidden_size))
packed_sequence[packed_text_indexes] = text_embeddings
...
packed_sequence[packed_vilex_indexes] = packed_vilex_embeddings
...
output = self.language_model.forward_inference(
    packed_query_sequence=packed_sequence,
    query_lens=packed_seqlens,
    packed_query_position_ids=packed_position_ids,
    ...,
    is_causal=False,
)
```
【F:modeling/bagel/bagel_vilex.py†L675-L749】

### 3.3 Context bookkeeping consistency

`update_context_vilex` now threads the single-element KV-length and RoPE arrays straight through to the model helpers. This keeps the cached positions monotonically increasing, the same way the training loader constructs them.【F:inferencer.py†L117-L145】

## 4. Remaining Watch-points

* The dataset still trims the final entry of `packed_position_ids` (`packed_position_ids[:-1]`), which is worth revisiting once the attention masks are audited for multi-sample batching.【F:data/vilex_dataset.py†L336-L344】
* Tail-drop is disabled in evaluation by design, but the hard-coded `num_queries = 32` comment suggests this should be surfaced through configuration if we want to experiment with different projector capacities.【F:data/vilex_dataset.py†L224-L236】

Overall, the evaluation helpers now rebuild the exact `(text -> VILEX -> <|eos|> -> <|vision_start|>)` sequence that training expects, ensuring the cached attention states and positional buffers line up with the learned pattern.
