# Vision-Language Model from Scratch in PyTorch

Build an end-to-end multimodal vision-language model that ingests an image plus a text prompt and autoregressively generates a caption. You will implement every component from raw tensor operations: a ViT image encoder, a vision-to-language projector, a causal text decoder, multimodal fusion, the training loop, and sampling-based generation.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** split_image_into_patches
- [x] **2.** flatten_patches
- [x] **3.** linear_projection
- [x] **4.** project_patches_to_embeddings
- [x] **5.** prepend_class_token
- [x] **6.** add_position_embeddings
- [x] **7.** compute_attention_scores
- [x] **8.** scale_attention_scores
- [x] **9.** apply_attention_mask
- [x] **10.** attention_softmax
- [x] **11.** attention_context
- [x] **12.** scaled_dot_product_attention
- [x] **13.** split_into_heads
- [x] **14.** merge_heads
- [x] **15.** project_qkv
- [x] **16.** split_qkv_into_heads
- [x] **17.** multi_head_attention_scores
- [x] **18.** merge_and_output_project
- [x] **19.** multi_head_self_attention
- [x] **20.** gelu_activation
- [x] **21.** mlp_first_layer
- [x] **22.** mlp_second_layer

---

Built on Deep-ML.
