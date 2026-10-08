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

---

Built on Deep-ML.
