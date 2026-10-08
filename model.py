"""
Vision-Language Model from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_image_into_patches
import torch

def split_image_into_patches(image, patch_size):
    """Split an image tensor (B, C, H, W) into a sequence of (patch_size, patch_size) patches.

    Returns a tensor of shape (B, num_patches, C, patch_size, patch_size) in row-major order.
    """
    # TODO: split the (B, C, H, W) image into (B, num_patches, C, patch_size, patch_size).
    B, C, H, W = image.shape
    grid_h = H//patch_size
    grid_w = W//patch_size

    return image.reshape((B, C, grid_h, patch_size, grid_w, patch_size)).permute(0, 2, 4, 1, 3, 5).reshape((B, grid_h*grid_w, C, patch_size, patch_size))

# Step 2 - flatten_patches
def flatten_patches(patches):
    # TODO: flatten each patch's channel and spatial dims into one vector, keep (B, N) leading dims.
    B, N, C, ph, pw = patches.shape

    return patches.reshape((B, N, C * ph * pw))

# Step 3 - linear_projection
import torch

def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    # TODO: compute the affine map y = x @ weight.T + bias
    return x @ weight.T + bias

# Step 4 - project_patches_to_embeddings
import torch

def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    # TODO: Linearly project flattened image patches into the ViT embedding dimension.
    return linear_projection(flat_patches, patch_proj_weight, patch_proj_bias)

# Step 5 - prepend_class_token
import torch

def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    # TODO: prepend the [CLS] token to every sequence in the batch
    B, N, D = patch_embeddings.shape
    return torch.cat([class_token.expand((B, 1, D)), patch_embeddings], dim=1)

# Step 6 - add_position_embeddings
import torch

def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    # TODO: combine tokens (B, S, D) with position_embeddings (1, S, D) via broadcasting.
    return tokens + position_embeddings

# Step 7 - compute_attention_scores
import torch

def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    # TODO: compute the raw attention scores as Q times K-transpose
    return q @ k.transpose(-1, -2)

# Step 8 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    # TODO: Divide raw attention scores by a constant derived from d_head.
    scale = 1.0/(d_head**0.5)
    return scores * scale

# Step 9 - apply_attention_mask
def apply_attention_mask(scores, mask):
    # TODO: add an additive mask (0 = allowed, -inf = blocked) to attention scores.
    if mask is not None:
        scores += mask
    return scores

# Step 10 - attention_softmax
import torch

def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    # TODO: convert masked attention scores into normalized weights over the key axis
    shifted = torch.exp(masked_scores - masked_scores.max(dim=-1, keepdim=True).values)
    return shifted/(shifted.sum(axis=-1, keepdim=True))

# Step 11 - attention_context
import torch

def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    # TODO: return a tensor of shape (..., Sq, d_head) from attn_weights and v
    return attn_weights @ v

# Step 12 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    # TODO: compose the five attention primitives into a single forward pass.
    d_head = q.shape[-1]
    scores = compute_attention_scores(q, k)
    scores = scale_attention_scores(scores, d_head)
    scores = apply_attention_mask(scores, mask)

    attn = attention_softmax(scores)
    return attention_context(attn, v)

# Step 13 - split_into_heads
import torch

def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    # TODO: split the last dim into (num_heads, d_head) and move heads next to batch
    B, S, d_model = x.shape
    d_head = d_model//num_heads
    return x.reshape((B, S, num_heads, d_head)).transpose(1, 2)

# Step 14 - merge_heads
import torch

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    # TODO: merge the multi-head dimension back into the model dimension
    B, H, S, D = x.shape
    return x.transpose(1,2).reshape((B, S, H*D))

# Step 15 - project_qkv
def project_qkv(x, wq, bq, wk, bk, wv, bv):
    # TODO: project x into separate query, key, and value tensors using three linear layers.
    return linear_projection(x, wq, bq), linear_projection(x, wk, bk), linear_projection(x, wv, bv)

# Step 16 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: reshape q, k, v from (B, S, d_model) into (B, num_heads, S, d_head) each
    return split_into_heads(q, num_heads), split_into_heads(k, num_heads), split_into_heads(v, num_heads)

# Step 17 - multi_head_attention_scores
import torch

def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    # TODO: run scaled dot-product attention across the head axis
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 18 - merge_and_output_project
import torch

def merge_and_output_project(context_heads, wo, bo):
    """Merge heads back to d_model and apply the output projection."""
    # TODO: merge multi-head context to (B, S, d_model) then apply linear projection with wo, bo
    out = merge_heads(context_heads)
    return linear_projection(out, wo, bo)

# Step 19 - multi_head_self_attention
import torch

def multi_head_self_attention(x, params, num_heads, mask=None):
    """Run full multi-head self-attention: QKV proj, head split, attention, merge, output proj."""
    # TODO: compose project_qkv, split_qkv_into_heads, multi_head_attention_scores, merge_and_output_project.
    q, k, v = project_qkv(x, params['wq'], params['bq'], params['wk'], params['bk'], params['wv'], params['bv'])
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    context_heads = multi_head_attention_scores(q_h, k_h, v_h, mask)
    return merge_and_output_project(context_heads, params['wo'], params['bo'])

# Step 20 - gelu_activation
import torch

def gelu_activation(x):
    """Apply the exact (erf-based) GELU activation elementwise to x."""
    # TODO: implement GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    # return x * 0.5 * (1 + torch.tanh((2/torch.pi)**0.5 * (x + 0.044715*(x**3))))
    return x * 0.5 * (1 + torch.erf(x/(2**0.5)))

# Step 21 - mlp_first_layer
import torch

def mlp_first_layer(x, w1, b1):
    """Apply the first linear layer of the MLP block followed by GELU."""
    # TODO: project x to the feed-forward dimension and apply GELU
    return gelu_activation(linear_projection(x, w1, b1))

# Step 22 - mlp_second_layer
import torch

def mlp_second_layer(h, w2, b2):
    # TODO: project the MLP hidden activations back down to d_model using w2 and b2
    return linear_projection(h, w2, b2)

# Step 23 - mlp_block
import torch

def mlp_block(x, params):
    """Two-layer position-wise MLP with GELU between the layers."""
    # TODO: Assemble the position-wise two-layer MLP block with GELU between layers.
    return mlp_second_layer(mlp_first_layer(x, params['w1'], params['b1']), params['w2'], params['b2'])

# Step 24 - compute_layernorm_stats
import torch

def compute_layernorm_stats(x, eps=1e-5):
    # TODO: return (mean, var) along the last dim, each with shape (..., 1).
    return x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True, correction=0)

# Step 25 - layer_norm
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize the last dim of x and apply learnable scale gamma and shift beta
    mean, var = compute_layernorm_stats(x)
    return gamma * ((x-mean)/torch.sqrt(var + eps)) + beta

# Step 26 - residual_add
import torch

def residual_add(residual, sublayer_output):
    """Add residual skip connection to a sublayer's output."""
    # TODO: return the element-wise sum of residual and sublayer_output
    return residual + sublayer_output

# Step 27 - pre_norm_sublayer
import torch

def pre_norm_sublayer(x, gamma, beta, sublayer_fn):
    """Apply pre-norm: LN(x) -> sublayer -> add residual x."""
    # TODO: layer-normalize x, run sublayer_fn on it, then add the residual
    return residual_add(x, sublayer_fn(layer_norm(x, gamma, beta)))

# Step 28 - vision_encoder_block
import torch

def vision_encoder_block(x, block_params, num_heads):
    # TODO: pre-norm MHSA sublayer, then pre-norm MLP sublayer, both with residuals.
    mhsa_fn = lambda x: multi_head_self_attention(x, block_params['attn'], num_heads)
    mlp_fn = lambda x: mlp_block(x, block_params['mlp'])

    attn_out = pre_norm_sublayer(x, block_params['ln1_gamma'], block_params['ln1_beta'], mhsa_fn)
    return pre_norm_sublayer(attn_out, block_params['ln2_gamma'], block_params['ln2_beta'], mlp_fn)

# Step 29 - vision_encoder
import torch

def vision_encoder(patch_sequence, encoder_params, num_heads):
    """Stack ViT encoder blocks then apply a final layer norm to the patch sequence."""
    # TODO: run patch_sequence through every block in encoder_params['blocks'], then final layer norm.
    x = patch_sequence
    for block_params in encoder_params['blocks']:
        x = vision_encoder_block(x, block_params, num_heads)
    
    return layer_norm(x, encoder_params['final_ln_gamma'], encoder_params['final_ln_beta'])

# Step 30 - extract_patch_features
import torch

def extract_patch_features(encoder_output):
    """Drop the [CLS] token from a ViT encoder output of shape (B, num_patches+1, d_model)."""
    # TODO: drop the class token and return only patch feature tokens
    return encoder_output[:, 1:, :]

# Step 31 - projector_first_layer
import torch

def projector_first_layer(patch_features, w1, b1):
    # TODO: apply the first projector linear layer followed by GELU
    return gelu_activation(patch_features @ w1 + b1)

# Step 32 - projector_second_layer
import torch

def projector_second_layer(hidden, w2, b2):
    """Map hidden activations (N, D_hidden) into the language space (N, D_lang)."""
    # TODO: apply the second linear layer of the projector (no activation).
    return hidden @ w2 + b2

# Step 33 - vision_language_projector
import torch

def vision_language_projector(patch_features, params):
    """Map (N, D_vision) patch features to (N, D_lang) image tokens."""
    # TODO: chain the two projector layers using params 'w1','b1','w2','b2'.
    a1 = projector_first_layer(patch_features, params['w1'], params['b1'])
    return projector_second_layer(a1, params['w2'], params['b2'])

# Step 34 - build_token_vocabulary
def build_token_vocabulary(texts, image_token='<image>', pad_token='<pad>'):
    # TODO: Build a whitespace token-to-id vocabulary with pad at 0 and image token at 1.
    vocab = {pad_token:0, image_token:1}
    vocab_set = set()

    for text in texts:
        for tok in text.strip().split():
            if tok not in vocab_set and tok not in (pad_token, image_token):
                vocab_set.add(tok)

    vocab.update({tok: i+2 for i, tok in enumerate(sorted(vocab_set))})
    return vocab

# Step 35 - encode_text_to_ids
def encode_text_to_ids(text, vocab):
    # TODO: split text on whitespace and map each token to its vocab id
    return [vocab[tok] for tok in text.strip().split()]

