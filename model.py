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
    B, S, d_model = x.shape if len(x.shape) == 3 else (1, *x.shape)
    d_head = d_model//num_heads
    heads = x.reshape((B, S, num_heads, d_head)).transpose(1, 2)

    if len(x.shape) < 3:
        heads = heads[0]
    return heads

# Step 14 - merge_heads
import torch

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    # TODO: merge the multi-head dimension back into the model dimension
    flag = False
    if len(x.shape) < 4:
        x = x[None, ...]
        flag = True

    B, H, S, D = x.shape
    merged = x.transpose(1,2).reshape((B, S, H*D))
    if flag:
        merged = merged[0]

    return merged

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

# Step 36 - embed_token_ids
import torch

def embed_token_ids(token_ids, embedding_matrix):
    """Look up embedding vectors for each token id.

    Args:
        token_ids: Long tensor of shape (T,) with values in [0, V).
        embedding_matrix: Tensor of shape (V, D_lang).

    Returns:
        Tensor of shape (T, D_lang).
    """
    # TODO: select the row of embedding_matrix for each token id
    return embedding_matrix[token_ids]

# Step 37 - add_text_position_embeddings
import torch

def add_text_position_embeddings(text_embeddings, position_embeddings):
    """Add learnable position embeddings to text token embeddings.

    text_embeddings: (T, D_lang)
    position_embeddings: (T_max, D_lang) with T_max >= T
    returns: (T, D_lang)
    """
    # TODO: add the first T rows of position_embeddings to text_embeddings
    T = text_embeddings.shape[0]
    return text_embeddings + position_embeddings[:T]

# Step 38 - find_image_placeholder_positions
import torch

def find_image_placeholder_positions(token_ids, image_token_id):
    """Return a list of indices where token_ids == image_token_id."""
    # TODO: scan token_ids and return every position whose value equals image_token_id
    seq_len = len(token_ids)
    return torch.arange(seq_len, dtype=torch.int64)[token_ids == image_token_id].tolist()

# Step 39 - insert_image_tokens
import torch

def insert_image_tokens(text_embeddings, image_tokens, placeholder_position):
    """Splice image tokens into the text embedding sequence at the placeholder slot."""
    # TODO: replace text_embeddings[placeholder_position] with the N image_tokens rows
    return torch.cat([text_embeddings[:placeholder_position], image_tokens, text_embeddings[placeholder_position+1:]])

# Step 40 - build_multimodal_embeddings
import torch

def build_multimodal_embeddings(token_ids, image_tokens, embedding_matrix, position_embeddings, image_token_id):
    # TODO: build fused multimodal embeddings by embedding text, adding positions, and splicing image tokens.
    text_embeddings = embed_token_ids(token_ids, embedding_matrix)
    text_embeddings = add_text_position_embeddings(text_embeddings, position_embeddings)
    placeholder_ids = find_image_placeholder_positions(token_ids, image_token_id)
    new_ids = token_ids.clone()

    while placeholder_ids:
        text_embeddings = insert_image_tokens(text_embeddings, image_tokens, placeholder_ids[0])
        new_ids[placeholder_ids[0]] = image_token_id - 1000
        placeholder_ids = find_image_placeholder_positions(new_ids, image_token_id)

    return text_embeddings

# Step 41 - build_label_tensor
import torch

def build_label_tensor(token_ids, image_token_id, pad_token_id, num_image_tokens, ignore_index=-100):
    """Build the label tensor aligned to the fused multimodal sequence."""
    # TODO: expand image placeholders, mask image and pad positions with ignore_index
    placeholder_ids = find_image_placeholder_positions(token_ids, image_token_id)
    image_tokens = torch.full((num_image_tokens,), ignore_index, dtype=torch.int64)

    token_ids = torch.where(token_ids==pad_token_id, torch.tensor(ignore_index), token_ids)

    while placeholder_ids:
        token_ids = insert_image_tokens(token_ids, image_tokens, placeholder_ids[0])
        placeholder_ids = find_image_placeholder_positions(token_ids, image_token_id)

    return token_ids

# Step 42 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) additive causal mask: 0 on/under diag, -inf above."""
    # TODO: build a lower-triangular additive mask with 0 allowed and -inf blocked
    return torch.triu(torch.full((seq_len, seq_len), -torch.inf), diagonal=1)

# Step 43 - decoder_block
def decoder_block(x, params, causal_mask):
    # TODO: run a pre-norm masked self-attention sublayer then a pre-norm MLP sublayer over x.
    mhsa_fn = lambda x: multi_head_self_attention(x, params['attn'], params['num_heads'], causal_mask)
    mlp_fn = lambda x: mlp_block(x, params['mlp'])

    attn_out = pre_norm_sublayer(x, params['ln1']['gamma'], params['ln1']['beta'], mhsa_fn)
    return pre_norm_sublayer(attn_out, params['ln2']['gamma'], params['ln2']['beta'], mlp_fn)

# Step 44 - language_model_decoder
import torch

def language_model_decoder(x, blocks_params, causal_mask):
    # TODO: apply every decoder block in blocks_params sequentially to x and return the result
    for block_params in blocks_params:
        x = decoder_block(x, block_params, causal_mask)

    return x

# Step 45 - final_layer_norm
import torch

def final_layer_norm(x, gamma, beta):
    # TODO: apply the existing layer_norm primitive to x using gamma and beta.
    return layer_norm(x, gamma, beta)

# Step 46 - language_model_head
def language_model_head(x, w_out, b_out):
    # TODO: project hidden states (L, D) to vocabulary logits (L, V) using w_out and b_out
    return x @ w_out + b_out

# Step 47 - encode_image_to_tokens
def encode_image_to_tokens(image, vision_params, projector_params):
    # TODO: add a batch dim if needed, then compose the full pipeline:
    # split -> flatten -> project -> prepend class token -> add positions
    # -> vision encoder -> drop class token -> projector, and squeeze the batch dim.
    flag = False
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        flag = True

    patches = split_image_into_patches(image, vision_params['patch_size'])
    flat_patches = flatten_patches(patches)

    patch_embeds = project_patches_to_embeddings(flat_patches, vision_params['patch_proj_weight'], vision_params['patch_proj_bias'])
    tokens = prepend_class_token(patch_embeds, vision_params['class_token'])

    patch_sequence = add_position_embeddings(tokens, vision_params['position_embeddings'])
    encoded = vision_encoder(patch_sequence, vision_params, vision_params['num_heads'])

    patch_features = extract_patch_features(encoded)
    out = vision_language_projector(patch_features, projector_params)
    if flag:
        out = out.squeeze(0)
    return out

# Step 48 - vision_language_forward
def vision_language_forward(image, token_ids, params):
    # TODO: route image + token_ids through the full vision-language model and return (L, V) logits.
    image_tokens = encode_image_to_tokens(image, params['vision'], params['projector'])
    mm_embed = build_multimodal_embeddings(token_ids, image_tokens, params['embedding'], params['pos_embedding'], params['image_token_id'])

    causal_mask = build_causal_mask(mm_embed.shape[0])
    decoded = language_model_decoder(mm_embed, params['decoder_blocks'], causal_mask)
    norm = final_layer_norm(decoded, params['final_ln']['gamma'], params['final_ln']['beta'])

    logits = language_model_head(norm, params['lm_head']['w_out'], params['lm_head']['b_out'])
    return logits

# Step 49 - shift_logits_and_labels
import torch

def shift_logits_and_labels(logits, labels):
    # TODO: align each logit with the next-position label and return (shifted_logits, shifted_labels).
    return logits[:-1], labels[1:]

# Step 50 - per_position_cross_entropy
import torch

def per_position_cross_entropy(shifted_logits, shifted_labels, ignore_index=-100):
    """Per-position next-token cross-entropy with 0 at ignored positions."""
    # TODO: log-softmax over vocab, gather target log-probs, zero out ignored positions
    # shifted = shifted_logits - shifted_logits.max(dim=-1, keepdim=True).values
    # logsumexp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    # logprobs = shifted - logsumexp

    logprobs = torch.log_softmax(shifted_logits, dim=-1)

    mask = shifted_labels == ignore_index
    labels = torch.where(mask, 0, shifted_labels)
    entropy = -logprobs[torch.arange(labels.shape[0]), labels]
    entropy[mask] = 0.0
    return entropy

# Step 51 - masked_mean_loss
import torch

def masked_mean_loss(per_position_losses, shifted_labels, ignore_index=-100):
    """Average per-position losses over positions whose label != ignore_index."""
    # TODO: average per_position_losses over positions where shifted_labels != ignore_index
    mask = shifted_labels != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return per_position_losses[mask].mean()

# Step 52 - greedy_next_token
def greedy_next_token(logits):
    # TODO: return the int token id with the highest logit at the final position
    return torch.argmax(logits[-1]).item()

# Step 53 - apply_temperature
import torch

def apply_temperature(logits, temperature):
    """Scale logits by dividing by temperature."""
    # TODO: return a tensor of logits rescaled by the temperature value
    if temperature <= 0.0:
        temperature = 1.0

    return logits/temperature

# Step 54 - top_k_filter
import torch

def top_k_filter(logits, k):
    """Keep only the top-k logits; set all others to -inf."""
    # TODO: keep top-k logits, replace the rest with -inf
    if k == 0:
        k = logits.shape[0]
    non_top_k = torch.argsort(-logits)[k:]
    top_k_logits = logits.clone()
    top_k_logits[non_top_k] = -torch.inf
    return top_k_logits

