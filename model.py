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

