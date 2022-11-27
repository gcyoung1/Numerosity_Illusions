# From https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch/blob/main/models.py
# FractalDb weights are in Google Drive linked from above repo
import torch
import torch.nn as nn
from functools import partial
import os
from zipfile import ZipFile

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


@register_model
def deit_tiny_patch16_224(pretrained=False, dataset='imagenet', **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dataset == 'imagenet':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        elif dataset == 'fractaldb':
            checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fractal_weights', 'deitt16_224_fractal1k_lr3e-4_300ep.pth')
            if not os.path.exists(checkpoint_path):
                zip_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fractal_weights.zip')
                extract_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fractal_weights')
                with ZipFile(zip_path, 'r') as fractalzip:
                    fractalzip.extractall(extract_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            model.load_state_dict(checkpoint["model"])
        else:
            raise NotImplementedError(f"Dataset {dataset} not recognized")
    return model

