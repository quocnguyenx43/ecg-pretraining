import torch
from einops.layers.torch import Rearrange
from einops import rearrange

def patchify(series, patch_size):
    """
    series: (batch_size, num_leads, seq_len)
    x: (batch_size, num_leads, n, patch_size)
    """
    assert series.shape[2] % patch_size == 0
    x = rearrange(series, 'b c (n p) -> b c n p', p=patch_size)
    return x

def unpatchify(x):
    """
    x: (batch_size, num_leads, n, patch_size)
    series: (batch_size, num_leads, seq_len)
    """
    series = rearrange(x, 'b c n p -> b c (n p)')
    return series

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: (batch_size, num_leads, n, embed_dim)
    """
    b, num_leads, n, d = x.shape
    len_keep = int(n * (1 - mask_ratio))

    noise = torch.rand(b, num_leads, n, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=2)

    # keep the first subset
    ids_keep = ids_shuffle[:, :, :len_keep]
    x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([b, num_leads, n], device=x.device)
    mask[:, :, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=2, index=ids_restore)

    return x_masked, mask, ids_restore

def get_1d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            temperature: float = 10000,
                            sep_embed: bool = False):
    """Positional embedding for 1D patches.
    """
    assert (embed_dim % 2) == 0, \
        'feature dimension must be multiple of 2 for sincos emb.'
    grid = torch.arange(grid_size, dtype=torch.float32)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (temperature ** omega)

    grid = grid.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat((grid.sin(), grid.cos()), dim=1)
    if sep_embed:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed, torch.zeros(1, embed_dim)],
                              dim=0)
    return pos_embed