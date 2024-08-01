import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

class DropPath(nn.Module):
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop_out_rate)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 input_dim: int, output_dim: int,
                 heads: int = 8, dim_head: int = 64, qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                        nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 input_dim: int, output_dim: int,
                 heads: int = 8, dim_head: int = 32, qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 hidden_dim: int = 3072,
                 drop_out_rate: float = 0., drop_path_rate: float = 0.):
        super().__init__()

        self.attn = PreNorm(
            dim=input_dim, fn=Attention(
                input_dim=input_dim, output_dim=output_dim,
                heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
                drop_out_rate=drop_out_rate,
        ))
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        self.ff = PreNorm(
            dim=output_dim, fn=FeedForward(
                input_dim=output_dim, output_dim=output_dim,
                hidden_dim=hidden_dim, drop_out_rate=drop_out_rate
        ))
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x