import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from typing import Optional

from models.transformer import TransformerBlock
from models.util import patchify, unpatchify, random_masking, get_1d_sincos_pos_embed


# Encoder using backbone ViT
class Encoder(nn.Module):
    def __init__(self,
                 num_leads: int, seq_len: int, patch_size: int,
                 mask_ratio: float = 0., num_classes: Optional[int] = None,
                 width: int = 768, depth: int = 12,
                 heads: int = 12, dim_head: int = 64, qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 mlp_dim: int = 3072,
                 drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):

        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {
            'num_leads': num_leads, 'seq_len': seq_len, 'patch_size': patch_size,
            'num_classes': num_classes if num_classes is not None else 'None',
            'width': width, 'depth': depth,
            'heads': heads, 'dim_head': dim_head, 'qkv_bias': qkv_bias, 'attn_drop_out_rate': attn_drop_out_rate,
            'mlp_dim': mlp_dim,
            'drop_out_rate': drop_out_rate, 'drop_path_rate': drop_path_rate
        }

        self.num_leads = num_leads
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.width = width
        self.depth = depth

        # embedding layers
        num_patches = seq_len // patch_size
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, width),
            nn.LayerNorm(width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embedding = nn.Parameter(torch.randn(num_leads, width))
        self.dropout = nn.Dropout(drop_out_rate)

        # transformer layers
        for i, rate in enumerate(torch.linspace(0, drop_path_rate, depth)):
            block = TransformerBlock(
                input_dim=width, output_dim=width,
                heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
                hidden_dim=mlp_dim,
                drop_out_rate=drop_out_rate, drop_path_rate=rate.item()
            )
            self.add_module(f'transformer_block{i}', block)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.mask_ratio = 0
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def __forward_encoding(self, x, lead_indices):
        if len(lead_indices) != x.shape[1]:
            raise ValueError(f'lead_indices not equal num leads')

        # patchifying & patch embedding
        x = patchify(x, self.patch_size)
        x = self.patch_embedding(x)
        b, l, n, d = x.shape

        # pos embedding
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # masking: length -> length * mask_ratio
        if self.mask_ratio > 0:
            x, mask, ids_restore = random_masking(x, self.mask_ratio)
        else:
            mask = torch.zeros([b, l, n], device=x.device)
            ids_restore = torch.arange(n, device=x.device).unsqueeze(0).repeat(b, l, 1)

        # sep embedding
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, l, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, l, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)

        # lead embedding
        n_masked_with_sep = x.shape[2]
        lead_embedding = self.lead_embedding[lead_indices]
        lead_embedding = lead_embedding.unsqueeze(0).unsqueeze(2)
        lead_embedding = lead_embedding.expand(b, -1, n_masked_with_sep, -1)
        x = x + lead_embedding

        # flatten patches
        x = rearrange(x, 'b c n p -> b (c n) p')
        x = self.dropout(x)

        # transformer layers
        for i in range(self.depth):
            x = getattr(self, f'transformer_block{i}')(x)
        return x, mask, ids_restore

    # for pretraining task
    def forward_encoding(self, x, lead_indices):
        x, mask, ids_restore = self.__forward_encoding(x, lead_indices)
        x = self.norm(x)
        return x, mask, ids_restore

    # for downstream task
    def forward(self, x, lead_indices):
        x, _, _ = self.__forward_encoding(x, lead_indices)
        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=x.shape[1])
        x = x[:, :, 1:-1, :]
        x = torch.mean(x, dim=(1, 2))
        x = self.norm(x)
        x = self.head(x)
        return x

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str
    

# Decoder using multi transformer layers
class Decoder(nn.Module):
    def __init__(self,
                 inp_embed_dim: int = 768, output_dim: int = 500, num_patches: int = 250,
                 width: int = 256, depth: int = 12,
                 heads: int = 12, dim_head: int = 64,
                 qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 mlp_dim: int = 4,
                 drop_out_rate: float = 0., drop_path_rate: float = 0.,):
        super().__init__()
        self._repr_dict = {
            'inp_embed_dim': inp_embed_dim, 'output_dim': output_dim, 'num_patches': num_patches,
            'width': width, 'depth': depth,
            'heads': heads, 'dim_head': dim_head,
            'qkv_bias': qkv_bias, 'attn_drop_out_rate': attn_drop_out_rate,
            'mlp_dim': mlp_dim,
            'drop_out_rate': drop_out_rate, 'drop_path_rate': drop_path_rate
        }

        self.width = width
        self.depth = depth

        # embeddings
        self.dec_embedding = nn.Linear(inp_embed_dim, width)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 2, width), requires_grad=False)

        # transformer layers
        for i, rate in enumerate(torch.linspace(0, drop_path_rate, depth)):
            block = TransformerBlock(
                input_dim=width, output_dim=width,
                heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
                hidden_dim=mlp_dim,
                drop_out_rate=drop_out_rate, drop_path_rate=rate.item()
            )
            self.add_module(f'transformer_block{i}', block)

        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, output_dim)

    def forward(self, x, ids_restore):
        b, l, n = ids_restore.shape
        x = self.dec_embedding(x)

        # append mask embeddings to sequence
        x = rearrange(x, 'b (c n) p -> b c n p', c=l)
        b, _, n_masked_with_sep, d = x.shape
        mask_embeddings = self.mask_embedding.unsqueeze(1)
        mask_embeddings = mask_embeddings.repeat(b, l, n + 2 - n_masked_with_sep, 1)

        # Unshuffle without SEP embedding
        x_wo_sep = torch.cat([x[:, :, 1:-1, :], mask_embeddings], dim=2)
        x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d))

        # positional embedding and SEP embedding
        x_wo_sep = x_wo_sep + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)
        left_sep = x[:, :, :1, :] + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = x[:, :, -1:, :] + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2)

        # lead-wise decoding
        x_decoded = []
        for i in range(l):
            x_lead = x[:, i, :, :]
            for i in range(self.depth):
                x_lead = getattr(self, f'transformer_block{i}')(x_lead)
            x_lead = self.norm(x_lead)
            x_lead = self.head(x_lead)
            x_decoded.append(x_lead[:, 1:-1, :])
        x = torch.stack(x_decoded, dim=1)
        return x

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str
    

# Generator
class Generator(nn.Module):
    def __init__(self,
                 num_leads: int = 12, seq_len: int = 2250, patch_size: int = 75,
                 # for encoder
                 mask_ratio: float = 0.,
                 embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, dim_head: int = 64,
                 # for decoder
                 decoder_embed_dim: int = 256, decoder_depth: int = 4,
                 decoder_num_heads: int = 4, decoder_dim_head: int = 64,
                 # common
                 mlp_ratio: int = 4, qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 drop_out_rate: float = 0., drop_path_rate: float = 0.,):

        super().__init__()
        self._repr_dict = {
            'num_leads': num_leads, 'seq_len': seq_len, 'patch_size': patch_size,
            'mask_ratio': mask_ratio,
            'embed_dim': embed_dim, 'depth': depth,
            'num_heads': num_heads, 'dim_head': dim_head,
            'decoder_embed_dim': decoder_embed_dim, 'decoder_depth': decoder_depth,
            'decoder_num_heads': decoder_num_heads, 'decoder_dim_head': decoder_dim_head,
            'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
        }

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Encoder
        self.encoder = Encoder(
            num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
            mask_ratio=mask_ratio,
            width=embed_dim, depth=depth,
            heads=num_heads, dim_head=dim_head,
            qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
            mlp_dim=mlp_ratio * embed_dim,
            drop_out_rate=drop_out_rate, drop_path_rate=drop_path_rate,
        )

        # Decoder
        self.decoder = Decoder(
            inp_embed_dim=embed_dim, output_dim=patch_size, num_patches=seq_len//patch_size,
            width=decoder_embed_dim, depth=decoder_depth,
            heads=decoder_num_heads, dim_head=decoder_dim_head,
            qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
            mlp_dim=mlp_ratio * decoder_embed_dim,
            drop_out_rate=drop_out_rate, drop_path_rate=drop_path_rate,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        e_pos_embed = get_1d_sincos_pos_embed(self.embed_dim, self.num_patches, sep_embed=True)
        self.encoder.pos_embedding.data.copy_(e_pos_embed.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False

        d_pos_embed = get_1d_sincos_pos_embed(self.decoder_embed_dim, self.num_patches, sep_embed=True)
        self.decoder.pos_embedding.data.copy_(d_pos_embed.float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        torch.nn.init.normal_(self.encoder.lead_embedding, std=.02)
        torch.nn.init.normal_(self.decoder.mask_embedding, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, lead_indices):
        # forward
        latent, mask, ids_restore = self.encoder.forward_encoding(x, lead_indices)
        pred = self.decoder(latent, ids_restore)
        return pred, mask
    

# Discriminator (ViT backbone as Encoder)
class Discriminator(nn.Module):
    def __init__(self,
                 num_leads: int, seq_len: int, patch_size: int, num_classes: Optional[int] = None,
                 width: int = 768, depth: int = 12,
                 heads: int = 12, dim_head: int = 64, qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 mlp_dim: int = 3072,
                 drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):

        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {
            'num_leads': num_leads, 'seq_len': seq_len, 'patch_size': patch_size,
            'num_classes': num_classes if num_classes is not None else 'None',
            'width': width, 'depth': depth,
            'heads': heads, 'dim_head': dim_head, 'qkv_bias': qkv_bias, 'attn_drop_out_rate': attn_drop_out_rate,
            'mlp_dim': mlp_dim,
            'drop_out_rate': drop_out_rate, 'drop_path_rate': drop_path_rate
        }

        self.depth = depth
        self.width = width
        self.num_patches = seq_len // patch_size

        # embedding layers
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, width),
            nn.LayerNorm(width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embedding = nn.Parameter(torch.randn(num_leads, width))
        self.dropout = nn.Dropout(drop_out_rate)

        # transformer layers
        for i, rate in enumerate(torch.linspace(0, drop_path_rate, depth)):
            block = TransformerBlock(
                input_dim=width, output_dim=width,
                heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
                hidden_dim=mlp_dim,
                drop_out_rate=drop_out_rate, drop_path_rate=rate.item()
            )
            self.add_module(f'transformer_block{i}', block)
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 1)

        # head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        e_pos_embed = get_1d_sincos_pos_embed(self.width, self.num_patches, sep_embed=True)
        self.pos_embedding.data.copy_(e_pos_embed.float().unsqueeze(0))
        self.pos_embedding.requires_grad = False

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.sep_embedding, std=.02)
        torch.nn.init.normal_(self.lead_embedding, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, combined_patches, lead_indices):
        if len(lead_indices) != combined_patches.shape[1]:
            raise ValueError(f'lead_indices not equal num leads')

        x = combined_patches
        # patchifying & patch embedding
        x = self.patch_embedding(x)
        b, l, n, d = x.shape

        # pos embedding
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # sep embedding
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, l, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, l, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)

        # lead embedding
        n_masked_with_sep = x.shape[2]
        lead_embedding = self.lead_embedding[lead_indices]
        lead_embedding = lead_embedding.unsqueeze(0).unsqueeze(2)
        lead_embedding = lead_embedding.expand(b, -1, n_masked_with_sep, -1)
        x = x + lead_embedding

        # flatten patches
        x = rearrange(x, 'b c n p -> b (c n) p')
        x = self.dropout(x)

        # transformer layers
        for i in range(self.depth):
            x = getattr(self, f'transformer_block{i}')(x)
        return x

    def forward_discriminator(self, combined_patches, lead_indices):
        """""
        output.shape: (batch_size * n_leads * n_patches, 1)
        """""
        x = self.forward_encoding(combined_patches, lead_indices)
        x = self.norm(x)
        # remove sep embedding
        x = rearrange(x, 'b (c n) p -> b c n p', c=combined_patches.shape[1])
        x = x[:, :, 1:-1, :]
        x = x.reshape(-1, x.shape[-1])
        x = self.classifier(x)
        return x

    # for downstream task
    def forward(self, combined_patches, lead_indices):
        """""
        output.shape: (batch_size, dim)
        """""
        x = self.forward_encoding(combined_patches, lead_indices)

        # remove sep embedding
        x = rearrange(x, 'b (c n) p -> b c n p', c=combined_patches.shape[1])
        x = x = x[:, :, 1:-1, :]

        # each sample into a vector
        x = torch.mean(x, dim=(1, 2))
        x = self.norm(x)
        x = self.head(x)
        return x

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str
    

class Full_Network(nn.Module):
    def __init__(self,
                 num_leads: int = 12, seq_len: int = 2250, patch_size: int = 75,
                 mask_ratio: float = 0.,
                 embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, dim_head: int = 64,
                 decoder_embed_dim: int = 256, decoder_depth: int = 4,
                 decoder_num_heads: int = 4, decoder_dim_head: int = 64,
                 mlp_ratio: int = 4, qkv_bias: bool = True, attn_drop_out_rate: float = 0.,
                 drop_out_rate: float = 0., drop_path_rate: float = 0.,
                 norm_pix_loss: bool = False):

        super().__init__()
        self._repr_dict = {
            'num_leads': num_leads, 'seq_len': seq_len, 'patch_size': patch_size,
            'mask_ratio': mask_ratio,
            'embed_dim': embed_dim, 'depth': depth,
            'num_heads': num_heads, 'dim_head': dim_head,
            'decoder_embed_dim': decoder_embed_dim, 'decoder_depth': decoder_depth,
            'decoder_num_heads': decoder_num_heads, 'decoder_dim_head': decoder_dim_head,
            'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
            'norm_pix_loss': norm_pix_loss
        }

        self.patch_size = patch_size

        self.generator = Generator(
            num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
            mask_ratio=mask_ratio,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, dim_head=dim_head,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, decoder_dim_head=decoder_dim_head,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
            drop_out_rate=drop_out_rate, drop_path_rate=drop_path_rate,
        )

        self.discriminator = Discriminator(
            num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
            width=decoder_embed_dim, depth=depth, heads=num_heads, dim_head=dim_head,
            mlp_dim=decoder_embed_dim*mlp_ratio, qkv_bias=qkv_bias, attn_drop_out_rate=attn_drop_out_rate,
            drop_out_rate=drop_out_rate, drop_path_rate=drop_path_rate,
        )

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        self.generator.initialize_weights()
        self.discriminator.initialize_weights()

    def mel_loss(self, x, pred, mask):
        """
        x: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        mask: (batch_size, num_leads, n), 0 is keep, 1 is remove,
        """
        target = patchify(x, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (batch_size, num_leads, n), mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def dis_loss(self, mask, pred_dis):
        """
        mask: (batch_size, num_leads, num_patches)
        pred_dis: (batch_size * num_leads * num_patches, 1)
        """
        flattend_mask = mask.reshape(-1, 1)
        loss = F.binary_cross_entropy_with_logits(pred_dis, flattend_mask)
        return loss

    def loss(self, x, pred, pred_dis, mask):
        mel_loss = self.mel_loss(x, pred, mask)
        dis_loss = self.dis_loss(mask, pred_dis)
        return mel_loss + dis_loss

    def forward(self, x, lead_indices):
        # forward # remove batch_size
        lead_indices = lead_indices[0]
        pred, mask = self.generator(x, lead_indices)
        mask_appended = mask.unsqueeze(-1).repeat(1, 1, 1, self.patch_size)
        fake_patches = (pred * mask_appended)
        real_patches = (patchify(x, self.patch_size) * (1 - mask_appended))
        combined_patches = fake_patches + real_patches
        dis_outs = self.discriminator.forward_discriminator(combined_patches, lead_indices)

        # loss
        loss = self.loss(x, pred, dis_outs, mask)
        return {'pred': pred, 'dis_outs': dis_outs, 'loss': loss}
    

def model_base(**kwargs):
    model = Full_Network(
        # num_leads=12, seq_len=2250, patch_size=75, mask_ratio=0.75,
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4,
        **kwargs
    )
    return model

def model_small(**kwargs):
    model = Full_Network(
        # num_leads=12, seq_len=2250, patch_size=75, mask_ratio=0.75,
        embed_dim=364, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4,
        **kwargs
    )
    return model

def model_test(**kwargs):
    model = Full_Network(
        # num_leads=12, seq_len=2250, patch_size=75, mask_ratio=0.75,
        embed_dim=128, depth=2, num_heads=2,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=2,
        mlp_ratio=1,
        **kwargs
    )
    return model