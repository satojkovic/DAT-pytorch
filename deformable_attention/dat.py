import torch
import torch.nn as nn
import einops


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.layer_norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


class DAT(nn.Module):
    def __init__(
        self, dim_heads, num_heads, num_offset_groups, offset_kernel_size, offset_stride
    ):
        super().__init__

        self.dim_heads = dim_heads
        self.num_heads = num_heads
        self.num_offset_groups = num_offset_groups
        self.offset_kernel_size = offset_kernel_size
        self.offset_stride = offset_stride
        assert self.offset_kernel_size == self.offset_stride
        self.pad_size = 0

        embed_dim = self.num_heads * self.dim_heads
        self.offset_dims = embed_dim // self.num_offset_groups

        self.offset_net = nn.Sequential(
            nn.Conv2d(
                self.offset_dims,
                self.offset_dims,
                self.offset_kernel_size,
                self.offset_stride,
                self.pad_size,
                groups=self.num_offset_groups,
            ),
            LayerNormProxy(self.offset_dims),
            nn.GELU(),
            nn.Conv2d(self.offset_dims, 2, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        pass
