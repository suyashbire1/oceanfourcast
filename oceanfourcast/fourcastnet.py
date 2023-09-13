import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
import math
from functools import partial
from collections import OrderedDict


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        # (in_features = d)
        # (hidden_features = d*mlp_ratio)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # (b, h*w, d)
        x = self.fc1(x)  # (b, h*w, d*mlp_ratio)
        x = self.act(x)  # (b, h*w, d*mlp_ratio)
        x = self.drop(x)  # (b, h*w, d*mlp_ratio)
        x = self.fc2(x)  # (b, h*w, d)
        x = self.drop(x)  # (b, h*w, d)
        return x


class PosEmbed(nn.Module):

    def __init__(self, img_size, device="cpu"):
        super(PosEmbed, self).__init__()
        self.img_size = img_size

        self.h = img_size[0]
        self.w = img_size[1]

        self.grid_x = torch.nn.Parameter(torch.zeros(1, 1, self.h,
                                                     self.w)).to(device)
        self.grid_y = torch.nn.Parameter(torch.zeros(1, 1, self.h,
                                                     self.w)).to(device)

    def forward(self, x):
        x = torch.cat((x, self.grid_x, self.grid_y), dim=1)
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels=20,
                 embed_dim=768,
                 device="cpu"):
        super(PatchEmbed, self).__init__()
        assert (img_size[0] % patch_size == 0 and img_size[1] % patch_size
                == 0), f"Input image size doesn't match model."
        self.img_size = img_size
        #self.patch_size = patch_size
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        #self.n_patches = self.h * self.w

        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              device=device)

    def forward(self, x):
        x = self.proj(x)  # (b, d, h, w)
        x = x.flatten(2)  # (b, d, h*w)
        x = x.transpose(1, 2)  # (b, h*w, d)
        return x


class AFNONet(nn.Module):

    def __init__(self,
                 embed_dim=256,
                 n_blocks=8,
                 sparsity=1e-2,
                 img_size=None,
                 in_channels=20,
                 out_channels=20,
                 mlp_ratio=4.,
                 drop_rate=0.5,
                 norm_layer=None,
                 depth=12,
                 patch_size=8,
                 use_blocks=True,
                 device='cpu'):

        super(AFNONet, self).__init__()
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.sparsity = sparsity

        if img_size is None:
            img_size = [720, 1440]

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        num_patches = self.h * self.w
        self.p = patch_size
        self.Co = out_channels
        self.Ci = in_channels

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_channels=in_channels,
                                      embed_dim=embed_dim,
                                      device=device)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim).to(device))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.dropout = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,
                  mlp_ratio=mlp_ratio,
                  drop=drop_rate,
                  norm_layer=norm_layer,
                  h=self.h,
                  w=self.w,
                  use_blocks=use_blocks,
                  device=device,
                  num_blocks=n_blocks,
                  sparsity=sparsity) for i in range(depth)
        ])

        self.head = nn.Linear(embed_dim,
                              out_channels * patch_size * patch_size,
                              bias=False)


#        if patch_size == 8:
#            self.pre_logits = nn.Sequential(OrderedDict([
#                ('conv1', nn.ConvTranspose2d(embed_dim, out_channels*16, kernel_size=(2, 2), stride=(2, 2))),
#                ('act1', nn.Tanh()),
#                ('conv2', nn.ConvTranspose2d(out_channels*16, out_channels*4, kernel_size=(2, 2), stride=(2, 2))),
#                ('act2', nn.Tanh())
#            ]))
#            assert (patch_size % 4 == 0), f"Patch size is not divisible by 4"
#            ks, st = patch_size//4, patch_size//4
#            self.head = nn.ConvTranspose2d(out_channels*4, out_channels, kernel_size=(ks, ks), stride=(st, st))
#        elif patch_size == 4:
#            self.pre_logits = nn.Sequential(OrderedDict([
#                ('conv1', nn.ConvTranspose2d(embed_dim, out_channels*4, kernel_size=(2, 2), stride=(2, 2))),
#                ('act1', nn.Tanh())
#            ]))
#
#            # Generator head
#            self.head = nn.ConvTranspose2d(out_channels*4, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        b = x.shape[0]  # (b, Ci, img_size[0], img_size[1])
        x = self.patch_embed(x)  # (b, h*w, d)
        x = x + self.pos_embed  # (b, h*w, d)
        x = self.pos_drop(x)  # (b, h*w, d)
        for blk in self.blocks:
            x = blk(x)  # (b, h*w, d)
        x = self.norm(x).transpose(1, 2)  # (b, d, h*w)
        x = torch.reshape(x,
                          [-1, self.embed_dim, self.h, self.w])  # (b, d, h, w)
        x = self.dropout(x)  # (b, d, h, w)
        x = torch.permute(x, (0, 2, 3, 1))  # (b, h, w, d)
        x = self.head(x)  # (b, h, w, Co*p*p)
        x = rearrange(x,
                      "b h w (Co p1 p2) -> b Co (h p1) (w p2)",
                      Co=self.Co,
                      p1=self.p,
                      p2=self.p,
                      h=self.h,
                      w=self.w)  # (b, Co, img_size[0], img_size[1])

        #        x = torch.reshape(x, [b, self.h,
        #            self.w*out_channels*patch_size*patch_size])                  # (b, h, w*out_channels*patch_size**2)
        #        x = torch.reshape(x, [b, self.h, out_channels*patch_size,
        #            self.w*patch_size])                                          # (b, h, out_channels*patch_size, w*patch_size)
        #        x = torch.reshape(x, [b, self.h*out_channels*patch_size,
        #            self.w*patch_size])                                          # (b, h*out_channels*patch_size, w*patch_size)
        #        x = torch.reshape(x, [b, out_channels, self.h*patch_size,
        #            self.w*patch_size])                                          # (b, out_channels, h*patch_size, w*patch_size)

        #        x = self.pre_logits(x)                                           # (b, out_channels*4, h*4, w*4)                        # hard-coded!
        #        x = self.head(x)                                                 # (b, out_channels, h*patch_size, w*patch_size)        # hard-coded!
        return x


class Block(nn.Module):

    def __init__(self,
                 embed_dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 h=14,
                 w=8,
                 use_blocks=False,
                 device='cpu',
                 num_blocks=8,
                 sparsity=1e-2):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = AdaptiveFourierNeuralOperator(embed_dim,
                                                    h=h,
                                                    w=w,
                                                    device=device,
                                                    num_blocks=num_blocks,
                                                    sparsity=sparsity)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        residual = x  # (b, h*w, d)
        x = self.norm1(x)  # (b, h*w, d)
        x = self.filter(x)  # (b, h*w, d)
        x = self.norm2(x)  # (b, h*w, d)
        x = self.mlp(x)  # (b, h*w, d)
        x = x + residual
        return x


class AdaptiveFourierNeuralOperator(nn.Module):

    def __init__(self,
                 embed_dim,
                 h,
                 w,
                 num_blocks=8,
                 softshrink=True,
                 device='cpu',
                 sparsity=1e-2):
        super(AdaptiveFourierNeuralOperator, self).__init__()
        self.embed_dim = embed_dim
        self.h = h
        self.w = w

        self.num_blocks = num_blocks
        assert self.embed_dim % self.num_blocks == 0
        self.block_size = self.embed_dim // self.num_blocks

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2,
                                                              self.num_blocks,
                                                              self.block_size,
                                                              self.block_size,
                                                              device=device))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2,
                                                              self.num_blocks,
                                                              self.block_size,
                                                              self.block_size,
                                                              device=device))
        self.b1 = torch.nn.Parameter(
            self.scale *
            torch.randn(2, self.num_blocks, self.block_size, device=device))
        self.b2 = torch.nn.Parameter(
            self.scale *
            torch.randn(2, self.num_blocks, self.block_size, device=device))
        self.relu = nn.ReLU()

        self.softshrink = softshrink
        self.sparsity = sparsity

    def multiply(self, input, weights):
        return torch.einsum('...lm,lmn->...ln', input, weights)

    def forward(self, x):
        b, hw, d = x.shape

        bias = x

        x = x.reshape(b, self.h, self.w, d).float()  # (b, h, w, d)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # (b, h, w//2+1, d)
        x = x.reshape(b, x.shape[1], x.shape[2], self.num_blocks,
                      self.block_size)  # (b, h, w//2+1, k, d/k)

        #w1 (2, k, d/k, d/k)
        x_real_1 = F.relu(
            self.multiply(x.real, self.w1[0]) -
            self.multiply(x.imag, self.w1[1]) +
            self.b1[0])  # (b, h, w//2+1, k, d/k)
        x_imag_1 = F.relu(
            self.multiply(x.real, self.w1[1]) +
            self.multiply(x.imag, self.w1[0]) +
            self.b1[1])  # (b, h, w//2+1, k, d/k)

        #w2 (2, k, d/k, d/k)
        x_real_2 = F.relu(
            self.multiply(x_real_1, self.w2[0]) -
            self.multiply(x_imag_1, self.w2[1]) +
            self.b2[0])  # (b, h, w//2+1, k, d/k)
        x_imag_2 = F.relu(
            self.multiply(x_real_1, self.w2[1]) +
            self.multiply(x_imag_1, self.w2[0]) +
            self.b2[1])  # (b, h, w//2+1, k, d/k)

        x = torch.stack([x_real_2, x_imag_2],
                        dim=-1).float()  # (b, h, w//2+1, k, 2*d/k)
        x = F.softshrink(x, lambd=self.sparsity)  # (b, h, w//2+1, k, 2*d/k)

        x = torch.view_as_complex(x)  # (b, h, w//2+1, k, d/k)
        x = x.reshape(b, x.shape[1], x.shape[2],
                      self.embed_dim)  # (b, h, w//2+1, d)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2),
                             norm='ortho')  # (b, h, w, d)
        x = x.reshape(b, hw, d)  # (b, h*w, d)
        return x + bias
