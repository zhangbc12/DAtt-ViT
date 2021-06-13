import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms
from load import get_rotate


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage(image)
    return image


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size=224, patch_size=32, num_classes=6, dim=512, depth=4, heads=8, mlp_dim=1024, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, pre = True):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if pre:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        x = self.mlp_head(x)

        return x


class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features= 3 * 224 * 224,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(0.8),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, 3, 224, 224)))
        img_transform = F.grid_sample(img, grid)

        return img_transform


class DAtt_ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.st_1 = LocalNetwork()
        self.st_2 = LocalNetwork()
        self.st_3 = LocalNetwork()
        self.transformer_1 = ViT()
        self.transformer_2 = ViT()
        self.transformer_3 = ViT()

        self.transformer = Transformer(dim=512, depth=4, heads=8, dim_head = 64, mlp_dim=1024, dropout=0.1)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 6)
        )

    def forward(self, img_1, img_2, img_3):
        img_1 = self.st_1(img_1)
        img_2 = self.st_2(img_2)
        img_3 = self.st_3(img_3)

        attention_latent_1 = self.transformer_1(img_1)
        attention_latent_2 = self.transformer_2(img_2)
        attention_latent_3 = self.transformer_3(img_3)

        latent = attention_latent_1 + attention_latent_2 + attention_latent_3
        y = self.transformer(latent)

        y = y[:, 0]

        y = self.to_latent(y)

        y = self.mlp_head(y)

if __name__ == '__main__':
    from thop import profile
    # 增加可读性
    from thop import clever_format

    model = DAtt_ViT()
    input = torch.randn(1, 3, 224, 224)
    input_1, input_2, input_3 = input, input, input
    y = model(input_1, input_2, input_3)
    print(2)