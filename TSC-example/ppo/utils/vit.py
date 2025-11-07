from os import XATTR_SIZE_MAX
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# https://github.com/lucidrains/vit-pytorch

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormDouble(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm1(x), self.norm2(y), **kwargs)

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

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         attn = self.attend(dots)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, embedding_dimension, heads=1, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.heads = heads
        self.dim_head = dim_head

        self.query = nn.Linear(embedding_dimension, heads * dim_head, bias=False)
        self.key = nn.Linear(embedding_dimension, heads * dim_head, bias=False)
        self.value = nn.Linear(embedding_dimension, heads * dim_head, bias=False)

        self.linear = nn.Linear(heads*dim_head, embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, embedding_dim = x.size()
        query = self.query(x).view(batch_size, self.heads, self.dim_head)
        key = self.key(x).view(batch_size, self.heads, self.dim_head).transpose(1, 2)
        value = self.value(x).view(batch_size, self.heads, self.dim_head)
        scores = torch.matmul(query, key) / torch.sqrt(torch.tensor(self.dim_head, dtype=float))
        attention_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, value).reshape(batch_size, self.heads * self.dim_head)
        out = self.dropout(self.linear(out))
        return out

class CrossAttention_Querydimchanged(nn.Module):
    def __init__(self, key_dimension, query_embedding, heads=1, dim_head=64, dropout=0.0):
        super(CrossAttention_Querydimchanged, self).__init__()
        self.heads = heads
        self.dim_head = dim_head

        self.query = nn.Linear(query_embedding, heads * dim_head, bias=False)
        self.key = nn.Linear(key_dimension, heads * dim_head, bias=False)
        self.value = nn.Linear(key_dimension, heads * dim_head, bias=False)

        self.linear = nn.Linear(heads*dim_head, key_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):        ### key, query
        batch_size, embedding_dim = x.size()
        query = self.query(y).view(batch_size, self.heads, self.dim_head)
        key = self.key(x).view(batch_size, self.heads, self.dim_head).transpose(1, 2)
        value = self.value(x).view(batch_size, self.heads, self.dim_head)
        scores = torch.matmul(query, key) / torch.sqrt(torch.tensor(self.dim_head, dtype=float))
        attention_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, value).reshape(batch_size, self.heads * self.dim_head)
        out = self.dropout(self.linear(out))
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, embedding_dimension, heads=1, dim_head=64, dropout=0.0, embedding_dimension_query=None):
        super(CrossAttention, self).__init__()
        self.embedding_dimension = embedding_dimension
        if embedding_dimension_query is None:
            self.embedding_dimension_query = embedding_dimension
        else:
            self.embedding_dimension_query = embedding_dimension_query
        self.heads = heads
        self.dim_head = dim_head

        self.query = nn.Linear(self.embedding_dimension_query, heads * dim_head, bias=False)
        self.key = nn.Linear(self.embedding_dimension, heads * dim_head, bias=False)
        self.value = nn.Linear(self.embedding_dimension, heads * dim_head, bias=False)

        self.linear = nn.Linear(heads*dim_head, embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        batch_size, embedding_dim = x.size()
        query = self.query(y).view(batch_size, self.heads, self.dim_head)
        key = self.key(x).view(batch_size, self.heads, self.dim_head).transpose(1, 2)
        value = self.value(x).view(batch_size, self.heads, self.dim_head)
        scores = torch.matmul(query, key) / torch.sqrt(torch.tensor(self.dim_head, dtype=float))
        attention_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, value).reshape(batch_size, self.heads * self.dim_head)
        out = self.dropout(self.linear(out))
        return out


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()

#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_q = nn.Linear(dim, inner_dim, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )   if project_out else nn.Identity()
    
#     def forward(self, x, k, v, ret=[]):
#         b, n, _, h = *x.shape, self.heads
#         q = self.to_q(x)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q, k, v])
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         attn = self.attend(dots)
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         if 'attn' in ret:
#             out = [out, attn]
#         return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
