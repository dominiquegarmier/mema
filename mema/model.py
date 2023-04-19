from __future__ import annotations

import torch.nn as nn
import torch
from einops import einsum, rearrange
from typing import Annotated
import torch.nn.functional as F


# Attention Is All You Need https://arxiv.org/abs/1706.03762
class Attention(nn.Module):
    '''
    some implementation details were taken from https://github.com/lucidrains/x-transformers
    which is licended under the...

    MIT License

    Copyright (c) 2020 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    '''

    dim: int
    value_dim: int
    out_dim: int

    num_heads: int
    k_dim_head: int
    v_dim_head: int

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        k_dim_head: int = 64,
        v_dim_head: int = 64,
        value_dim: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()

        if value_dim is None:
            value_dim = dim
        if out_dim is None:
            out_dim = value_dim

        self.dim = dim
        self.value_dim = value_dim
        self.out_dim = out_dim

        self.num_heads = num_heads
        self.k_dim_head = k_dim_head
        self.v_dim_head = v_dim_head

        v_dim = self.v_dim_head * self.num_heads
        q_dim = k_dim = self.k_dim_head * self.num_heads

        self.w_q = nn.Linear(self.dim, q_dim, bias=False)
        self.w_k = nn.Linear(self.dim, k_dim, bias=False)
        self.w_v = nn.Linear(self.value_dim, v_dim, bias=False)
        self.w_o = nn.Linear(v_dim, self.out_dim, bias=False)

    def forward(
        self,
        q: Annotated[torch.Tensor, ..., 'T', 'K'],
        k: Annotated[torch.Tensor, ..., 'T', 'K'],
        v: Annotated[torch.Tensor, ..., 'T', 'V'],
        mask: Annotated[torch.Tensor, 'T', 'T'] | None = None,
    ) -> Annotated[torch.Tensor, ..., 'T', 'O']:
        assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
        assert q.shape[-1] == k.shape[-1] == self.dim
        assert v.shape[-1] == self.value_dim
        if mask is not None:
            assert mask.shape == (q.shape[-2], k.shape[-2])
        B = q.shape[:-2]  # batch shape

        q_i = rearrange(self.w_q(q), '... T (H k) -> ... H T k', H=self.num_heads)
        k_i = rearrange(self.w_k(k), '... T (H k) -> ... H T k', H=self.num_heads)
        v_i = rearrange(self.w_v(v), '... T (H v) -> ... H T v', H=self.num_heads)

        # use scaled dot product similarity
        s_qk = einsum('... H i k, ... H j k -> ... H i j', q_i, k_i)
        s_qk = s_qk / (q_i.shape[-1] ** 0.5)

        # apply mask
        if mask is not None:
            mask = mask.view(*B, *mask.shape)
            mask_value = -torch.finfo(s_qk.dtype).max
            s_qk = s_qk.masked_fill(~mask, mask_value)

        # softmax
        attn: Annotated[torch.Tensor, ..., 'H', 'T', 'T'] = F.softmax(s_qk, dim=-1)

        vals = einsum('... H T i, ... H i v -> ... H T v', attn, v_i)
        out = self.w_o(rearrange(vals, '... H T v -> ... T (H v)'))
        return out
