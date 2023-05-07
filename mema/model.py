from __future__ import annotations

from typing import Annotated
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from einops import rearrange


# https://arxiv.org/abs/1910.07467
class RMSNorm(nn.Module):
    eps: float
    dim: int

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.var_eps = eps
        self.dim = dim

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Annotated[torch.Tensor, ...]) -> Annotated[torch.Tensor, ...]:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.var_eps)

        if self.weight.dtype != x.dtype:
            x = x.to(self.weight.dtype)

        return self.weight * x


class MultiLayerPerceptron(nn.Module):
    dim: int
    hidden_dim: int

    lin_in: nn.Linear
    lin_out: nn.Linear
    act: nn.GELU

    def __init__(
        self, dim: int, hidden_dim: int | None, expansion_ratio: float = 1.0
    ) -> None:
        """\
        if hidden_dim is not specified, it will be set to dim * expansion_ratio else expansion_ratio will be ignored
        """
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * expansion_ratio)

        self.lin_in = nn.Linear(self.dim, self.hidden_dim)
        self.lin_out = nn.Linear(self.hidden_dim, self.dim)
        self.act = nn.GELU()  # TODO support other activations

    def forward(
        self, x: Annotated[torch.Tensor, ..., 'D']
    ) -> Annotated[torch.Tensor, ..., 'D']:
        return self.lin_out(self.act(self.lin_in(x)))


class RotaryPositionalEmbedding(nn.Module):
    ferqs: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('freqs', torch.zeros(1))

    def foward(self) -> None:
        raise NotImplementedError

    def apply_rotary_embeds(self) -> None:
        raise NotImplementedError

    def _rotate_half(self) -> None:
        raise NotImplementedError


def apply_causal_mask(
    x: Annotated[torch.Tensor, ..., 'T', 'T']
) -> Annotated[torch.Tensor, ..., 'T', 'T']:
    assert x.shape[-2] == x.shape[-1]
    L = x.shape[-1]

    mask_value = -torch.finfo(x.dtype).max
    mask = torch.triu(torch.ones(L, L, device=x.device), 1)

    return torch.masked_fill(x, ~mask, mask_value)


# Attention Is All You Need https://arxiv.org/abs/1706.03762
class MultiHeadedAttention(nn.Module):

    """\
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
    """

    dim: int
    value_dim: int
    out_dim: int

    num_heads: int
    k_dim_head: int
    v_dim_head: int

    use_rotary_pos_emb: bool

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        k_dim_head: int = 64,
        v_dim_head: int = 64,
        value_dim: int | None = None,
        out_dim: int | None = None,
        use_rotary_pos_emb: bool = True,
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

        self.use_rotary_pos_emb = use_rotary_pos_emb

    def forward(
        self,
        q: Annotated[torch.Tensor, ..., 'T', 'K'],
        k: Annotated[torch.Tensor, ..., 'T', 'K'],
        v: Annotated[torch.Tensor, ..., 'T', 'V'],
        mask: Annotated[torch.Tensor, 'T', 'T'] | None = None,
        rotary_freqs: Annotated[torch.Tensor, 'T', 'L'] | None = None,
        rotary_pos_scale: float = 1.0,
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

        if self.use_rotary_pos_emb:
            assert rotary_freqs is not None
            q_i, k_i, v_i = self.rotary_pos_emb(
                q_i, k_i, v_i, rotary_freqs, rotary_pos_scale
            )

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

    def _rotate_half(
        self, x: Annotated[torch.Tensor, ..., 'T', 'K']
    ) -> Annotated[torch.Tensor, ..., 'T', 'K']:
        x = rearrange(x, '... (j d) -> ... j d', j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        freqs: Annotated[torch.Tensor, 'T', 'L'],
        x: Annotated[torch.Tensor, ..., 'T', 'K'],
        scale: float | int,
    ) -> Annotated[torch.Tensor, ..., 'T', 'K']:
        seq_len = x.shape[-2]
        freqs = freqs[-seq_len:, :]
        return (x * freqs.cos() * scale) + (self._rotate_half(x) * freqs.sin() * scale)

    def rotary_pos_emb(
        self,
        q: Annotated[torch.Tensor, ..., 'T', 'K'],
        k: Annotated[torch.Tensor, ..., 'T', 'K'],
        v: Annotated[torch.Tensor, ..., 'T', 'V'],
        freqs: Annotated[torch.Tensor, 'T', 'L'],
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = freqs.shape[-1]
        q_scale, k_scale = (scale, scale**-1.0)
        (ql, qr), (kl, kr), (vl, vr) = map(
            lambda t: (t[..., :L], t[..., L:]), (q, k, v)
        )

        args = ((ql, q_scale), (kl, k_scale), (vl, k_scale))
        ql, kl, vl = map(lambda args: self._rotary_pos_emb(freqs, *args), args)
        q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))
        return q, k, v
