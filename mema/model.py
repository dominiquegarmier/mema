from __future__ import annotations

from typing import Annotated
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from einops import rearrange


# Attention Is All You Need https://arxiv.org/abs/1706.03762
class Attention(nn.Module):
    '''\
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

        # apply rotary positional embeddings
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


# Attention Is All You Need https://arxiv.org/abs/1706.03762
class AttentionLayer(nn.Module):
    dim: int
    out_dim: int

    attention: Attention
    norm: nn.LayerNorm

    use_ff: bool
    ff_linear: nn.Linear | None = None
    ff_norm: nn.LayerNorm | None = None

    def __init__(
        self,
        dim: int = 2048,
        out_dim: int | None = None,
        num_heads: int = 16,
        use_ff: bool = True,
    ) -> None:
        self.dim = dim
        self.out_dim = out_dim or dim

        self.attention = Attention(dim=self.dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(self.dim)

        self.use_ff = use_ff
        if self.use_ff:
            self.ff_linear = nn.Linear(self.dim, self.out_dim, bias=False)
            self.ff_norm = nn.LayerNorm(self.out_dim)

    def forward(
        self,
        x: Annotated[torch.Tensor, ..., 'T', 'D'],
        mask: Annotated[torch.Tensor, 'T', 'T'],
    ) -> Annotated[torch.Tensor, ..., 'T', 'O']:
        # TODO add support for rotary positional embeddings
        attn = self.attention(x, x, x, mask=mask)
        out = self.norm(x + attn)
        if self.use_ff:
            if TYPE_CHECKING:
                assert self.ff_linear is not None
                assert self.ff_norm is not None
            out = self.ff_norm(out + self.ff_linear(out))
        return out


# Neural Nearest Neighbor Networks https://arxiv.org/abs/1810.12575
class NeuralKNearestNeighbor(nn.Module):
    '''\
    implementation details were taken from https://github.com/DominiqueGarmier/neural-nearest-neighbor
    which is licended under the...

    MIT License

    Copyright (c) 2023 Dominique Garmier

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    '''

    _k: int
    _temp: float
    _dim: int
    _feature: int

    _no_values: bool = False

    def __init__(
        self, k: int, dim: int, temp: float, feature: int | None = None
    ) -> None:
        super().__init__()
        self._k = k
        self._temp = temp

        self._dim = dim
        self._feature = feature or dim
        if feature is None:
            self._no_values = True

    def forward(
        self,
        query: Annotated[torch.Tensor, ..., 'D'],
        keys: Annotated[torch.Tensor, ..., 'D', 'N'],
        values: Annotated[torch.Tensor, ..., 'F', 'N'] | None = None,
    ) -> Annotated[torch.Tensor, ..., 'K', 'F']:
        if values is None:
            assert self._no_values
            values = keys
        assert query.shape[-1] == keys.shape[-2] == self._dim
        assert values.shape[-2] == self._feature
        assert keys.shape[-1] == values.shape[-1]

        sims = self._similarity(query, keys)
        omega = self._compute_omega(s=sims, k=self._k, t=self._temp)
        k_nearest = einsum(omega, values, '... N K, ... F N -> ... K F')
        return k_nearest

    def _similarity(
        self,
        query: Annotated[torch.Tensor, ..., 'D'],
        key: Annotated[torch.Tensor, ..., 'D', 'N'],
    ) -> Annotated[torch.Tensor, ..., 'N']:
        return -einsum(query, key, '... D, ... D N -> ... N') / (self._dim**0.5)

    def _compute_omega(
        self, s: Annotated[torch.Tensor, ..., 'N'], k: int, t: float
    ) -> Annotated[torch.Tensor, ..., 'N', 'K']:
        alpha = F.softmax(s, dim=-1)
        omega = torch.empty(*s.shape, k)

        omega[..., 0] = F.softmax(alpha / t, dim=-1)
        for i in range(1, k):
            alpha = alpha + torch.log(1 - omega[..., i - 1])
            omega[..., i] = F.softmax(alpha / t, dim=-1)

        return omega


class TopicalTransformer(nn.Module):
    dim: int
    query_dim: int

    num_layers: int
    layers: nn.ModuleList

    def __init__(
        self, dim: int = 1024, query_dim: int = 256, num_layers: int = 2
    ) -> None:
        super().__init__()

        self.dim = dim
        self.query_dim = query_dim

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            ...  # TODO

    def forward(
        self, x: Annotated[torch.Tensor, ..., 'T', 'D']
    ) -> Annotated[torch.Tensor, ..., 'Q']:
        raise NotImplementedError


class MemoryLayer(nn.Module):
    '''\
    layer that injects memory using nknn
    '''

    dim: int
    key_dim: int
    value_dim: int
    memory_dim: int

    top_k: int
    context: int

    query_model: TopicalTransformer
    neural_knn: NeuralKNearestNeighbor

    ff_dim: nn.Linear
    ff_context: nn.Linear
    ff_norm: nn.LayerNorm

    def __init__(
        self,
        dim: int = 1024,
        top_k: int = 32,
        context: int = 2048,
        key_dim: int = 256,
        value_dim: int = 256,
        memory_dim: int = 16384,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_dim = memory_dim

        self.top_k = top_k
        self.context = context

        self.query_model = TopicalTransformer(dim=self.dim, query_dim=self.key_dim)
        self.neural_knn = NeuralKNearestNeighbor(
            k=top_k, dim=self.key_dim, temp=0.1, feature=self.value_dim
        )

        self.ff_dim = nn.Linear(self.value_dim, self.value_dim, bias=False)
        self.ff_context = nn.Linear(self.value_dim, self.dim, bias=False)
        self.ff_norm = nn.LayerNorm(self.value_dim)

    def forward(
        self,
        x: Annotated[torch.Tensor, ..., 'T', 'D'],
        keys: Annotated[torch.Tensor, ..., 'K', 'N'],
        values: Annotated[torch.Tensor, ..., 'V', 'N'],
    ) -> Annotated[torch.Tensor, ..., 'T', 'D']:
        assert x.shape[-1] == self.dim
        query: Annotated[torch.Tensor, ..., 'Q'] = self.query_model(x)
        value = self.neural_knn(query, keys, values)

        cast = self.ff_dim(value)
        cast = rearrange(cast, '... K V -> ... V K')
        cast = self.ff_context(cast)
        cast = rearrange(cast, '... V T -> ... T V')

        out = self.ff_norm(x + cast)
        return out
