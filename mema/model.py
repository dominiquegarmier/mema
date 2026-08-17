from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from einops import rearrange

from mema.config import MemaConfig
from mema.nknn import NeuralKNearestNeighbor


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


class LayerNorm(nn.Module):
    norm: nn.LayerNorm

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def foward(self, x: Annotated[torch.Tensor, ...]) -> Annotated[torch.Tensor, ...]:
        return self.norm(x)


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


# https://arxiv.org/abs/2104.09864
class RotaryPositionEmbedding(nn.Module):
    """\
    Roatary Embeddings as presented in the paper https://arxiv.org/abs/2104.09864
    does not contain any trainable parameters can be placed in the root nn.Module to allow for efficient caching.
    """

    theta: float
    dim: int

    _seq_len_cached: int
    _freqs_cis: Annotated[torch.Tensor, torch.complex64, 'T', 'D']
    _scale: Annotated[torch.Tensor, 'D']

    def __init__(self, dim: int, seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

        assert seq_len >= 1
        freqs_cis = self._get_freqs_cis(seq_len)
        self.register_buffer('_freqs_cis', freqs_cis)

    def _get_freqs_cis(
        self, seq_len: int, device: torch.device | None = None
    ) -> Annotated[torch.Tensor, torch.complex64, 'T', 'D']:
        self._seq_len_cached = seq_len
        half = self.dim // 2  # only apply to half of the dimensions, see the paper
        freqs = self.theta ** -(
            torch.arange(0, half, device=device or 'cpu').float() / half
        )
        seq = torch.arange(seq_len, device=freqs.device)
        freqs = einsum(seq, freqs, 'T, D -> T D')
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def get_freqs_cis(
        self, seq_len: int, device: torch.device
    ) -> Annotated[torch.Tensor, torch.complex64, 'T', 'D/2']:
        if seq_len > self._seq_len_cached:
            next_power_of_two = 2 ** math.ceil(math.log2(seq_len))
            freqs_cis = self._get_freqs_cis(next_power_of_two, device=device)
            self.register_buffer('_freqs_cis', freqs_cis)
        return self._freqs_cis[-seq_len:, :]

    @staticmethod
    def rotate_half(
        x: Annotated[torch.Tensor, ..., 'D'],
    ) -> Annotated[torch.Tensor, ..., 'D']:
        x = rearrange(x, '... (j d) -> ... j d', j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        x: Annotated[torch.Tensor, ..., 'T', 'D'],
    ) -> Annotated[torch.Tensor, ..., 'T', 'D']:
        """applies rotary embeddings to x"""
        freqs_cis = self.get_freqs_cis(x.shape[-2], device=x.device)
        assert x.shape[-1] == freqs_cis.shape[-1]

        freqs_cos = torch.view_as_real(freqs_cis)
        freqs_sin = torch.view_as_complex(freqs_cis)
        return (x * freqs_cos) + (self.rotate_half(x) * freqs_sin)


# https://arxiv.org/abs/2108.12409
class AttentionLinearBias(nn.Module):
    _cached_seq_len: int
    _bias_buffer: Annotated[torch.Tensor, 'T', 'T']
    _factor: float

    def __init__(self, seq_len: int, factor: float = 1 / (2**8)) -> None:
        super().__init__()
        self._factor = factor
        self._cached_seq_len = seq_len
        self.register_buffer('_bias_buffer', self._get_bias(seq_len))

    @staticmethod
    def _get_bias(
        l: int, device: torch.device | None = None
    ) -> Annotated[torch.Tensor, 'T', 'T']:
        a = torch.arange(0, l, device=device or 'cpu').reshape(-1, 1)
        return -torch.relu(a - a.T)

    def forward(
        self, seq_len: int, n_heads: int, device: torch.device
    ) -> Annotated[torch.Tensor, 'T', 'T', 'H']:
        if seq_len > self._cached_seq_len:
            next_power_of_two = 2 ** math.ceil(math.log2(seq_len))
            self._cached_seq_len = next_power_of_two
            self.register_buffer(
                '_bias_buffer', self._get_bias(next_power_of_two, device)
            )
        bias = self._bias_buffer[:seq_len, :seq_len].reshape(-1, -1, 1)
        head_factor = self._factor ** (1 / n_heads)
        k = torch.pow(head_factor, torch.arange(0, n_heads, device=device)).reshape(
            1, 1, -1
        )
        return bias * k


# Attention Is All You Need https://arxiv.org/abs/1706.03762
class Attention(nn.Module):
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

    causal: bool
    use_flash: bool

    rotary_pos_emb: RotaryPositionEmbedding | None
    attention_linear_bias: AttentionLinearBias | None

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        use_flash: bool = False,
        num_heads: int = 8,
        k_dim_head: int = 64,
        v_dim_head: int = 64,
        value_dim: int | None = None,
        out_dim: int | None = None,
        rotary_pos_emb: RotaryPositionEmbedding | None = None,
        attention_linear_bias: AttentionLinearBias | None = None,
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

        self.causal = causal
        self.use_flash = use_flash  # TODO implement flash

        # positional embedding
        assert (
            rotary_pos_emb is None or attention_linear_bias is None
        ), "can't use RoPE and ALiBi at the same time"
        self.rotary_pos_emb = rotary_pos_emb
        self.attention_linear_bias = attention_linear_bias

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

        if self.rotary_pos_emb is not None:
            rope_dim = self.rotary_pos_emb.dim

            def _apply(x: torch.Tensor) -> torch.Tensor:
                if TYPE_CHECKING:
                    assert self.rotary_pos_emb is not None
                return torch.cat(
                    (self.rotary_pos_emb(x[..., :rope_dim]), x[..., rope_dim:]), dim=-1
                )

            q_i = _apply(q_i)
            k_i = _apply(k_i)
            v_i = _apply(v_i)

        # use scaled dot product similarity
        s_qk = einsum(q_i, k_i, '... H i k, ... H j k -> ... H i j')

        s_qk = s_qk / (q_i.shape[-1] ** 0.5)

        # apply mask
        if mask is not None:
            mask = mask.view(*B, *mask.shape)
            mask_value = -torch.finfo(s_qk.dtype).max
            s_qk = s_qk.masked_fill(~mask, mask_value)

        # softmax
        attn: Annotated[torch.Tensor, ..., 'H', 'T', 'T'] = F.softmax(s_qk, dim=-1)

        vals = einsum(attn, v_i, '... H T i, ... H i v -> ... H T v')
        out = self.w_o(rearrange(vals, '... H T v -> ... T (H v)'))
        return out


class Memory(nn.Module):
    k: int

    dim: int
    dim_mem: int
    dim_key: int

    nknn: NeuralKNearestNeighbor
    norm: LayerNorm | RMSNorm
    proj_in: nn.Linear
    proj_out: nn.Linear

    def __init__(
        self, dim: int, k: int = 8, dim_mem: int = 256, dim_key: int | None = None
    ) -> None:
        super().__init__()

        self.dim = dim

        if dim_key is None:
            dim_key = dim_mem

        self.dim_mem = dim_mem
        self.dim_key = dim_key

        self.k = k

        self.proj_in = nn.Linear(self.dim, self.dim_key)
        self.nknn = NeuralKNearestNeighbor(self.k, self.dim, temp=0.1, feature=dim_mem)
        self.proj_out = nn.Linear(self.dim_mem * self.k, self.dim)
        self.norm = RMSNorm(self.dim)

    def foward(
        self,
        x: Annotated[torch.Tensor, ..., 'B', 'T', 'D'],
        k: Annotated[torch.Tensor, ..., 'N', 'K'],
        v: Annotated[torch.Tensor, ..., 'N', 'V'],
    ) -> Annotated[torch.Tensor, ..., 'B', 'T', 'D']:
        q = self.proj_in(x)
        nearest = self.nknn(q, k, v)
        nearest = rearrange(nearest, '... K F -> ... (K F)')

        out = self.proj_out(nearest)
        out = self.norm(out)
        return out


class MemaBlock(nn.Module):
    dim: int

    attn: Attention
    mlp: MultiLayerPerceptron
    memory: Memory | None = None

    pre_norm: RMSNorm | LayerNorm
    post_norm: RMSNorm | LayerNorm

    attn_dropout: nn.Dropout | None = None
    mem_dropout: nn.Dropout | None = None
    dropout: nn.Dropout | None = None

    def __init__(
        self,
        dim: int,
        memory: bool = False,
        dropout: float | None = None,
        attn_dropout: float | None = None,
        mem_dropout: float | None = None,  # TODO memory ingest?
    ) -> None:
        super().__init__()

        self.dim = dim
        self.attn = Attention(self.dim)
        self.mlp = MultiLayerPerceptron(self.dim, hidden_dim=4 * self.dim)

        self.pre_norm = RMSNorm(self.dim)
        self.post_norm = RMSNorm(self.dim)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        if attn_dropout is not None:
            self.attn_dropout = nn.Dropout(attn_dropout)

        assert memory is not None or mem_dropout is None, 'mem_dropout requires memory'
        if memory:
            self.memory = Memory(self.dim)
        if mem_dropout is not None:
            self.mem_dropout = nn.Dropout(mem_dropout)

    def forward(
        self,
        x: Annotated[torch.Tensor, ..., 'B', 'T', 'D'],
        mem_keys: Annotated[torch.Tensor, ..., 'N', 'K'] | None,
        mem_values: Annotated[torch.Tensor, ..., 'N', 'V'] | None,
    ) -> Annotated[torch.Tensor, ..., 'T', 'D']:
        if self.memory is not None:
            assert mem_values is not None and mem_keys is not None

        k = self.pre_norm(x)
        attn = self.attn(k, k, k)

        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        if self.memory is not None:
            mem = self.memory(k, mem_keys, mem_values)
            if self.mem_dropout is not None:
                mem = self.mem_dropout(mem)
            attn = attn + mem

        r = k + attn
        m = self.post_norm(r)
        m = self.mlp(m)

        if self.dropout is not None:
            m = self.dropout(m)

        return r + m


class MemaModel(nn.Module):
    config: MemaConfig

    embedding: nn.Embedding
    head: nn.Linear

    blocks: nn.ModuleList
    last_norm: RMSNorm | LayerNorm

    def __init__(self, config: MemaConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.last_norm = LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.vocab_size)

        self.layers = nn.ModuleList()
        for i in range(self.config.n_layers):
            memory = (
                i % self.config.mem_layer_spacing == 0
                and i < self.config.n_layers * self.config.n_mem_layers  # noqa
            )

            layer = MemaBlock(
                dim=self.config.d_model,
                memory=memory,
                dropout=self.config.dropout,
                attn_dropout=self.config.attn_dropout,
                mem_dropout=self.config.mem_dropout if memory else None,
            )

            self.layers.append(layer)

    def forward(
        self, tokens: Annotated[torch.Tensor, ..., 'T']
    ) -> Annotated[torch.Tensor, ..., 'T', 'V']:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.last_norm(x)
        return self.head(x)
