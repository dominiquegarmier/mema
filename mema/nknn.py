from __future__ import annotations

from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


# Neural Nearest Neighbor Networks https://arxiv.org/abs/1810.12575
class NeuralKNearestNeighbor(nn.Module):
    """\
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
    """

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
