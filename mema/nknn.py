from __future__ import annotations

from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from einops import rearrange


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

    k: int
    temp: float
    dim: int
    feature: int
    mask: bool

    no_values: bool = False

    def __init__(
        self,
        k: int,
        dim: int,
        temp: float,
        feature: int | None = None,
        mask: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.temp = temp

        self.dim = dim
        self.feature = feature or dim
        if feature is None:
            self._no_values = True

        self.mask = mask

    def forward(
        self,
        query: Annotated[torch.Tensor, ..., 'B', 'T', 'D'],
        keys: Annotated[torch.Tensor, ..., 'N', 'D'],
        values: Annotated[torch.Tensor, ..., 'N', 'F'] | None = None,
    ) -> Annotated[torch.Tensor, ..., 'B', 'T', 'K', 'F']:
        if values is None:
            assert self.no_values
            values = keys
        assert query.shape[-1] == keys.shape[-2] == self.dim
        assert values.shape[-2] == self.feature
        assert keys.shape[-1] == values.shape[-1]

        sims = self._similarity(query, keys)
        omega = self._compute_omega(s=sims, k=self.k, t=self.temp)

        k_nearest = einsum(
            omega, values, '... B N T K, ... N F -> ... B T K F'
        )  # TODO this is not well defined
        return k_nearest

    def _similarity(
        self,
        query: Annotated[torch.Tensor, ..., 'B', 'T', 'D'],
        key: Annotated[torch.Tensor, ..., 'N', 'D'],
    ) -> Annotated[torch.Tensor, ..., 'B', 'N', 'T']:
        """
        "B": inner batch that shares same keys
        """
        assert query.shape[-1] == key.shape[-1] == self.dim
        assert query.shape[-3] == key.shape[-3]
        assert query.shape[:-3] == key.shape[:-3]  # check outer batch dimensions

        sim = -einsum(query, key, '... B T D, ... N D -> ... B N T') / (self.dim**0.5)
        return sim

    def _compute_omega(
        self, s: Annotated[torch.Tensor, ..., 'B', 'N', 'T'], k: int, t: float
    ) -> Annotated[torch.Tensor, ..., 'B', 'N', 'T', 'K']:
        alpha = F.softmax(s, dim=-1)
        omega = torch.empty(*s.shape, k)

        omega[..., 0] = F.softmax(alpha / t, dim=-1)
        for i in range(1, k):
            alpha = alpha + torch.log(1 - omega[..., i - 1])
            omega[..., i] = F.softmax(alpha / t, dim=-1)

        return omega
