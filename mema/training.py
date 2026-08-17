from __future__ import annotations

from typing import Annotated

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange

logger = get_logger(__name__, log_level='INFO')


def cross_entropy_loss(
    logits: Annotated[torch.Tensor, ..., 'T', 'V'],
    targets: Annotated[torch.Tensor, ..., 'T'],
) -> Annotated[torch.Tensor, ..., 'T', 'V']:
    logits = rearrange(logits, '... V -> V ...')
    return F.cross_entropy(logits, targets)
