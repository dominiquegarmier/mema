from __future__ import annotations

from typing import Annotated
from collections.abc import Generator
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import TypeAlias

import torch
from datasets import Dataset


def similarity_batcher() -> None:
    raise NotImplementedError


def tokenize() -> None: ...


def tokenize_dataset(dataset: Dataset):
    dataset.map(tokenize)


TSeq: TypeAlias = Annotated[torch.Tensor, 'B', 'T', 'D']
TMemKeys: TypeAlias = Annotated[torch.Tensor, 'M', 'K']
TMemVals: TypeAlias = Annotated[torch.Tensor, 'M', 'V']


def sequencer(dataset: Iterable[str], seq_len: int) -> Generator[tuple[str, str]]:
    for sample in dataset:
        for i in range(len(sample) // seq_len):  # TODO +1 here?
            sub_sample = sample[i * seq_len : (i + 1) * seq_len + 1]  # noqa
            yield sub_sample[:-1], sub_sample[1:]


class Batches:
    def __iter__(self) -> Batches:
        raise NotImplementedError

    def __next__(self) -> tuple[TSeq, TSeq, TMemKeys, TMemVals]:
        raise NotImplementedError
