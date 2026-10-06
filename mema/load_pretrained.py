from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeAlias

import safetensors
import torch
import torch.nn as nn


def load_from_pretrained_gpt_neox() -> None:
    raise NotImplementedError


def convert_from_gpt_neox(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    raise NotImplementedError


@dataclass
class ParameterTree:
    root: str | None
    children: dict[str, ParameterTree]
    value: torch.Tensor | None


# TODO we might need to change the tree topology
class NodeVisitor(metaclass=ABCMeta):
    @abstractmethod
    def root(self, root: str | None) -> str | None: ...

    @abstractmethod
    def visitor(self, id: str) -> NodeVisitor: ...

    def visit(self, node: ParameterTree) -> ParameterTree:
        children: dict[str, ParameterTree] = {}
        for id, child in node.children.items():
            children[id] = self.visitor(id).visit(child)

        return ParameterTree(self.root(node.root), children, node.value)


class IdentityVisitor(NodeVisitor):
    def root(self, root: str | None) -> str | None:
        return root

    def visitor(self, id: str) -> NodeVisitor:
        return IdentityVisitor()


class GPTNeoXVisitor(IdentityVisitor): ...
