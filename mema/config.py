from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlibiConfig:
    ...


@dataclass(frozen=True)
class RoPEConfig:
    ...


@dataclass(frozen=True)
class AttentionConfig:
    dropout: float = 0.0
    rotary_pos_emb: bool = False
    relative_position: AlibiConfig | None = None
    softmax_scale: float | None = None


@dataclass(frozen=True)
class InitConfig:
    load_pretrained: bool = False


# same defaults as https://github.com/mosaicml/llm-foundry
@dataclass(frozen=True)
class MemaConfig:
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 24
    exp_ration: int = 4
    max_seq_len: int = 2048
    vocab_size: int = 50368
    attn_config: AttentionConfig = AttentionConfig()
    init_config: InitConfig = InitConfig()
