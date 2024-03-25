from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlibiConfig: ...


@dataclass(frozen=True)
class RoPEConfig: ...


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
    n_mem_layers: int = 4
    mem_layer_spacing: int = 4  # memory layer every n layers
    dropout: float = 0.0
    mem_dropout: float = 0.0
    attn_dropout: float = 0.0

    exp_ration: int = 4
    n_mem: int = 2048
    max_seq_len: int = 2048
    vocab_size: int = 50368

    attn_config: AttentionConfig = AttentionConfig()
    init_config: InitConfig = InitConfig()


def validate_config(config: MemaConfig) -> None:
    if config.d_model % config.n_heads != 0:
        raise ValueError('d_model must be divisible by n_heads')
    if config.n_mem_layers * config.mem_layer_spacing >= config.n_layers:
        raise ValueError('too many memory layers, and too much spacing between them')
