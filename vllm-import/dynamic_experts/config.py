"""Configuration for dynamic expert loading."""

from dataclasses import dataclass, field


@dataclass
class DynamicExpertConfig:
    """Configuration controlling dynamic expert weight management.

    Args:
        n_slots: Number of expert weight slots to keep GPU-resident per layer.
            Experts beyond this count are kept on pinned CPU and swapped in
            on demand. Set equal to num_local_experts for full residency
            (no memory savings, but enables lazy loading + prefetch).
        enable_predictor: Whether to use a learned predictor (V6Model) for
            expert prefetching. When False, a FIFO heuristic is used instead.
        predictor_top_k: Number of top experts the predictor returns per token.
        train_on_forward: Whether to collect training data during forward
            passes for online predictor fine-tuning.
        predictor_device: Device for the predictor model (default: same as
            model weights).
    """

    n_slots: int = 32
    """Number of GPU-resident expert slots per MoE layer."""

    enable_predictor: bool = False
    """Enable V6Model-based expert prediction for prefetching."""

    predictor_top_k: int = 4
    """Top-k experts to predict and prefetch per token."""

    train_on_forward: bool = False
    """Collect training data during forward passes for online learning."""

    predictor_device: str | None = None
    """Device for the predictor model. Defaults to the model's device."""

    def __post_init__(self):
        if self.enable_predictor and self.n_slots < self.predictor_top_k:
            raise ValueError(
                f"n_slots ({self.n_slots}) must be >= predictor_top_k "
                f"({self.predictor_top_k}) when predictor is enabled."
            )
