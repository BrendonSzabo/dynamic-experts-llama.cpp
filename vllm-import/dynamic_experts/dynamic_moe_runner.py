"""DynamicMoERunner — MoE runner with predictor-driven expert prefetching.

Extends MoERunner to collect predictor training data from each forward pass
and trigger expert prefetch for the next token.
"""

from __future__ import annotations

import torch

from dynamic_experts.config import DynamicExpertConfig
from dynamic_experts.expert_prefetcher import ExpertPrefetcher
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


class DynamicMoERunner(MoERunner):
    """MoERunner that calls predict-and-prefetch each forward, collects
    training data, and exposes ``train_on_drain()`` for post-inference."""

    def __init__(self, *args, dynamic_config: DynamicExpertConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dyn_config = dynamic_config or DynamicExpertConfig()
        self._prefetcher: ExpertPrefetcher | None = None
        self._collect_hidden: list[torch.Tensor] = []
        self._collect_masks: list[torch.Tensor] = []

    def attach_prefetcher(self, pf: ExpertPrefetcher) -> None:
        self._prefetcher = pf

    def _apply_quant_method(self, *args, **kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        rl = args[1] if len(args) > 1 else kwargs.get("router_logits")

        result = super()._apply_quant_method(*args, **kwargs)

        if hs is not None and rl is not None and not self.routed_experts.quant_method.is_monolithic:
            _, topk_ids = self.router.select_experts(
                hidden_states=hs, router_logits=rl,
            )
            mask = torch.zeros(
                hs.shape[0], self.routed_experts.global_num_experts,
                dtype=torch.float32, device=hs.device,
            )
            mask.scatter_(1, topk_ids, 1.0)

            if self._prefetcher is not None:
                self._prefetcher.predict_and_prefetch(
                    hidden_states=[hs], current_masks=[mask], layers=[0],
                )

            if self._dyn_config.train_on_forward:
                self._collect_hidden.append(hs.detach().clone())
                self._collect_masks.append(mask)

        return result

    def train_on_drain(self) -> float | None:
        """Drain collected data and call V6Model.train_step().

        Returns the average loss across all training steps, or None
        if no data was collected.
        """
        if not self._collect_hidden or not self._collect_masks:
            return None
        if self._prefetcher is None:
            return None

        model = self._prefetcher.predictor.model
        losses: list[float] = []
        for hs, mask in zip(self._collect_hidden, self._collect_masks):
            # Predict for next token: input current (hs, mask), target = mask
            # (self-supervised: the predictor learns what the router chose)
            loss = model.train_step(
                ht=hs, Et=mask,
                layer=torch.zeros(hs.shape[0], dtype=torch.long, device=hs.device),
                targets=mask,
            )
            losses.append(loss)

        self._collect_hidden.clear()
        self._collect_masks.clear()
        return sum(losses) / len(losses) if losses else None
