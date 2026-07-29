"""DynamicRoutedExperts — n_slots GPU-resident expert weights.

Each layer owns two EagerExpertCache instances (w13 + w2) managing
*n_slots* GPU slots.  ``w13_weight`` and ``w2_weight`` directly
reference cache slot storage — no copy between cache and params.
"""

from __future__ import annotations

import torch

from dynamic_experts.config import DynamicExpertConfig
from dynamic_experts.eager_expert_cache import EagerExpertCache, SENTINEL
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class DynamicRoutedExperts(RoutedExperts):
    def __init__(self, *args, dynamic_config: DynamicExpertConfig | None = None, **kwargs):
        self._dyn_config = dynamic_config or DynamicExpertConfig()
        self._caches: dict[str, EagerExpertCache] = {}
        self._shrunk = False
        self._real_local_experts: int = 0
        super().__init__(*args, **kwargs)
        self._real_local_experts = self.local_num_experts

    @property
    def _primary_cache(self) -> EagerExpertCache | None:
        for c in self._caches.values():
            return c
        return None

    def shrink_to_slots(self, readers: dict[str, callable]) -> None:
        if self._shrunk or not readers:
            return
        n_slots = self._dyn_config.n_slots
        device = next(
            (p.device for p in self.parameters() if p.device.type == "cuda"),
            torch.device("cuda"),
        )
        n_exp = self._real_local_experts

        for pname, reader_fn in readers.items():
            param = getattr(self, pname, None)
            if param is None:
                continue
            shape = tuple(param.shape[1:])
            dtype = param.dtype
            cache = EagerExpertCache(
                n_layers=1, n_experts=n_exp, n_slots=n_slots,
                weight_shape=shape, dtype=dtype,
                reader_fn=lambda l, e, r=reader_fn: r(e), device=device,
            )
            self._caches[pname] = cache
            new_param = torch.nn.Parameter(cache.slots[0], requires_grad=False)
            setattr(self, pname, new_param)
            self.register_parameter(pname, new_param)

        self.local_num_experts = n_slots
        self._shrunk = True

    def _slot_of(self, expert_ids: torch.Tensor) -> torch.Tensor:
        c = self._primary_cache
        return expert_ids if c is None else c.slot_of[0, expert_ids]

    def _ensure_experts(self, expert_ids: torch.Tensor) -> None:
        for c in self._caches.values():
            c.ensure(0, expert_ids)

    def forward_modular(
        self, x, topk_weights, topk_ids, **kwargs,
    ) -> torch.Tensor:
        if self._shrunk:
            self._ensure_experts(topk_ids.unique())
            return super().forward_modular(
                x, topk_weights, self._slot_of(topk_ids), **kwargs,
            )
        return super().forward_modular(x, topk_weights, topk_ids, **kwargs)

    def forward_monolithic(
        self, x, router_logits=None, input_ids=None,
    ) -> torch.Tensor:
        if not self._shrunk:
            return super().forward_monolithic(
                x, router_logits=router_logits, input_ids=input_ids,
            )
        if self._dyn_config.n_slots >= self._real_local_experts:
            return super().forward_monolithic(
                x, router_logits=router_logits, input_ids=input_ids,
            )
        raise NotImplementedError(
            "DynamicRoutedExperts does not support monolithic kernels "
            f"when n_slots ({self._dyn_config.n_slots}) < n_experts "
            f"({self._real_local_experts}). Use modular kernels."
        )
