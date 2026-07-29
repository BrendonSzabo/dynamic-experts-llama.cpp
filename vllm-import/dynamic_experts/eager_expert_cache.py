"""EagerExpertCache — slot-based GPU cache for expert weights.

Manages *n_slots* GPU-resident expert weight slots per layer. Missing experts
are fetched from a pluggable reader (mmap, pinned tensor, or pread-based)
and async-copied to GPU via a dedicated CUDA stream.

Supports score-based eviction: when all slots are full and a new expert is
requested, the expert with the lowest utility score is evicted.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

SENTINEL = -1


class EagerExpertCache:
    """GPU cache with slot-based expert weight management.

    Args:
        n_layers: Number of MoE layers this cache serves.
        n_experts: Total number of experts (per layer).
        n_slots: Number of GPU-resident slots (per layer).
        weight_shape: Per-expert weight tensor shape, e.g. ``(E_up, hidden)``.
        dtype: Data type of weight tensors.
        reader_fn: Callable ``(layer, expert) -> Tensor`` that returns a CPU
            tensor containing one expert's weights. The tensor should be
            contiguous and may be mmap-backed.
        device: CUDA device for the L1 slots.
    """

    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        n_slots: int,
        weight_shape: tuple[int, ...],
        dtype: torch.dtype,
        reader_fn: Callable[[int, int], torch.Tensor],
        device: torch.device | str = "cuda",
    ):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_slots = n_slots
        self.shape = weight_shape
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self._reader = reader_fn

        # ── L1 GPU storage ──
        self.slots = torch.zeros(
            n_layers, n_slots, *weight_shape,
            dtype=dtype, device=self.device,
        )

        # ── Mapping tables (on GPU, for fast indexing) ──
        self.slot_of = torch.full(
            (n_layers, n_experts), SENTINEL,
            dtype=torch.int32, device=self.device,
        )
        self.expert_in = torch.full(
            (n_layers, n_slots), SENTINEL,
            dtype=torch.int32, device=self.device,
        )
        self.slot_in_use = torch.zeros(
            n_layers, n_slots, dtype=torch.bool, device=self.device,
        )

        # ── Async copy infrastructure ──
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_events: list[list[torch.cuda.Event | None]] = [
            [None] * n_slots for _ in range(n_layers)
        ]

    # ── slot management ─────────────────────────────────────────────────

    def _find_free_slot(
        self, layer: int, scores: torch.Tensor | None = None
    ) -> int:
        """Allocate a GPU slot, evicting the least-useful expert if needed.

        Args:
            layer: Layer index within this cache.
            scores: Shape ``(n_experts,)`` tensor with per-expert utility
                scores. Higher = more useful. If None, FIFO eviction is used.

        Returns:
            Slot index (0-based).
        """
        free = torch.where(~self.slot_in_use[layer])[0]
        if free.numel() > 0:
            slot = int(free[0].item())
            self.slot_in_use[layer, slot] = True
            return slot

        # Eviction path.
        loaded_mask = self.slot_of[layer] != SENTINEL
        loaded_experts = torch.where(loaded_mask)[0]
        if scores is not None:
            victim_idx = int(torch.argmin(scores[loaded_experts]).item())
            victim_expert = int(loaded_experts[victim_idx].item())
        else:
            victim_expert = int(loaded_experts[0].item())

        slot = int(self.slot_of[layer, victim_expert].item())
        # Wait for any in-flight copy on this slot before reuse.
        ev = self.copy_events[layer][slot]
        if ev is not None:
            ev.synchronize()
            self.copy_events[layer][slot] = None

        self.slot_of[layer, victim_expert] = SENTINEL
        self.expert_in[layer, slot] = SENTINEL
        # Slot stays in_use — we reuse it immediately.
        return slot

    # ── public API ──────────────────────────────────────────────────────

    def prefetch(
        self,
        layer: int,
        expert_ids: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> None:
        """Ensure *expert_ids* are in L1, starting async copies for missing ones.

        Args:
            layer: Layer index.
            expert_ids: 1D int tensor (GPU) of expert IDs to load.
            scores: Optional ``(n_experts,)`` utility tensor for eviction.
        """
        current_slots = self.slot_of[layer, expert_ids]
        missing_mask = current_slots == SENTINEL
        if not missing_mask.any():
            return

        missing_experts = expert_ids[missing_mask].unique()
        for eid in missing_experts.tolist():
            slot = self._find_free_slot(layer, scores)
            self.slot_of[layer, eid] = slot
            self.expert_in[layer, slot] = eid

            cpu_weight = self._reader(layer, eid)
            with torch.cuda.stream(self.copy_stream):
                self.slots[layer, slot].copy_(cpu_weight, non_blocking=True)
                event = self.copy_stream.record_event()
            self.copy_events[layer][slot] = event

    def ensure(
        self,
        layer: int,
        expert_ids: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> None:
        """Like :meth:`prefetch` but also blocks until all copies finish.

        Use when the weights must be available on the default stream
        immediately after this call.
        """
        self.prefetch(layer, expert_ids, scores)
        self._wait_all(layer)

    def get_weights(
        self, layer: int, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return expert weight tensors, blocking only for in-flight copies.

        Args:
            layer: Layer index.
            expert_ids: ``(batch,)`` int tensor (GPU) of expert IDs.

        Returns:
            ``(batch, *weight_shape)`` tensor on GPU.
        """
        slots = self.slot_of[layer, expert_ids]
        missing = slots == SENTINEL
        if missing.any():
            self._blocking_load(layer, expert_ids[missing])

        for s in slots.unique().tolist():
            if s == SENTINEL:
                continue
            ev = self.copy_events[layer][s]
            if ev is not None and not ev.query():
                ev.synchronize()
                self.copy_events[layer][s] = None

        return self.slots[layer, slots]

    # ── internal ────────────────────────────────────────────────────────

    def _wait_all(self, layer: int) -> None:
        for s in range(self.n_slots):
            ev = self.copy_events[layer][s]
            if ev is not None and not ev.query():
                ev.synchronize()
                self.copy_events[layer][s] = None

    def _blocking_load(self, layer: int, expert_ids: torch.Tensor) -> None:
        """Synchronous fallback for experts not yet prefetched."""
        for eid in expert_ids.tolist():
            slot = self._find_free_slot(layer)
            self.slot_of[layer, eid] = slot
            self.expert_in[layer, slot] = eid
            cpu_weight = self._reader(layer, eid)
            self.slots[layer, slot].copy_(cpu_weight, non_blocking=False)
            self.copy_events[layer][slot] = None
