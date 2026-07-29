"""DynamicExpertLoader — model loader for dynamic expert weight management.

Registers the ``"dynamic_expert"`` load format. After loading model weights
normally, it moves expert weights to pinned CPU memory, creates
``DynamicRoutedExperts`` instances, and sets up predictor-driven prefetching.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from dynamic_experts.config import DynamicExpertConfig
from dynamic_experts.dynamic_routed_experts import DynamicRoutedExperts
from dynamic_experts.expert_predictor import ExpertPredictor
from dynamic_experts.expert_prefetcher import ExpertPrefetcher
from dynamic_experts.file_format import (
    DynamicExpertFileReader,
    write_dynamic_expert_file,
)
from dynamic_experts.v6_model import V6Model
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe import layer as moe_layer
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


class DynamicExpertLoader(BaseModelLoader):
    """Model loader enabling dynamic expert weight management.

    1. Loads weights normally via ``DefaultModelLoader``.
    2. Extracts all expert weight tensors, writes them into a byte-aligned
       file, and memory-maps it for zero-copy reads.
    3. Replaces each MoE layer's ``RoutedExperts`` GPU fused tensors with
       *n_slots*-sized versions backed by per-layer ``EagerExpertCache``.
    4. Optionally creates and attaches ``ExpertPredictor`` / ``ExpertPrefetcher``.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        self._dyn_config = DynamicExpertConfig(**{
            k: v for k, v in extra.items()
            if k in DynamicExpertConfig.__dataclass_fields__
        })
        self._default_loader = DefaultModelLoader(load_config)

    def download_model(self, *args, **kwargs) -> None:
        self._default_loader.download_model(*args, **kwargs)

    def load_model(self, *args, **kwargs) -> torch.nn.Module:
        moe_layer.set_dynamic_expert_config(self._dyn_config)
        model: torch.nn.Module = self._default_loader.load_model(*args, **kwargs)
        self._post_load(model)
        moe_layer.set_dynamic_expert_config(None)
        return model

    def load_weights(self, model, *args, **kwargs) -> None:
        self._default_loader.load_weights(model, *args, **kwargs)

    # ── internals ──────────────────────────────────────────────────────

    def _post_load(self, model: torch.nn.Module) -> None:
        layers = self._find_dynamic_layers(model)
        if not layers:
            return

        first = layers[0]
        n_layers = len(layers)
        n_experts = first._real_local_experts

        param_specs = self._discover_expert_params(first)
        tmp_path = self._model_file_path("expert_weights.bin")

        def _expert_generator():
            for li, re in enumerate(layers):
                for eid in range(n_experts):
                    tensors = tuple(
                        getattr(re, ps.name).data[eid].cpu()
                        for ps in param_specs
                    )
                    yield (li, eid, tensors)

        write_dynamic_expert_file(
            path=tmp_path, expert_weights=_expert_generator(),
            n_layers=n_layers, n_experts=n_experts, params=param_specs,
        )

        reader = DynamicExpertFileReader(tmp_path)
        readers = {
            ps.name: (lambda l, e, nm=ps.name: reader.read_expert_copy(l, e)[nm])
            for ps in param_specs
        }

        for li, re in enumerate(layers):
            layer_readers = {
                pname: (lambda e, _l=li, _r=r: _r(_l, e))
                for pname, r in readers.items()
            }
            re.shrink_to_slots(layer_readers)

        if self._dyn_config.enable_predictor:
            self._setup_predictor(model, layers, n_experts)

    def _setup_predictor(
        self, model, layers, n_experts,
    ) -> None:
        from vllm.model_executor.models.interfaces import MixtureOfExperts

        if not isinstance(model, MixtureOfExperts):
            return
        try:
            D = model.get_hidden_size()
            L = len(layers)
            E = n_experts
        except (AttributeError, NotImplementedError):
            return

        v6 = V6Model(D=D, L=L, E=E)
        pred_path = self._model_file_path("predictor.pt")
        if pred_path.exists():
            v6.load_state_dict(torch.load(pred_path, map_location="cpu", weights_only=True))

        predictor = ExpertPredictor(predictor_model=v6, n_slots=self._dyn_config.n_slots)
        self._predictor = predictor
        self._predictor_path = pred_path

        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        for _, runner in [
            (n, m) for n, m in model.named_modules() if isinstance(m, MoERunner)
        ]:
            re = getattr(runner, "routed_experts", None)
            if not isinstance(re, DynamicRoutedExperts):
                continue
            primary = re._primary_cache
            if primary is None:
                continue
            pf = ExpertPrefetcher(
                predictor=predictor, cache=primary,
                n_layers=1, top_k=self._dyn_config.predictor_top_k,
            )
            if hasattr(runner, "attach_prefetcher"):
                runner.attach_prefetcher(pf)

    def save_predictor(self) -> None:
        pred = getattr(self, "_predictor", None)
        path = getattr(self, "_predictor_path", None)
        if pred is not None and path is not None:
            torch.save(pred.predictor_model.state_dict(), path)

    @staticmethod
    def _discover_expert_params(re: DynamicRoutedExperts) -> list:
        from dynamic_experts.file_format import ParamSpec

        specs = []
        for pname, param in re.named_parameters():
            if param.ndim < 2:
                continue
            if param.shape[0] != re._real_local_experts:
                continue
            if param.stride(0) == 0:
                continue  # broadcast view (e.g. w13_input_scale) — skip
            dtype = param.dtype
            if dtype in (torch.int64, torch.bool, torch.int32):
                continue
            specs.append(ParamSpec(
                name=pname, shape=tuple(param.shape[1:]), dtype=dtype,
            ))
        return specs

    def _model_file_path(self, filename: str) -> Path:
        download_dir = self.load_config.download_dir
        if download_dir:
            return Path(download_dir) / filename
        return Path(tempfile.gettempdir()) / f"vllm_dyn_{os.getpid()}_{filename}"

    @staticmethod
    def _find_dynamic_layers(model: torch.nn.Module) -> list[DynamicRoutedExperts]:
        return [
            mod for _, mod in model.named_modules()
            if isinstance(mod, DynamicRoutedExperts)
        ]
