"""Dynamic Expert Loading for vLLM.

Provides lazy-loading and predictive prefetching of MoE expert weights,
reducing GPU memory pressure for large mixture-of-experts models.
"""

from dynamic_experts.config import DynamicExpertConfig
from dynamic_experts.dynamic_loader import DynamicExpertLoader
from dynamic_experts.dynamic_moe_runner import DynamicMoERunner
from dynamic_experts.dynamic_routed_experts import DynamicRoutedExperts
from dynamic_experts.eager_expert_cache import EagerExpertCache
from dynamic_experts.expert_predictor import ExpertPredictor
from dynamic_experts.expert_prefetcher import ExpertPrefetcher
from dynamic_experts.file_format import (
    DynamicExpertFileReader,
    write_dynamic_expert_file,
)
from dynamic_experts.v6_model import V6Model, V6ModelTrainingData, V6ModelPredictionData

__all__ = [
    "DynamicExpertConfig",
    "DynamicExpertLoader",
    "DynamicMoERunner",
    "DynamicRoutedExperts",
    "DynamicExpertFileReader",
    "write_dynamic_expert_file",
    "EagerExpertCache",
    "ExpertPredictor",
    "ExpertPrefetcher",
    "V6Model",
    "V6ModelTrainingData",
    "V6ModelPredictionData",
]
