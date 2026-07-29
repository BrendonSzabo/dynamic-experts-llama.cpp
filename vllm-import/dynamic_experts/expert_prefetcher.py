# expert_prefetcher.py
import torch


class ExpertPrefetcher:
    def __init__(self, predictor, cache, n_layers, top_k):
        self.predictor = predictor
        self.cache = cache
        self.n_layers = n_layers
        self.top_k = top_k
        # Store predicted expert IDs for the next token (per layer)
        self.next_experts = [None] * n_layers
        self.next_scores = [None] * n_layers

    @torch.no_grad()
    def predict_and_prefetch(self, hidden_states, current_masks, layers=None):
        """
        Run the predictor for the given layers and schedule async prefetch.
        hidden_states: list of tensors (batch, D), one per layer (current token).
        current_masks: list of tensors (batch, E), one per layer (current token).
        If layers is None, process all layers.
        """
        if layers is None:
            layers = range(self.n_layers)
        for l in layers:
            ht = hidden_states[l]
            Et = current_masks[l]
            top_ids, scores = self.predictor.predict_next(l, ht, Et)
            self.next_experts[l] = top_ids  # (batch, top_k)
            self.next_scores[l] = scores
            # Prefetch unique experts across the whole batch for this layer
            unique_ids = torch.unique(top_ids)  # shape (num_unique,)
            self.cache.prefetch(l, unique_ids, scores.mean(dim=0))  # mean score as utility

    def get_weights_for_layer(self, layer, expert_ids):
        """
        Called inside the MoE forward to retrieve the actual expert weights.
        expert_ids: (batch,) tensor of expert indices needed by the LLM for this token.
        Returns: (batch, *weight_shape) tensor.
        """
        return self.cache.get_weights(layer, expert_ids)

    def get_next_experts(self, layer):
        """Return the predicted expert IDs for the next token (used for prefetching)."""
        return self.next_experts[layer]