# expert_predictor.py
import torch

class ExpertPredictor:
    def __init__(self, predictor_model, n_slots):
        """
        predictor_model: a V6Model instance that takes (ht, Et, layer) and returns logits.
        """
        self.model = predictor_model
        self.n_slots = n_slots

    @torch.no_grad()
    def predict_next(self, layer, ht, Et):
        """
        Predict the top‑k expert IDs for the NEXT token at the given layer.
        ht: hidden states (batch, D) for current token.
        Et: current expert mask (batch, E) for current token.
        Returns: (batch, top_k) expert IDs, and (batch, E) scores (logits).
        """
        self.model.eval()
        logits = self.model(ht, Et, layer)  # (batch, E)
        _, top_ids = logits.topk(self.n_slots, dim=-1)  # (batch, top_k)
        return top_ids, logits