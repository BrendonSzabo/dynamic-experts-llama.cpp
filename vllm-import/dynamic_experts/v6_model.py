from dataclasses import dataclass

import torch

class V6Model(torch.nn.Module):
    def __init__(self, D, L, E, H=16):
        '''D => num hidden states; L => num layers; E => num experts'''
        super().__init__()
        self.L = L
        self.trunk = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.ReLU())
        self.W1 = torch.nn.Parameter(torch.empty(L, H, H + E))
        self.b1 = torch.nn.Parameter(torch.zeros(L, H))
        self.W2 = torch.nn.Parameter(torch.empty(L, E, H))
        self.b2 = torch.nn.Parameter(torch.zeros(L, E))
        torch.nn.init.kaiming_uniform_(self.W1)
        torch.nn.init.kaiming_uniform_(self.W2)
        self.opt = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)
        self.loss_progression = []
        self.metric_progression = []

    def forward(self, ht, Et, layer):
        z = self.trunk(ht)
        x = torch.cat([z, Et], dim=-1)
        h = torch.nn.functional.relu(torch.einsum('nd,lhd->nlh', x, self.W1) + self.b1)
        out = torch.einsum('nlh,leh->nle', h, self.W2) + self.b2
        return out[torch.arange(ht.shape[0]), layer]
    
    def metric(self, logits, targets, num_slots):
        if num_slots == targets:
            """Jaccard index (intersection over union) for top‑k predictions."""
            _, top = logits.topk(num_slots, dim=-1)
            preds = torch.zeros_like(logits).scatter_(-1, top, 1.0)
            intersection = (preds * targets).sum(-1)
            union = ((preds + targets) > 0).float().sum(-1)
            return (intersection / (union + 1e-8)).mean().item()
        """Recall@k (fraction of active experts recalled in top‑k predictions)."""
        _, top = logits.topk(num_slots, dim=-1)
        preds = torch.zeros_like(logits).scatter_(-1, top, 1.0)
        hits = (preds * targets).sum(-1)
        metric = (hits / targets.sum(-1).clamp(1)).mean().item()
        self.metric_progression.append(metric)
        return metric
        
    @torch.no_grad()
    def predict(self, ht, Et, layer):
        """
        Predict the next expert mask.

        Args:
            ht: hidden states, shape (N, D)
            Et: current expert mask, shape (N, E)
            layer: layer indices (long), shape (N,)

        Returns:
            Predicted expert mask (indices, boolean mask, or probabilities)
        """
        self.eval()
        logits = self.forward(ht, Et, layer)
        return torch.sigmoid(logits)
    
    def train_step(self, ht, Et, layer, targets):
        """
        Perform one training step:
            1. forward pass
            2. compute loss
            3. backward pass
            4. optimizer step
        Returns the loss value (float).
        """
        self.train()
        logits = self.forward(ht, Et, layer)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        self.loss_progression.append(loss.item())
        return loss.item()
    
@dataclass
class V6ModelTrainingData:
    h_t: torch.Tensor
    e_t: torch.Tensor
    l: int
    y: torch.Tensor
    
@dataclass
class V6ModelPredictionData:
    h_t: torch.Tensor
    e_t: torch.Tensor
    l: torch.Tensor