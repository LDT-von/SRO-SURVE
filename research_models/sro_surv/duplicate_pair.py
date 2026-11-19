"""Training-only ranking loss for a pair of nearly identical pathology reads."""

import torch
from torch.nn import functional as F


def duplicate_pair_loss(logits, reads, features, valid, axes=None):
    """Choose groups from detached input features; optimize only relative logits.

    Inputs are [B,K,N] competitive logits and row-normalized reads, [B,N,D]
    pathology features, and a real-token mask. Each patient contributes equally.
    """
    costs = []
    if axes is not None and axes.shape != (features.shape[0], features.shape[-1]):
        raise ValueError(
            "expected one frozen direction per patient in pathology feature space"
        )
    for b, (z, a, x, mask) in enumerate(
        zip(logits, reads.detach(), features.detach(), valid)
    ):
        zero = z.sum() * 0
        if a.shape[0] < 2 or mask.sum() < 8:
            costs.append(zero)
            continue
        x, a, z = x[mask], a[:, mask], z[:, mask]
        with torch.no_grad():
            pairs = torch.triu_indices(len(a), len(a), offset=1, device=a.device)
            root = a.sqrt()
            bc = (root @ root.T)[pairs[0], pairs[1]]
            best = bc.argmax()
            if bc[best] <= 0.90:
                costs.append(zero)
                continue
            i, j = pairs[:, best]
            # Use a robust spread criterion: a lone extreme token cannot
            # produce an interquartile range when all other tokens coincide.
            if axes is not None:
                x = (x @ axes[b].detach()).unsqueeze(-1)
            quartiles = torch.quantile(x, x.new_tensor([0.25, 0.75]), dim=0)
            span = quartiles[1] - quartiles[0]
            d = span.argmax()
            if span[d] <= 1e-6:
                costs.append(zero)
                continue
            common = root[i] * root[j]
            lo = common * (x[:, d] <= quartiles[0, d])
            hi = common * (x[:, d] >= quartiles[1, d])
            if torch.minimum(lo.sum(), hi.sum()) < 0.10 * common.sum():
                costs.append(zero)
                continue
            contrast = hi / hi.sum() - lo / lo.sum()
        gap = (contrast * (z[i] - z[j])).sum()
        # Explicit tie-break: abs(gap) would supply a zero derivative at zero.
        orientation = torch.where(gap.detach() >= 0, 1.0, -1.0)
        costs.append(F.relu(1 - orientation * gap).square())
    return torch.stack(costs).mean()
