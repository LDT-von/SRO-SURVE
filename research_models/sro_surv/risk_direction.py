"""Opposite-half, training-only directions for the pathology read loss."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def risk_directions(means, times, events):
    """Return a fixed axis per training row, in the supplied patch coordinates.

    The Cox score at zero (Breslow ties) supplies one outcome per patient.
    A ridge solve on standardized bag means shrinks unstable directions.
    Quadratic work is in the training patient count, not patch count or width.
    """
    means = means.detach().to(device="cpu", dtype=torch.float64)
    times = times.detach().cpu()
    events = events.detach().cpu().bool()
    n = len(means)
    if means.ndim != 2 or times.shape != (n,) or events.shape != (n,) or n < 2:
        raise ValueError("paired training bag means and outcomes required")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(197))
    halves = order.tensor_split(2)
    directions = torch.zeros_like(means)
    for fit, apply in (halves, halves[::-1]):
        e, t = events[fit], times[fit]
        if not e.any() or len(fit) < 2:
            continue
        # Row r lists all patients at risk when the r-th event occurs.
        at_risk = (t[None, :] >= t[e][:, None]).to(means)
        residual = e.to(means) - (at_risk / at_risk.sum(1, keepdim=True)).sum(0)
        x = means[fit] - means[fit].mean(0)
        sd = x.square().mean(0).sqrt()
        # Remove constant features and give near-constant ones a scale floor.
        scale = sd.clamp_min(1e-4)
        x = x / scale
        gram = x @ x.T + len(fit) * torch.eye(len(fit), dtype=x.dtype)
        axis = (x.T @ torch.linalg.solve(gram, residual)) / scale
        directions[apply] = F.normalize(axis, dim=0)
    return directions
