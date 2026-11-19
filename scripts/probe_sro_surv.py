"""Compare loss placement in the actual ten-read encoder on invented bags."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_models.sro_surv.model import SROSurv
from research_models.common_private_synergy.components.losses import discrete_hazard_nll
from slotspe_proof.slot_diversity_losses import slot_read_overlap_loss, warmup_factor


def bags(n, generator, axis):
    # Both train and evaluation use the same two population features.
    latent = torch.randn(n, 2, generator=generator)
    signs = torch.where(
        torch.rand(n, 48, generator=generator) < latent[:, :1].sigmoid(), 1.0, -1.0
    )
    p = signs.unsqueeze(-1) * axis + 0.5 * torch.randn(
        n, 48, len(axis), generator=generator
    )
    g = [
        latent[:, 1:] + torch.randn(n, 4, generator=generator) * 0.6 for _ in range(12)
    ]
    times = torch.exp(-latent.sum(-1) * 0.6 + torch.randn(n, generator=generator) * 0.6)
    censor = torch.rand(n, generator=generator) * 3 + 0.3
    obs = torch.minimum(times, censor)
    mask = torch.arange(48)[None] < torch.randint(32, 49, (n, 1), generator=generator)
    return (
        p,
        g,
        mask,
        torch.ones(n, len(g), dtype=torch.bool),
        signs,
        obs,
        times <= censor,
    )


def add_nuisance(b, axis, gen):
    # Outcome-independent two-state patch artifact. Only pathology is modified.
    direction = torch.zeros_like(axis)
    direction[0] = 1
    direction = F.normalize(direction - direction.dot(axis) * axis, dim=0)
    artifact = 2 * torch.randint(2, b[0].shape[:2], generator=gen).float() - 1
    return (b[0] + 3 * artifact.unsqueeze(-1) * direction, *b[1:]), artifact


def forward(model, b, ix, penalties=None, axes=None):
    p, g, pm, gm, _, _, _ = b
    return model(
        p[ix],
        [v[ix] for v in g],
        pm[ix],
        gm[ix],
        penalties=penalties,
        axes=None if axes is None else axes[ix],
    )


def scores(model, b, edges, artifact):
    model.eval()
    with torch.no_grad():
        z, parts, _ = forward(model, b, slice(None))
        result = {
            "NLL": float(
                discrete_hazard_nll(z, torch.bucketize(b[-2], edges, right=True), b[-1])
            )
        }
        from sksurv.metrics import concordance_index_censored

        risk = -(1 - z.sigmoid()).cumprod(-1).sum(-1)
        result["C_index"] = float(
            concordance_index_censored(b[-1].numpy(), b[-2].numpy(), risk.numpy())[0]
        )
        # Ground-truth mode labels are used only for evaluation.
        mode_fraction = (parts["p"][1] * (b[4][:, None, :] > 0)).sum(-1)
        result["p_mode_fraction_range"] = float(
            (mode_fraction.amax(1) - mode_fraction.amin(1)).mean()
        )
        pairs = torch.triu_indices(mode_fraction.shape[1], mode_fraction.shape[1], 1)

        def mean_gap(fraction):
            return float((fraction[:, pairs[0]] - fraction[:, pairs[1]]).abs().mean())

        result["p_mode_pair_gap"] = mean_gap(mode_fraction)
        if artifact is not None:
            artifact_fraction = (parts["p"][1] * (artifact[:, None, :] > 0)).sum(-1)
            result["p_artifact_pair_gap"] = mean_gap(artifact_fraction)
        for m, (slots, a) in parts.items():
            _, stats = slot_read_overlap_loss(a, b[2 if m == "p" else 3])
            vec = F.normalize(slots, dim=-1)
            pairs = torch.triu_indices(slots.shape[1], slots.shape[1], 1)
            result[m + "_BC10"] = stats["mean_pair_overlap"]
            result[m + "_cos10"] = float(
                (vec @ vec.transpose(1, 2))[:, pairs[0], pairs[1]].mean()
            )
        return result


def run(
    seed, nuisance_audit=False, confirmation=False, risk_only=False, sro_sweep=False
):
    gen = torch.Generator().manual_seed(
        (20261120 if confirmation else 20261119) if nuisance_audit else seed + 800
    )
    axis = F.normalize(torch.randn(16, generator=gen), dim=0)
    train, val = bags(64, gen, axis), bags(48, gen, axis)
    artifact = None
    if nuisance_audit:
        train, _ = add_nuisance(train, axis, gen)
        val, artifact = add_nuisance(val, axis, gen)
    edges = torch.quantile(train[-2][train[-1]], torch.tensor([0.25, 0.5, 0.75]))
    results = {}
    from research_models.sro_surv.risk_direction import (
        risk_directions,
    )

    p, g, mask, _, _, time, event = train
    means = (p * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    train_axes = risk_directions(means, time, event).to(p)
    arms = [
        ("off", []),
        ("last", [10]),
        ("first_last", [1, 10]),
        ("pair", [1, 10]),
        ("risk_pair", [1, 10]),
    ]
    margins = {}
    if sro_sweep:
        all_rounds = list(range(1, 11))
        arms = [
            ("off", []),
            ("first_last", [1, 10]),
            ("all_090", all_rounds),
            ("all_070", all_rounds),
            ("all_060", all_rounds),
        ]
        margins = {"all_070": 0.70, "all_060": 0.60}
    for arm, rounds in arms[-1:] if risk_only else arms:
        axes = train_axes if arm == "risk_pair" else None
        torch.manual_seed(seed)
        c = dict(
            width=32,
            bins=4,
            slots_per_modality=8,
            iterations=10,
            omics_dropout=0.25,
            read_loss_rounds=rounds,
            sro_margin=margins.get(arm, 0.90),
            read_loss=(
                "duplicate_pair"
                if arm == "pair"
                else "risk_pair"
                if arm == "risk_pair"
                else "sro"
            ),
        )
        model = SROSurv(16, [4] * 12, c)
        model.eval()
        probe = {}
        for step in [1, 10]:
            model.read_rounds = (step,)
            penalties = []
            forward(model, train, slice(0, 8), penalties, axes)
            grad = torch.autograd.grad(penalties[0], model.encoders["p"].queries)[0]
            probe[str(step)] = float(grad.norm())
        model.read_rounds = tuple(rounds)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        order_rng = torch.Generator().manual_seed(seed + 99)
        for epoch in range(1, 13):
            model.train()
            for ix in torch.randperm(64, generator=order_rng).split(8):
                penalties = []
                z, _, _ = forward(model, train, ix, penalties, axes)
                loss = discrete_hazard_nll(
                    z, torch.bucketize(train[-2][ix], edges, right=True), train[-1][ix]
                )
                if penalties:
                    loss = (
                        loss
                        + 0.1 * warmup_factor(epoch) * torch.stack(penalties).mean()
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        results[arm] = dict(
            **scores(model, val, edges, artifact), initial_grad_norm=probe
        )
    return dict(seed=seed, arms=results)


def symmetry_checks():
    u = torch.randn(2, 1, 7, dtype=torch.double).repeat(1, 2, 1).requires_grad_()
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[:, -1] = False
    a = u.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
    loss, _ = slot_read_overlap_loss(a, mask)
    grad = torch.autograd.grad(loss, u)[0]
    assert loss.item() > 0.999 and grad.abs().max() < 1e-12
    a = torch.eye(2, dtype=torch.double)[None].requires_grad_()
    loss, _ = slot_read_overlap_loss(a, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() == 0 and torch.autograd.grad(loss, a)[0].abs().max() == 0

    from research_models.sro_surv.duplicate_pair import (
        duplicate_pair_loss,
    )

    z = torch.zeros(1, 2, 10, dtype=torch.double, requires_grad=True)
    x = torch.tensor([-1.0] * 4 + [1.0] * 4 + [100.0, 200.0], dtype=z.dtype).reshape(
        1, 10, 1
    )
    valid = torch.arange(10)[None] < 8
    reads = valid[:, None].expand_as(z).double() / 8
    penalty = duplicate_pair_loss(z, reads, x, valid)
    grad = torch.autograd.grad(penalty, z)[0]
    assert penalty.item() == 1 and grad[:, 0, :4].min() > 0 and grad[:, 1, :4].max() < 0
    assert grad[:, :, 8:].abs().max() == 0
    assert duplicate_pair_loss(z, reads, torch.zeros_like(x), valid).item() == 0
    from research_models.sro_surv.model import SlotEncoder

    from research_models.sro_surv.risk_direction import (
        risk_directions,
    )

    torch.manual_seed(3)
    x_train = torch.randn(16, 7)
    times = torch.rand(16)
    event = torch.ones(16, dtype=torch.bool)
    original = risk_directions(x_train, times, event)
    times[0] = 5
    changed = risk_directions(x_train, times, event)
    assert torch.equal(original[0], changed[0]) and not torch.equal(original, changed)
    assert not risk_directions(x_train, times, event & False).any()
    torch.manual_seed(42)
    enc = SlotEncoder(1, 16, 8, 10).double()
    with torch.no_grad():
        enc.queries.copy_(enc.queries[0].expand_as(enc.queries).clone())
    penalties = []
    enc(x, valid, read_rounds=(1, 10), penalties=penalties, pair_split=True)
    qgrad = torch.autograd.grad(torch.stack(penalties).mean(), enc.queries)[0]
    assert (qgrad - qgrad.mean(0)).norm() > 1e-8


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuisance-audit", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--risk-only", action="store_true")
    parser.add_argument("--sro-sweep", action="store_true")
    args = parser.parse_args()
    if args.confirmation and not args.nuisance_audit:
        parser.error("confirmation uses --nuisance-audit")
    if args.sro_sweep and args.risk_only:
        parser.error("--sro-sweep and --risk-only are mutually exclusive")
    symmetry_checks()
    seeds = (
        (43, 47, 53)
        if args.confirmation
        else (19, 29, 41)
        if args.nuisance_audit
        else (3, 7, 11)
    )
    print(
        json.dumps(
            [
                run(
                    seed,
                    args.nuisance_audit,
                    args.confirmation,
                    args.risk_only,
                    args.sro_sweep,
                )
                for seed in seeds
            ],
            indent=2,
        )
    )
