"""Synthetic problem-finding experiments 5–7 for SRO-Surv."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_models.common_private_synergy.components.losses import discrete_hazard_nll
from research_models.common_private_synergy.components.slots import READ_EPS
from research_models.common_private_synergy.patient_pilot_engine import (
    fit_boundaries,
    label_times,
)
from research_models.sro_surv.model import SROSurv
from slotspe_proof.slot_diversity_losses import slot_read_overlap_loss, warmup_factor


def sample(count, support, seed):
    """Factors A, B and R affect outcome equally and occupy disjoint patch sets."""
    rng = torch.Generator().manual_seed(seed)
    factors = torch.randn(count, 4, generator=rng)
    groups = torch.repeat_interleave(torch.arange(3), torch.tensor(support))
    patches = 0.60 * torch.randn(count, len(groups), 12, generator=rng)
    group_markers = F.one_hot(groups, 3).to(patches)
    patches[:, :, :3] += 1.5 * group_markers
    patches[:, :, 6] += 2.25 * (2 * (torch.arange(len(groups)) % 2) - 1)
    omic_nuisance = torch.randn(count, 1, generator=rng)
    omics = [
        (factors[:, 3:] if pathway < 4 else omic_nuisance)
        + 0.6 * torch.randn(count, 4, generator=rng)
        for pathway in range(8)
    ]
    log_time = -0.8 * factors[:, :3].sum(dim=1) - 0.5 * factors[:, 3]
    time = (log_time + 0.6 * torch.randn(count, generator=rng)).exp()
    censor = 0.3 + 3 * torch.rand(count, generator=rng)
    return patches, groups, omics, factors, torch.minimum(time, censor), time <= censor


def inputs(data, override=None):
    background, groups, omics, factors, _, _ = data
    pathology = factors[:, :3].clone()
    if override is not None:
        factor, value = override
        pathology[:, factor] = value
    patch = background.clone()
    group_markers = F.one_hot(groups, 3).to(patch)
    patch[:, :, 3:6] += 0.55 * pathology[:, None, :] * group_markers[None, :, :]
    p_mask = torch.ones(patch.shape[:2], dtype=torch.bool)
    g_mask = torch.ones((len(patch), len(omics)), dtype=torch.bool)
    return patch, omics, p_mask, g_mask


def forward(model, data, rows=slice(None), penalties=None):
    patch, omic, p_mask, g_mask = data
    return model(
        patch[rows], [pathway[rows] for pathway in omic],
        p_mask[rows], g_mask[rows], penalties=penalties,
    )


def train(data, seed, dual):
    torch.manual_seed(seed)
    config = dict(width=32, bins=4, slots_per_modality=8, iterations=10, omics_dropout=0.25)
    if dual:
        config.update(
            read_loss="sro", read_loss_modalities=["p", "g"],
            read_loss_rounds=list(range(1, 11)), sro_margin=0.90,
        )
    model = SROSurv(12, [4] * 8, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    time, event = data[-2:]
    edges = fit_boundaries(time, event, 4)
    completed = label_times(time, event, edges)
    order = torch.Generator().manual_seed(seed + 99)
    features = inputs(data)
    for epoch in range(1, 13):
        for rows in torch.randperm(len(time), generator=order).split(16):
            penalties = []
            logits, _, _ = forward(model, features, rows, penalties)
            expected_penalties = 20 if dual else 0
            if len(penalties) != expected_penalties:
                raise RuntimeError("did not apply the planned SRO to both sides")
            loss = discrete_hazard_nll(logits, completed[rows], event[rows])
            if penalties:
                loss = loss + 0.10 * warmup_factor(epoch) * torch.stack(penalties).sum() / 10
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), edges


def binding_stats(assignments, groups):
    """Count only confident token winners, so ties cannot create fragmentation."""
    values, indices = assignments.topk(2, dim=1)
    confident = (values[:, 0] >= 0.30) & (values[:, 0] - values[:, 1] >= 0.10)
    winning_slots = F.one_hot(indices[:, 0], assignments.shape[1]).to(assignments)
    owned = winning_slots * confident[:, :, None]
    labels = F.one_hot(groups, 3).to(assignments)
    counts = owned.transpose(1, 2) @ labels
    coverage = counts / labels.sum(dim=0)
    purity = counts / counts.sum(dim=2, keepdim=True).clamp_min(1)
    subparts = (coverage >= 0.20) & (purity >= 0.70)
    fragmented = (subparts.sum(dim=1) >= 2) & ((coverage * subparts).sum(dim=1) >= 0.65)
    mixed = ((purity >= 0.25).sum(dim=2) >= 2) & (counts.sum(dim=2) >= 2)
    return {
        "fragmentation_rate_by_factor": fragmented.float().mean(dim=0).tolist(),
        "mixed_slot_fraction": float(mixed.float().mean()),
        "assigned_coverage_by_factor": coverage.sum(dim=1).mean(dim=0).tolist(),
        "best_slot_coverage_by_factor": coverage.amax(dim=1).mean(dim=0).tolist(),
    }


def check_binding_controls():
    groups = torch.arange(3).repeat_interleave(4)
    clean = torch.arange(3).repeat_interleave(4)
    split = torch.tensor([0, 0, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2])
    mixed = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2])

    def measure(winners):
        return binding_stats(F.one_hot(winners, 4).T[None].float(), groups)

    assert measure(clean)["fragmentation_rate_by_factor"] == [0.0, 0.0, 0.0]
    assert measure(split)["fragmentation_rate_by_factor"] == [1.0, 0.0, 0.0]
    assert measure(mixed)["fragmentation_rate_by_factor"] == [0.0, 0.0, 0.0]
    assert measure(mixed)["mixed_slot_fraction"] == 0.25
    assert binding_stats(torch.full((1, 4, 12), 0.25), groups)["assigned_coverage_by_factor"] == [0.0] * 3


def probe(fit_x, choose_x, heldout_x, targets):
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    reg.fit(fit_x, targets[0])
    return np.stack([
        r2_score(y, reg.predict(x), multioutput="raw_values")
        for x, y in [(choose_x, targets[1]), (heldout_x, targets[2])]
    ])


def risk(logits):
    return -(1 - logits.sigmoid()).cumprod(dim=-1).sum(dim=-1)


@torch.no_grad()
def evaluate(model, edges, fit, choose, heldout):
    fit_slots = forward(model, inputs(fit))[1]["p"][0]
    choose_slots = forward(model, inputs(choose))[1]["p"][0]
    heldout_inputs = inputs(heldout)
    fractions = []
    with model.encoders["p"].register_forward_hook(
        lambda _module, _inputs, outputs: fractions.append(outputs[2])
    ):
        logits, parts, _ = forward(model, heldout_inputs)
    slots, attention = parts["p"]
    count = heldout_inputs[2].sum(dim=1, keepdim=True)
    masses = fractions[0] * count
    assignments = (attention * (masses + READ_EPS * count)[:, :, None] - READ_EPS).clamp_min(0)
    torch.testing.assert_close(assignments.sum(dim=1), torch.ones_like(attention[:, 0]))
    groups = heldout[1]
    group_markers = F.one_hot(groups, 3).to(attention)
    peak_read = (attention @ group_markers).amax(dim=1).mean(dim=0)

    targets = [data[3][:, :3] for data in (fit, choose, heldout)]
    all_slots_r2 = probe(fit_slots.flatten(1), choose_slots.flatten(1), slots.flatten(1), targets)[1]
    single_slot_r2 = np.stack([
        probe(fit_slots[:, slot], choose_slots[:, slot], slots[:, slot], targets)
        for slot in range(slots.shape[1])
    ])
    selected = single_slot_r2[:, 0].argmax(axis=0)
    oracle = [
        torch.stack([inputs(data)[0][:, groups == index, index + 3].mean(dim=1)
                     for index in range(3)], dim=1)
        for data in (fit, choose, heldout)
    ]
    oracle_r2 = probe(*oracle, targets)[1]
    selection_r2 = single_slot_r2[selected, 1]
    full_effect, selected_slot_effect = [], []
    for factor in range(3):
        low_logits, _, low_terms = forward(model, inputs(heldout, (factor, -1.0)))
        high_logits, _, high_terms = forward(model, inputs(heldout, (factor, +1.0)))
        torch.testing.assert_close(low_terms["g"], high_terms["g"], rtol=0, atol=0)
        low_risk = risk(low_logits)
        full_effect.append(float((risk(high_logits) - low_risk).mean()))
        swap_effect = []
        for slot in selected:
            replacement = low_terms["p"].clone()
            replacement[:, slot] = high_terms["p"][:, slot]
            changed = model.predict({**low_terms, "p": replacement})
            swap_effect.append(float((risk(changed) - low_risk).mean()))
        selected_slot_effect.append(swap_effect)

    time, event = heldout[-2:]
    _, overlap = slot_read_overlap_loss(attention, heldout_inputs[2])
    return {
        "read_BC": overlap["mean_pair_overlap"],
        "C_index": concordance_index_censored(event.numpy(), time.numpy(), risk(logits).numpy())[0],
        "NLL": float(discrete_hazard_nll(logits, label_times(time, event, edges), event)),
        "experiment_5": binding_stats(assignments, groups),
        "experiment_6": {
            "peak_read_mass_by_factor": peak_read.tolist(),
            "all_slots_R2": all_slots_r2.tolist(),
            "oracle_true_group_mean_R2": oracle_r2.tolist(),
        },
        "experiment_7": {
            "chosen_slots_A_B_R_zero_based": selected.tolist(),
            "chosen_slot_R2_rows_chosen_slots_columns_factors": selection_r2.tolist(),
            "full_input_risk_change_by_factor": full_effect,
            "single_contribution_risk_change_rows_perturbed_factor_columns_chosen_slots": selected_slot_effect,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=(19, 29, 41))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_binding_controls()
    results = []
    with threadpool_limits(limits=1):
        for support in ((16, 16, 16), (23, 23, 2)):
            train_data, fit, choose, heldout = [
                sample(n, support, seed)
                for n, seed in [(144, 246500), (128, 246501), (64, 246502), (192, 246503)]
            ]
            for seed in args.seed:
                for dual in (False, True):
                    model, edges = train(train_data, seed, dual)
                    results.append({
                        "support_A_B_R": support, "seed": seed,
                        "arm": "dual_sro" if dual else "off",
                        **evaluate(model, edges, fit, choose, heldout),
                    })
    record = {
        "scope": "synthetic only, width 32, 12 epochs; same synthetic draws across model seeds",
        "binding_positive_and_negative_controls": "passed",
        "results": results,
    }
    output = json.dumps(record, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(output + "\n")


if __name__ == "__main__":
    main()
