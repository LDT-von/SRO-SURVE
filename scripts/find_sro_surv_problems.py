"""问题查找实验 1–4 and 8: paired, synthetic-only SRO diagnostics."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_models.common_private_synergy.components.losses import discrete_hazard_nll
from research_models.common_private_synergy.components.slots import masked_slot_read
from research_models.sro_surv.model import SROSurv
from slotspe_proof.slot_diversity_losses import slot_read_overlap_loss, warmup_factor


TRAIN_COUNT, HOLDOUT_COUNT = 96, 128
BATCH_SIZE = 16
PATHWAYS = 12
ROUNDS = tuple(range(1, 11))
ARMS = {"off": (), "pathology": ("p",), "omics": ("g",), "both": ("p", "g")}


@dataclass(frozen=True)
class Bags:
    patches: torch.Tensor
    pathways: torch.Tensor
    masks: dict
    modes: dict
    artifacts: dict
    latent: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor

    def with_artifact(self, amplitude):
        patches, pathways = self.patches.clone(), self.pathways.clone()
        patches[:, :, 1] += amplitude * self.artifacts["p"]
        pathways[:, :, 1] += amplitude * self.artifacts["g"]
        return replace(self, patches=patches, pathways=pathways)

    def forward(self, model, rows=slice(None), **kwargs):
        return model(
            self.patches[rows],
            list(self.pathways[rows].unbind(dim=1)),
            self.masks["p"][rows],
            self.masks["g"][rows],
            **kwargs,
        )


def make_bags(count, generator):
    latent = torch.randn(count, 2, generator=generator)

    def binary(shape, probability):
        return 2.0 * (torch.rand(shape, generator=generator) < probability).float() - 1.0

    p_shape, g_shape = (count, 40), (count, PATHWAYS)
    modes = {
        "p": binary(p_shape, latent[:, 0:1].sigmoid()),
        "g": binary(g_shape, latent[:, 1:2].sigmoid()),
    }
    artifacts = {"p": binary(p_shape, 0.5), "g": binary(g_shape, 0.5)}
    patches = 0.45 * torch.randn(*p_shape, 16, generator=generator)
    pathways = 0.45 * torch.randn(*g_shape, 4, generator=generator)
    patches[:, :, 0] += modes["p"]
    pathways[:, :, 0] += modes["g"]
    time = (-0.75 * latent.sum(-1) + 0.60 * torch.randn(count, generator=generator)).exp()
    censor = 0.30 + 3 * torch.rand(count, generator=generator)
    masks = {
        "p": torch.arange(p_shape[1])[None, :] < torch.randint(30, 41, (count, 1), generator=generator),
        "g": torch.ones(g_shape, dtype=torch.bool),
    }
    return Bags(patches, pathways, masks, modes, artifacts, latent, torch.minimum(time, censor), time <= censor)


def fit_model(train, edges, seed, modalities, epochs):
    torch.manual_seed(seed)
    model = SROSurv(16, [4] * PATHWAYS, {
        "width": 32,
        "bins": 4,
        "slots_per_modality": 8,
        "iterations": len(ROUNDS),
        "omics_dropout": 0.25,
        "read_loss": "sro" if modalities else None,
        "read_loss_rounds": ROUNDS if modalities else (),
        "read_loss_modalities": modalities,
    })
    if modalities:
        initial_penalties = []
        model.eval()
        with torch.no_grad():
            train.forward(model, slice(0, 2), penalties=initial_penalties)
        if len(initial_penalties) != len(modalities) * len(ROUNDS):
            raise RuntimeError("configured branch did not supply all requested SRO terms")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    labels = torch.bucketize(train.time, edges, right=True)
    order_rng = torch.Generator().manual_seed(seed + 99)
    for epoch in range(1, epochs + 1):
        model.train()
        for rows in torch.randperm(len(train.time), generator=order_rng).split(BATCH_SIZE):
            penalties = []
            logits, _, _ = train.forward(model, rows, penalties=penalties)
            loss = discrete_hazard_nll(logits, labels[rows], train.event[rows])
            if penalties:
                read_loss = torch.stack(penalties).sum() / len(ROUNDS)
                loss = loss + 0.10 * warmup_factor(epoch) * read_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def pair_gap(reads, labels):
    fractions = (reads * (labels[:, None, :] > 0)).sum(-1)
    i, j = torch.triu_indices(fractions.shape[1], fractions.shape[1], 1)
    return float((fractions[:, i] - fractions[:, j]).abs().mean())


@torch.no_grad()
def evaluate(model, train, holdout, edges):
    _, train_parts, _ = train.forward(model)
    trace = {}
    logits, parts, _ = holdout.forward(model, trace=trace)
    risk = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
    labels = torch.bucketize(holdout.time, edges, right=True)
    scores = {
        "C_index": float(concordance_index_censored(holdout.event.numpy(), holdout.time.numpy(), risk.numpy())[0]),
        "NLL": float(discrete_hazard_nll(logits, labels, holdout.event)),
    }
    for index, modality in enumerate(("p", "g")):
        slots, reads = parts[modality]
        probe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        probe.fit(train_parts[modality][0].flatten(1).numpy(), train.latent[:, index].numpy())
        predicted = probe.predict(slots.flatten(1).numpy())
        measures = trace[modality]
        scores[modality] = {
            "BC_by_round": {str(r): float(a["read_BC"].mean()) for r, a in measures.items()},
            "slot_cosine_round_10": float(measures[10]["slot_cosine_sum"].sum() / measures[10]["slot_cosine_pairs"].sum()),
            "known_signal_read_gap": pair_gap(reads, holdout.modes[modality]),
            "artifact_read_gap": pair_gap(reads, holdout.artifacts[modality]),
            "latent_recovery_R2": float(r2_score(holdout.latent[:, index].numpy(), predicted)),
        }
    return scores


def symmetry_experiment():
    logits = torch.zeros(1, 2, 7, dtype=torch.float64, requires_grad=True)
    mask = torch.arange(7)[None, :] < 6
    reads, _ = masked_slot_read(logits, mask)
    penalty, stats = slot_read_overlap_loss(reads, mask)
    gradient = torch.autograd.grad(penalty, logits)[0]
    return {
        "BC": stats["mean_pair_overlap"],
        "loss": stats["loss"],
        "maximum_absolute_logit_gradient": float(gradient.abs().max()),
        "scope": "forced identical reading rows under the same slot update rule on both sides",
    }


@torch.no_grad()
def readout_experiment():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = SROSurv(16, [4] * PATHWAYS, {
            "width": 32, "bins": 4, "slots_per_modality": 8,
            "iterations": len(ROUNDS), "omics_dropout": 0.25,
        }).double().eval()
        terms = {m: torch.randn(16, 8, 4, dtype=torch.float64) for m in ("p", "g")}
        deltas = {m: torch.randn_like(t[:, 0]) for m, t in terms.items()}
    shifted = {m: t.clone() for m, t in terms.items()}
    for modality, contributions in shifted.items():
        contributions[:, 0] += deltas[modality]
        contributions[:, 1] -= deltas[modality]
    return {
        "maximum_absolute_logit_difference": float((model.predict(shifted) - model.predict(terms)).abs().max()),
        "scope": "direct forward check under sum-preserving redistribution; no shared/private content labels",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 29, 41])
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs must be positive and model seeds must be distinct")
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(611194)
    train = make_bags(TRAIN_COUNT, generator)
    holdout = make_bags(HOLDOUT_COUNT, generator)
    edges = torch.quantile(train.time[train.event], torch.tensor([0.25, 0.5, 0.75]))
    results = {}
    for condition, strength in (("clean", 0.0), ("artifact", 3.0)):
        train_input, holdout_input = train.with_artifact(strength), holdout.with_artifact(strength)
        results[condition] = []
        for seed in args.seeds:
            results[condition].append({
                "seed": seed,
                "arms": {
                    name: evaluate(fit_model(train_input, edges, seed, modes, args.epochs), train_input, holdout_input, edges)
                    for name, modes in ARMS.items()
                },
            })
    print(json.dumps({
        "scope": "Synthetic only; paired arms on one fixed train/holdout draw; reduced-width model.",
        "protocol": {
            "train": TRAIN_COUNT, "holdout": HOLDOUT_COUNT, "epochs": args.epochs,
            "data_seed": 611194, "model_seeds": args.seeds, "batch_size": BATCH_SIZE,
            "width": 32, "read_rounds": ROUNDS, "modality_arms": ARMS,
            "artifact_amplitude": {"clean": 0, "artifact": 3},
            "SRO_aggregation": "per-side round means summed; each side ramps to weight 0.10",
        },
        "experiments": {
            "问题查找实验 1": "Read pathology BC and slot cosine.",
            "问题查找实验 2": "Read omics BC and slot cosine.",
            "问题查找实验 3": "Compare the known-signal and artifact read gaps and held-out latent recovery.",
            "问题查找实验 4": symmetry_experiment(),
            "问题查找实验 8": readout_experiment(),
        },
        "results": results,
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
