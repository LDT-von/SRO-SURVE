"""问题查找实验 9: shared and private factor recovery from frozen slots."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from find_sro_surv_problems import Bags, PATHWAYS, fit_model


def make_role_bags(count, seed):
    rng = torch.Generator().manual_seed(seed)
    # Independent factors: shared, pathology-private, and omics-private.
    latent = torch.randn(count, 3, generator=rng)
    patches = 0.55 * torch.randn(count, 40, 16, generator=rng)
    pathways = 0.55 * torch.randn(count, PATHWAYS, 4, generator=rng)
    for inputs, private_index in ((patches, 1), (pathways, 2)):
        half = inputs.shape[1] // 2
        inputs[:, :half, 0] += latent[:, :1]
        inputs[:, half:, 0] += latent[:, private_index:private_index + 1]
        inputs[:, :half, 1] += 1.0
        inputs[:, half:, 1] -= 1.0
    time = (-0.70 * latent.sum(1) + 0.60 * torch.randn(count, generator=rng)).exp()
    censor = 0.30 + 3 * torch.rand(count, generator=rng)
    masks = {
        "p": torch.ones(patches.shape[:2], dtype=torch.bool),
        "g": torch.ones(pathways.shape[:2], dtype=torch.bool),
    }
    return Bags(
        patches, pathways, masks, {}, {}, latent, torch.minimum(time, censor), time <= censor
    )


@torch.no_grad()
def evaluate(model, fit, selection, holdout):
    fit_parts = fit.forward(model)[1]
    selection_parts = selection.forward(model)[1]
    holdout_parts = holdout.forward(model)[1]
    result = {}
    for side, private_index in (("p", 1), ("g", 2)):
        slots = [parts[side][0].numpy() for parts in (fit_parts, selection_parts, holdout_parts)]

        def scores(features):
            probe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            probe.fit(features[0], fit.latent.numpy())
            return [
                r2_score(data.latent.numpy(), probe.predict(x), multioutput="raw_values")
                for data, x in zip((selection, holdout), features[1:])
            ]

        per_slot = np.asarray([
            scores([x[:, index] for x in slots]) for index in range(slots[0].shape[1])
        ])
        chosen = per_slot[:, 0, :].argmax(0)[[0, private_index]]
        result[side] = {
            "all_slots_R2": scores([x.reshape(len(x), -1) for x in slots])[1].tolist(),
            "selected_slot_indices_zero_based": chosen.tolist(),
            "selected_slot_R2_rows_shared_and_own_private": per_slot[chosen, 1, :].tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 29, 41])
    args = parser.parse_args()
    if args.epochs < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("epochs must be positive and model seeds must be distinct")
    if args.output is not None and args.output.exists():
        parser.error("choose a fresh output file")
    torch.set_num_threads(1)
    train, fit, selection, holdout = (
        make_role_bags(count, seed)
        for count, seed in ((128, 611201), (128, 611202), (96, 611203), (192, 611204))
    )
    edges = torch.quantile(train.time[train.event], torch.tensor([0.25, 0.5, 0.75]))
    runs = []
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            for arm, modes in (("off", ()), ("both", ("p", "g"))):
                model = fit_model(train, edges, seed, modes, args.epochs)
                runs.append({
                    "seed": seed,
                    "arm": arm,
                    **evaluate(model, fit, selection, holdout),
                })
    report = {
        "experiment": "问题查找实验 9",
        "scope": "Synthetic factor roles only; model fitted with survival outcomes, slots frozen for probes.",
        "factor_columns": ["shared", "pathology_private", "omics_private"],
        "samples_by_split": {"model_train": 128, "probe_fit": 128, "slot_selection": 96, "holdout": 192},
        "epochs": args.epochs,
        "model_seeds": args.seeds,
        "results": runs,
    }
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
