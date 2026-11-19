"""Single-stage slot survival training and validation."""

import math
from pathlib import Path
import torch
from ..common_private_synergy.components.losses import discrete_hazard_nll
from ..common_private_synergy.full_slot_diagnostics import slot_summary
from ..common_private_synergy.patient_pilot import ROOT, sha
from ..common_private_synergy.patient_pilot_engine import (
    PreparedStore,
    fit_boundaries,
    label_times,
)
from ..common_private_synergy.scaled.adapter import fit_pathway_moments
from .model import SROSurv
from .risk_direction import risk_directions
from slotspe_proof.slot_diversity_losses import warmup_factor
from ..common_private_synergy.scaled.engine import (
    _resume_state,
    _save_checkpoint,
    _write_receipt,
)


def validate_protocol(c):
    if c["schema"] not in ("sro_surv_v1", "scaled_slot16_v2"):
        raise ValueError("unknown SRO-Surv configuration schema")
    if (c.get("read_loss"), tuple(c.get("read_loss_rounds", ()))) not in (
        (None, ()),
        ("sro", (10,)),
        ("sro", (1, 10)),
        ("sro", tuple(range(1, 11))),
        ("duplicate_pair", (1, 10)),
        ("risk_pair", (1, 10)),
    ):
        raise ValueError("unknown read loss or iteration schedule")
    if "sro_margin" in c:
        margin = c["sro_margin"]
        if (
            c.get("read_loss") != "sro"
            or type(margin) not in (int, float)
            or not 0 <= margin < 1
        ):
            raise ValueError("SRO margin must be a fixed number in [0, 1)")
    if any(
        key in c
        for key in (
            "role_weight",
            "task_mode",
            "role_epochs",
            "common_rank",
            "allocation",
        )
    ):
        raise ValueError("role settings cannot be used with ungrouped slots")
    if (
        c["slots_per_modality"],
        c["iterations"],
        c["width"],
        c["bins"],
    ) != (8, 10, 256, 4):
        raise ValueError(
            "expected plain 8+8 slots, 10 iterations, width 256 and four hazards"
        )


class InputAdapter:
    def patches_input(self, x):
        return (x - self.mean) / self.scale

    def omics_input(self, xs):
        return [(x - m) / s for x, m, s in zip(xs, self.pathway_mean, self.pathway_sd)]


def scaled_batch(store, ids, adapter, boundaries, c, device, *, generator):
    """The archived batch, with the scaled omics encoder input.

    The sampling rule, the without-replacement draw, the zero-padding rule and
    the mask are the archived ones: every real patch is cached, a fresh subset
    is drawn without replacement on each access, and bags shorter than the
    budget are zero-padded rather than excluded.
    """
    ps, pm, times, events = [], [], [], []
    per_patient_omics = []
    for pid in ids:
        patches, omics = store.load(pid)
        count = min(len(patches), c["patches"])
        indices = (
            torch.randperm(len(patches), generator=generator)[:count].sort().values
        )
        p = adapter.patches_input(patches[indices])
        g = adapter.omics_input(omics)
        padded = torch.zeros(c["patches"], p.shape[1], dtype=p.dtype)
        padded[:count] = p
        ps.append(padded)
        per_patient_omics.append(g)
        pm.append(torch.arange(c["patches"]) < count)
        row = store.manifest["patients"][pid]
        times.append(row["time"])
        events.append(row["censor"] < 1)
    # The pathways own DIFFERENT gene counts, so the omics batch cannot be one
    # rectangular tensor.  It is a LIST of per-pathway [patients, genes_i]
    # tensors, which is exactly what the 329 independent pathway MLPs consume.
    gs = [
        torch.stack([patient_omics[i] for patient_omics in per_patient_omics])
        for i in range(len(adapter.pathway_mean))
    ]
    times = torch.tensor(times, dtype=torch.float64)
    events = torch.tensor(events, dtype=torch.bool)
    result = {
        "p": torch.stack(ps),
        "g": gs,
        "p_mask": torch.stack(pm),
        "g_mask": torch.ones(len(ids), len(adapter.pathway_mean), dtype=torch.bool),
        "time": times,
        "event": events,
        "completed": label_times(times, events, boundaries),
    }
    return {
        key: [block.to(device) for block in value] if key == "g" else value.to(device)
        for key, value in result.items()
    }


@torch.no_grad()
def evaluate(model, store, ids, adapter, boundaries, c, device, seed):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    losses, logits, events, times = [], [], [], []
    slots = {m: [] for m in ("p", "g")}
    attention = {m: [] for m in slots}
    deletions = {m: [] for m in slots}
    trace_totals = {}
    for i in range(0, len(ids), c["batch_size"]):
        b = scaled_batch(
            store,
            ids[i : i + c["batch_size"]],
            adapter,
            boundaries,
            c,
            device,
            generator=rng,
        )
        trace = {} if c.get("read_diagnostics", False) else None
        z, parts, terms = model(b["p"], b["g"], b["p_mask"], b["g_mask"], trace=trace)
        for modality, rounds in (trace or {}).items():
            for t, measures in rounds.items():
                totals = trace_totals.setdefault(modality, {}).setdefault(t, {})
                for key, value in measures.items():
                    totals[key] = totals.get(key, 0.0) + value.sum().item()

        def nll(pred):
            return discrete_hazard_nll(
                pred, b["completed"], b["event"], reduction="none"
            )

        loss = nll(z)
        losses.append(loss.cpu())
        logits.append(z.cpu())
        events.append(b["event"].cpu())
        times.append(b["time"].cpu())
        for m, (v, a) in parts.items():
            slots[m].append(v.cpu())
            attention[m].append(a.cpu())
            removed = []
            for j in range(terms[m].shape[1]):
                retained = terms[m].clone()
                retained[:, j] = 0
                removed.append(nll(model.predict({**terms, m: retained})) - loss)
            deletions[m].append(torch.stack(removed, 1).cpu())
    from sksurv.metrics import concordance_index_censored

    risk = -(1 - torch.cat(logits).sigmoid()).cumprod(-1).sum(-1)
    for rounds in trace_totals.values():
        for values in rounds.values():
            values["read_BC"] /= len(ids)
            values["mean_real_tokens"] = values.pop("real_tokens") / len(ids)
            for name in ("message", "slot"):
                total = values.pop(name + "_cosine_sum")
                count = values[name + "_cosine_pairs"]
                values[name + "_cosine"] = total / count if count else None
    return dict(
        read_trace=trace_totals,
        NLL=torch.cat(losses).mean().item(),
        C_index=float(
            concordance_index_censored(
                torch.cat(events).numpy(), torch.cat(times).numpy(), risk.numpy()
            )[0]
        ),
        slots={
            m: slot_summary(torch.cat(slots[m]), torch.cat(attention[m])) for m in slots
        },
        readout_deletion_delta_NLL={
            m: torch.cat(v).mean(0).tolist() for m, v in deletions.items()
        },
    )


def run_sro_surv(
    root,
    out,
    manifest,
    c,
    source_hashes,
    *,
    split_guard=None,
    synthetic=False,
    resume=False,
):
    """Train both ungrouped slot encoders directly from the survival objective.

    With `resume=True` an existing output directory is continued from its last
    checkpoint.  With `resume=False` (the default) a fresh output directory is
    required, exactly as in the archived arms.
    """
    if not synthetic and split_guard is None:
        raise ValueError("verified patient split admission required")
    root, out = Path(root), Path(out)
    if out.exists() and not resume:
        raise FileExistsError(
            "fresh run required; pass resume=True to continue an interrupted run"
        )
    if source_hashes != {name: sha(ROOT / name) for name in source_hashes}:
        raise RuntimeError("source changed before execution")
    identity = split_guard() if split_guard else {"schema": "invented_smoke_only"}
    manifest_hash = sha(root / "prepared/manifest.json")
    train, val = (manifest["folds"][str(c["fold"])][key] for key in ("train", "val"))
    if (
        not train
        or not val
        or len(set(train)) != len(train)
        or len(set(val)) != len(val)
        or set(train) & set(val)
    ):
        raise ValueError("nonempty unique disjoint train and validation required")
    torch.set_num_threads(c["torch_threads"])
    torch.manual_seed(c["seed"])
    device = torch.device(c["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if not synthetic:
        validate_protocol(c)
        if len(manifest["omic_sizes"]) != 329:
            raise ValueError("expected 329 pathways")
    store = PreparedStore(root, manifest)
    means = torch.stack([store.load(i)[0].double().mean(0) for i in train])
    adapter = InputAdapter()
    adapter.mean, adapter.scale = (
        means.mean(0),
        means.std(0, unbiased=False).clamp_min(1e-4),
    )
    sizes = manifest["omic_sizes"]
    adapter.pathway_mean, adapter.pathway_sd = fit_pathway_moments(store, train, sizes)
    times = torch.tensor(
        [manifest["patients"][i]["time"] for i in train], dtype=torch.float64
    )
    events = torch.tensor(
        [manifest["patients"][i]["censor"] < 1 for i in train], dtype=torch.bool
    )
    boundaries = fit_boundaries(times, events, c["bins"])
    model = SROSurv(len(adapter.mean), sizes, c).double().to(device)
    axes = None
    if model.risk_pair:
        axes = risk_directions(adapter.patches_input(means), times, events).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=c["task_lr"])
    rng = torch.Generator().manual_seed(c["seed"] + 9000)
    saved = (
        _resume_state(out, c, source_hashes, manifest_hash, identity)
        if resume
        else None
    )
    if resume and saved is None:
        raise ValueError("missing plain-slot checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    logs, best = [], None
    if saved:
        model.load_state_dict(saved["state_dict"])
        optim.load_state_dict(saved["optimizer_state"])
        rng.set_state(saved["train_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        logs, best = saved["logs"], saved["best"]
    receipt = dict(
        config=c,
        source_SHA256=source_hashes,
        epoch_logs=logs,
        best=best,
        last=logs[-1] if logs else None,
        model_parameters=sum(p.numel() for p in model.parameters()),
        omics_encoder_parameters=model.omics_encoder.parameter_count(),
        slot_budget={"p": 8, "g": 8},
        scope="development validation",
        status="running",
        split_identity=identity,
    )
    for epoch in range(len(logs) + 1, c["task_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train), generator=rng).tolist()
        aux_total = 0.0
        survival_total = 0.0
        weight = 0.10 * warmup_factor(epoch) if model.read_rounds else 0.0
        for i in range(0, len(order), c["batch_size"]):
            indices = order[i : i + c["batch_size"]]
            ids = [train[j] for j in indices]
            b = scaled_batch(store, ids, adapter, boundaries, c, device, generator=rng)
            penalties = [] if model.read_rounds else None
            z, _, _ = model(
                b["p"],
                b["g"],
                b["p_mask"],
                b["g_mask"],
                penalties=penalties,
                axes=axes[indices] if axes is not None else None,
            )
            loss = discrete_hazard_nll(z, b["completed"], b["event"])
            survival_total += loss.detach().item() * len(ids)
            if penalties:
                auxiliary = torch.stack(penalties).mean()
                aux_total += auxiliary.detach().item() * len(ids)
                loss = loss + weight * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite survival loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if sha(root / "prepared/manifest.json") != manifest_hash or (
            split_guard and split_guard() != identity
        ):
            raise RuntimeError("prepared split changed")
        metrics = evaluate(
            model, store, val, adapter, boundaries, c, device, c["seed"] + 19000
        )
        row = dict(epoch=epoch, C_index=metrics["C_index"], NLL=metrics["NLL"])
        logs.append(
            dict(
                **row,
                validation=metrics,
                train_NLL=survival_total / len(train),
                train_read_loss=aux_total / len(train),
                read_loss_weight=weight,
            )
        )
        improved = best is None or (row["C_index"], -row["NLL"]) > (
            best["C_index"],
            -best["NLL"],
        )
        if improved:
            best = row
        state = dict(
            config=c,
            state_dict=model.state_dict(),
            optimizer_state=optim.state_dict(),
            adapter=vars(adapter),
            sizes=sizes,
            boundaries=boundaries,
            source_SHA256=source_hashes,
            split_identity=identity,
            manifest_SHA256=manifest_hash,
            logs=logs,
            best=best,
            train_generator_state=rng.get_state(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
        )
        if improved:
            _save_checkpoint(out / "task_best.pt", state)
        _save_checkpoint(out / "task_last.pt", state)
        receipt.update(best=best, last=row)
        _write_receipt(out / "receipt.json", receipt)
    for label in ("best", "last"):
        saved = torch.load(
            out / ("task_" + label + ".pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(saved["state_dict"])
        checked = evaluate(
            model, store, val, adapter, boundaries, c, device, c["seed"] + 19000
        )
        if any(
            not math.isclose(checked[k], receipt[label][k], abs_tol=1e-12, rel_tol=0)
            for k in ("NLL", "C_index")
        ):
            raise RuntimeError("checkpoint replay mismatch")
    receipt.update(status="completed", checkpoint_reload_verified=True)
    _write_receipt(out / "receipt.json", receipt)
    return receipt
