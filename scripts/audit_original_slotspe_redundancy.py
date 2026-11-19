"""Read-only structural probes of original SlotSPE on invented tensors."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slotspe_proof.model import build_model, make_args


def audit():
    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = (
        build_model(
            ROOT / "third_party/SlotSPE",
            make_args([4, 5, 3], encoding_dim=16),
            "original",
        )
        .double()
        .eval()
    )
    x = {
        "x_wsi": torch.randn(2, 32, 16, dtype=torch.double),
        **{
            f"x_omic{i}": torch.randn(2, d, dtype=torch.double)
            for i, d in enumerate((4, 5, 3), 1)
        },
        "omic_missing": False,
    }
    decoders = (model.slot_decoder_wsi, model.slot_decoder_omic)
    captures = []
    hooks = [
        module.register_forward_hook(lambda _m, _x, out: captures.append(out.detach()))
        for module in (model.slot_attention_wsi, model.slot_attention_omic)
    ]
    try:
        with torch.no_grad():
            rng = torch.get_rng_state()
            model(**x)
            for decoder in decoders:
                decoder.k = 8
            torch.set_rng_state(rng)
            model(**x)
    finally:
        for h in hooks:
            h.remove()
        for decoder in decoders:
            decoder.k = 2
    assert len(captures) == 4 and all(
        torch.equal(a, b) for a, b in zip(captures[:2], captures[2:])
    )

    # Fixed mask, change only a slot that Top-K did not select.
    p, g = captures[:2]
    keep = torch.arange(p.shape[1])[None].expand(p.shape[0], -1) < 2
    p_changed = p.clone()
    p_changed[:, 2] += torch.randn_like(p_changed[:, 2])
    with torch.no_grad():
        patches = model.wsi_mlp(x["x_wsi"])
        recon_delta = (
            (
                model.reconstruction_head_wsi(patches, p, mask=keep)
                - model.reconstruction_head_wsi(patches, p_changed, mask=keep)
            )
            .abs()
            .max()
            .item()
        )
        self_delta = (
            (
                model.self_attention_wsi(p, mask=keep).mean(1)
                - model.self_attention_wsi(p_changed, mask=keep).mean(1)
            )
            .abs()
            .max()
            .item()
        )
        cross_delta = (
            (
                model.cross_attention(p, g).mean(1)
                - model.cross_attention(p_changed, g).mean(1)
            )
            .abs()
            .max()
            .item()
        )
    assert recon_delta == 0.0 and self_delta > 0.0 and cross_delta > 0.0

    z, _ = model(**x)
    params = [v for decoder in decoders for v in decoder.pred_keep_slot.parameters()]
    loss = model.loss_fn(z, y=torch.tensor([0, 1]), c=torch.tensor([0.0, 1.0]), t=None)
    disconnected = sum(
        g is None for g in torch.autograd.grad(loss, params, allow_unused=True)
    )
    assert disconnected == len(params)
    return {
        "scope": "invented tensors, original untrained weights, executed on CPU",
        "candidate_slots_equal_when_changing_top2_to_top8_with_fixed_state": True,
        "main_loss_disconnected_gate_tensors": [disconnected, len(params)],
        "unselected_pathology_slot_max_abs_change": {
            "masked_pathology_reconstruction": recon_delta,
            "pooled_intra_modal_features": self_delta,
            "pooled_cross_modal_features": cross_delta,
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
