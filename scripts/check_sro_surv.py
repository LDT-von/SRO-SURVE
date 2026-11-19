"""Check the plain slot model using invented inputs only."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from check_scaled16_synthetic import ROOT, invent_prepared
from research_models.sro_surv.engine import (
    run_sro_surv,
    validate_protocol,
)
from research_models.sro_surv.model import SROSurv
from research_models.common_private_synergy.components.losses import discrete_hazard_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sro_surv_v1.json",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for the synthetic training and replay"
    )
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    validate_protocol(c)
    torch.manual_seed(3)
    torch.set_num_threads(2)
    c.update(width=16, device=args.device, patches=16, torch_threads=2, task_epochs=1)
    model = SROSurv(6, [4, 5], c)
    p = torch.randn(4, 12, 6)
    g = [torch.randn(4, 4), torch.randn(4, 5)]
    z, parts, _ = model(
        p, g, torch.ones(4, 12, dtype=torch.bool), torch.ones(4, 2, dtype=torch.bool)
    )
    discrete_hazard_nll(z, torch.arange(4), torch.ones(4, dtype=torch.bool)).backward()
    for m in ("p", "g"):
        assert parts[m][0].shape == (4, 8, 16)
        assert model.encoders[m].queries.grad.norm(dim=-1).gt(0).all()
        assert model.weights[m].grad.abs().gt(0).all()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = invent_prepared(root, count=32, features=6)
        out = root / "run"
        one = run_sro_surv(root, out, manifest, c, {}, synthetic=True)
        assert one["checkpoint_reload_verified"]
        log = one["epoch_logs"][0]
        assert set(log["validation"]["read_trace"]["p"]) == {1, 5, 10}
        if c.get("read_loss"):
            assert log["read_loss_weight"] > 0 and log["train_read_loss"] > 0
        else:
            assert log["read_loss_weight"] == 0 and log["train_read_loss"] == 0
        c["task_epochs"] = 2
        two = run_sro_surv(root, out, manifest, c, {}, synthetic=True, resume=True)
        assert len(two["epoch_logs"]) == 2 and two["checkpoint_reload_verified"]
    print(
        "PASS: both query banks, 16 weights, survival NLL, validation, resume and checkpoint replay"
    )


if __name__ == "__main__":
    main()
