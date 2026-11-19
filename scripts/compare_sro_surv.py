"""Compare pathology slot redundancy with and without SRO on one prepared fold."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.patient_pilot import (
    ROOT,
    private_path,
    sha,
    write,
)


DEFAULT_PREPARED = ROOT / "local_evidence/blca_f0_prepared"
DEFAULT_OUTPUT = ROOT / "local_evidence/sro_redundancy_compare"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_arm(name, config, prepared, output):
    config_path = output / f"{name}.json"
    log_path = output / f"{name}.log"
    arm_output = output / name
    write(config_path, config)
    print(f"Running {name} with epoch budget {config['task_epochs']}. Log at {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [
                sys.executable, "-X", "utf8", "-B",
                str(ROOT / "scripts/run_sro_surv.py"), "--run",
                "--config", str(config_path),
                "--prepared-root", str(prepared), "--output-root", str(arm_output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise SystemExit(f"{name} failed. See {log_path} and any traceback in {arm_output}")
    receipt = read_json(arm_output / "receipt.json")
    if (
        receipt["status"] != "completed"
        or not receipt["checkpoint_reload_verified"]
        or receipt["last"]["epoch"] != config["task_epochs"]
        or receipt["config"] != {**config, "configuration_SHA256": sha(config_path)}
    ):
        raise RuntimeError(f"Unexpected final checkpoint or configuration for {name}")
    return receipt


def final_diagnostics(receipt):
    last = receipt["epoch_logs"][-1]
    validation = last["validation"]
    readings = validation["read_trace"]["p"]
    pathology = validation["slots"]["p"]
    return {
        "metrics": {
            "pathology_read_BC_round_1": readings["1"]["read_BC"],
            "pathology_read_BC_round_5": readings["5"]["read_BC"],
            "pathology_read_BC_round_10": readings["10"]["read_BC"],
            "pathology_message_cosine_round_10": readings["10"]["message_cosine"],
            "pathology_slot_pair_cosine": pathology["pair_cosine_mean"],
            "pathology_zero_slot_fraction": pathology["zero_slot_fraction"],
            "pathology_across_patient_std_rms": pathology["across_patient_std_rms"],
            "validation_C_index": validation["C_index"],
            "validation_NLL": validation["NLL"],
        },
        "validation_patients": pathology["patients"],
        "pathology_trace": readings,
        "pathology_slot_summary": pathology,
        "pathology_readout_deletion_delta_NLL": validation["readout_deletion_delta_NLL"]["p"],
        "train_read_loss": last["train_read_loss"],
        "read_loss_weight": last["read_loss_weight"],
    }


def format_value(value):
    return "undefined" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5, help="epochs per arm (default 5)")
    parser.add_argument("--batch-size", type=int, help="same override for both arms")
    parser.add_argument("--device", help="same device for both arms (default cuda:0)")
    args = parser.parse_args()
    if args.epochs < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("epochs and batch size must be positive")

    prepared = args.prepared_root.resolve()
    manifest = prepared / "prepared/manifest.json"
    if not (manifest.is_file() and (prepared / "resolved_config.json").is_file()):
        parser.error(f"prepared cache is missing at {prepared}. See RUN_ON_3090.md")
    manifest_hash = sha(manifest)
    output = private_path(args.output_root)
    if output.exists():
        parser.error(f"choose a fresh --output-root. Already exists at {output}")

    sro = read_json(ROOT / "configs/sro_surv_v1.json")
    sro.update(task_epochs=args.epochs, read_diagnostics=True)
    if args.batch_size is not None:
        sro["batch_size"] = args.batch_size
    if args.device is not None:
        sro["device"] = args.device
    control = dict(sro)
    for key in ("read_loss", "read_loss_rounds", "sro_margin"):
        del control[key]

    output.mkdir(parents=True)
    control_receipt = run_arm("no_sro", control, prepared, output)
    if sha(manifest) != manifest_hash:
        raise RuntimeError("Prepared manifest changed during the control run")
    sro_receipt = run_arm("sro", sro, prepared, output)
    if sha(manifest) != manifest_hash or any(
        control_receipt[key] != sro_receipt[key]
        for key in ("split_identity", "source_SHA256", "model_parameters")
    ):
        raise RuntimeError("Prepared data or model sources changed between arms")

    baseline = final_diagnostics(control_receipt)
    treatment = final_diagnostics(sro_receipt)
    deltas = {}
    print(f"\nDevelopment validation at epoch {args.epochs}")
    print(f"{'Metric':42} {'No SRO':>11} {'SRO':>11} {'SRO - No SRO':>13}")
    for metric, before in baseline["metrics"].items():
        after = treatment["metrics"][metric]
        delta = None if before is None or after is None else after - before
        deltas[metric] = delta
        print(
            f"{metric:42} {format_value(before):>11} "
            f"{format_value(after):>11} {format_value(delta):>13}"
        )
    report = output / "redundancy_comparison.json"
    temporary = report.with_suffix(".json.tmp")
    write(temporary, {
        "scope": "BLCA DSS fold 0 development validation",
        "checkpoint_epoch": args.epochs,
        "seed": sro["seed"],
        "batch_size": sro["batch_size"],
        "patch_budget": sro["patches"],
        "device": sro["device"],
        "prepared_manifest_sha256": manifest_hash,
        "split_identity_sha256": sro_receipt["split_identity"]["identity_sha256"],
        "no_sro": baseline,
        "sro": treatment,
        "sro_minus_no_sro": deltas,
    })
    temporary.replace(report)
    print(f"\nComparison saved at {report}")


if __name__ == "__main__":
    main()
