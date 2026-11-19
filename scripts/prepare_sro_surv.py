"""Check or prepare the BLCA DSS fold-0 inputs from separate data paths."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.patient_pilot import (
    ROOT,
    private_path,
    sha,
    write,
)
from research_models.common_private_synergy.pilot_split import prepare
from slotspe_proof.patient import _config


SPLIT = Path("5fold_uni2h/blca/fold_0.csv")
CLINICAL = Path("clinical/all/blca.csv")
RNA = Path("raw_rna_data_inter/blca_rna_inter.csv")
SIGNATURE = Path("signatures/combine_signatures.csv")
FEATURE_SUFFIXES = {".pt", ".h5"}
EXPECTED_PATHWAYS = 329


def csv_header(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream), [])


def check_inputs(inputs, features):
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required CSV is absent: {path}")
    clinical_fields = {"case id", "survival_months_dss", "censorship_dss", "wsi"}
    missing = clinical_fields - set(csv_header(inputs[CLINICAL]))
    if missing:
        raise ValueError(f"Missing clinical columns {sorted(missing)}: {inputs[CLINICAL]}")
    split = inputs[Path("splits") / SPLIT]
    if not {"train", "val"}.issubset(csv_header(split)):
        raise ValueError(f"Split CSV must have train and val columns: {split}")
    if len(csv_header(inputs[RNA])) < 2:
        raise ValueError(f"Expected a gene-by-patient CSV: {inputs[RNA]}")
    if len(csv_header(inputs[SIGNATURE])) < EXPECTED_PATHWAYS:
        raise ValueError(f"Expected at least {EXPECTED_PATHWAYS} signature columns: {inputs[SIGNATURE]}")
    if not features.is_dir():
        raise FileNotFoundError(f"UNI2-h feature directory is absent: {features}")
    if not any(path.suffix in FEATURE_SUFFIXES and path.is_file() for path in features.iterdir()):
        raise ValueError(f"Pass the directory containing per-slide .h5 or .pt files: {features}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-csv", type=Path, required=True)
    parser.add_argument("--rna-csv", type=Path, required=True)
    parser.add_argument("--signature-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True, help=f"path ending in {SPLIT.as_posix()}")
    parser.add_argument("--features-root", type=Path, required=True, help="directory containing BLCA UNI2-h slide feature files")
    parser.add_argument("--prepare", action="store_true", help="write the patient cache; default checks only inputs")
    parser.add_argument("--output-root", type=Path, help="fresh child of local_evidence/, required with --prepare")
    args = parser.parse_args()
    if args.prepare != (args.output_root is not None):
        parser.error("--prepare and --output-root are required together")
    if tuple(part.lower() for part in args.split_csv.parts[-3:]) != SPLIT.parts:
        parser.error(f"--split-csv must point to {SPLIT.as_posix()}")

    inputs = {
        CLINICAL: args.clinical_csv.resolve(),
        RNA: args.rna_csv.resolve(),
        SIGNATURE: args.signature_csv.resolve(),
        Path("splits") / SPLIT: args.split_csv.resolve(),
    }
    features = args.features_root.resolve()
    check_inputs(inputs, features)
    if not args.prepare:
        print(json.dumps({"status": "input_layout_checked", "features_root": str(features)}, indent=2))
        return

    out = private_path(args.output_root)
    out.mkdir(parents=True)
    metadata = out / "metadata"
    copies = {}
    for relative, source in inputs.items():
        target = metadata / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        checksum = sha(target)
        if sha(source) != checksum:
            raise RuntimeError(f"CSV changed during staging: {source}")
        copies[relative.as_posix()] = {"source_path": str(source), "sha256": checksum}
    write(out / "metadata_sources.json", copies)

    config = _config({
        "source_root": str(ROOT / "third_party/SlotSPE"),
        "metadata_root": str(metadata),
        "features_root": str(features),
        "study": "blca",
        "label_col": "survival_months_dss",
        "signature": "combine",
        "splits_dir": "5fold_uni2h",
        "folds": [0],
        "seeds": [3],
    })
    write(out / "resolved_config.json", config)
    manifest = prepare(config, out)
    pathways = len(manifest["omic_sizes"])
    if pathways != EXPECTED_PATHWAYS:
        raise ValueError(
            f"SRO-Surv requires {EXPECTED_PATHWAYS} pathways after RNA/signature matching; found {pathways}"
        )
    print(json.dumps({
        "status": "prepared",
        "prepared_root": str(out),
        "split_counts": manifest["folds"]["0"]["counts"],
        "pathways": pathways,
        "encoding_dim": manifest["encoding_dim"],
    }, indent=2))


if __name__ == "__main__":
    main()
