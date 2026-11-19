# SRO-Surv

Slot Read Overlap Regularization for Survival Prediction. The default configuration uses SRO on all 10 histopathology slot iterations, with margin 0.90.

From this folder:

```powershell
python -X utf8 -B scripts/run_sro_surv.py
python -X utf8 -B scripts/check_sro_surv.py
```

The first command prints the configuration. The second runs a small synthetic integration check. Add `--device cuda:0` to the second command for the CUDA check.

- Model and training engine: `research_models/sro_surv/`
- Default configuration: `configs/sro_surv_v1.json`
- Explicit no-SRO control: `configs/scaled_slot16_v2.json`
- Shared runtime and original SlotSPE source: `research_models/common_private_synergy/`, `slotspe_proof/`, and `third_party/SlotSPE/`
- Python dependencies: `requirements-audit.txt`. Use a compatible existing PyTorch environment.

Patient execution is opt-in via `--run --prepared-root <existing-prepared-directory> --output-root <fresh-directory>`. The prepared directory and source referenced by its configuration must be available and pass the recorded split and source checks. Run outputs must be kept inside this project's ignored `local_evidence/`. This package contains source and configuration files only. Use the source attribution in `NOTICE.md`. `SOURCE_SNAPSHOT.json` records export provenance and SHA-256 checksums.

Running the synthetic check does not establish performance on patients.
