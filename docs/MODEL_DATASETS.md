# Model-specific legal intent datasets

Run `python scripts/materialize_model_datasets.py` or trigger the **Materialize model datasets** GitHub Action. The action commits generated JSONL files under `datasets/models/`.

## Shared protocol
- Canonical train: all records from `datasets/legal/train.jsonl`.
- Frozen evaluation: all 132 records from `datasets/legal/eval.jsonl`; never split or augment it.
- Deterministic 80/20 split of canonical train using seed 0. With 14,806 records this is **11,844 train / 2,962 calibration** (floor at 80%). The earlier notes saying 11,845/2,961 used a different rounding convention; the script uses an explicit floor.
- All formats retain the same source examples and labels. Tev1 uses letter labels A-L in the canonical label order. Strands uses 12 described options and an integer option index.

## Outputs
- `canonical/{train,eval}.jsonl`, `canonical/labels.json`
- `laya/{train,calibration,eval}.jsonl`
- `tev1/{train,validation,eval}.jsonl`
- `strands/{train,calibration,eval}.jsonl`
- `jev/eval.jsonl` (evaluation only; Jev is not locally fine-tuned here)
- `manifest.json` with split seed and record counts

Important: the original source dataset is synthetically augmented; converting formats does not create new independent examples. The descriptions in Strands options are explicit model-facing descriptions and should be kept identical across repeated runs. Jev receives only the frozen eval set because it is a closed/API model.
