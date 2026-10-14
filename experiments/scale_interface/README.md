# ScaleStitch experiments

Frozen 3DGraphLLM experiments for **When Size Helps and Hurts: Physical Measurements in 3D Question Answering**.

This version adds complete controlled question sets, declared missing-channel masks, equal-length masked packets, candidate-answer instructions, strict object-ID scoring, and the property, instruction, and feature diagnostics used in the article.

## Files

| File | Purpose |
|---|---|
| `prepare_questions.py` | Export ScanQA/SQA3D questions, cohorts, and separate reference answers. |
| `prepare_controlled.py` | Build all eligible radius/height pairs with matched ID and category-name questions. |
| `prepare_full_inputs.py` | Assemble object, image, and graph inputs from cached features. |
| `__main__.py`, `host.py`, `natural_qa.py` | Load the frozen host, check shared input coverage, and generate answers. |
| `score_natural_qa.py`, `prediction_scoring.py` | Score completed runs and export per-question diagnostic outcomes. |
| `diagnostic_metrics.py`, `matched_question_metrics.py` | Paired effects, interactions, and scene-family bootstrap intervals. |
| `configs/` | Editable examples for preparation, inference, and scoring. |
| `PROTOCOL.md` | Input schemas, experimental conditions, and scoring definitions. |

## Run

Run from the repository root; paths in JSON configs are relative to that directory. Set the asset paths before running. Every `--output` must name a new directory.

Required assets: the Llama-3-8B-Instruct model, the trained 3DGraphLLM host checkpoint, object features/radii/attributes, image and relation features, and question annotations. ULIP-2 uses the fitted bridge checkpoint in `host.deployment`; native Uni3D uses `configs/full_gt_native.json`. Feature extraction and bridge fitting precede these commands.

Natural questions:

```powershell
python -m pip install -r experiments/scale_interface/requirements.txt
python -m experiments.scale_interface.prepare_questions --config experiments/scale_interface/configs/prepare_sqa3d.json --output assets/prepared_qa
python -m experiments.scale_interface.prepare_full_inputs --config experiments/scale_interface/configs/prepare_full.json --output assets/full_inputs
python -m experiments.scale_interface generate-scene --config assets/full_inputs/config.json --output runs/full_01
python -m experiments.scale_interface.score_natural_qa --config experiments/scale_interface/configs/score_natural_qa.json --output runs/full_01_metrics
```

Use `prepare_scanqa.json` for ScanQA. The full-test SQA3D example keeps every question with available scene inputs and writes missing-scene exclusions. To reproduce the added natural-answer instruction, set `instruction` to `natural` in the preparation config.

Controlled questions:

```powershell
python -m experiments.scale_interface.prepare_controlled --config experiments/scale_interface/configs/prepare_controlled.json --output assets/controlled
```

Set `data.rows` in a copy of `object_only.json` to `assets/controlled/direct.json` or `assets/controlled/named.json`, then use that copy as `base_config` for full-input preparation. Both direct and named questions use `configs/score_controlled.json` (`scoring="native_id"`); its `questions` path must point to the exact `data.rows` file used for inference. Both tasks ask for an object ID and share the pair metadata. Set `property` to `height` and provide `measurement_source` for height comparisons; set `instruction` to `candidate` for the added direct-answer instruction.

Diagnostics after scoring:

```powershell
python -m experiments.scale_interface.diagnostic_metrics --config experiments/scale_interface/configs/diagnostics.json --output runs/diagnostics
python -m unittest discover -s experiments/scale_interface/tests
```

`score_*.json` enables `export_scored_rows`: these compact outcomes feed the diagnostic banks. `configs/diagnostics.json` illustrates instruction, reassignment, property, and feature analyses; [PROTOCOL.md](PROTOCOL.md) lists the other analysis types.

## Outputs and controls

- Preparation: inference questions, fixed cohorts, separate `scoring/` references, and source hashes. Full-input preparation also writes features, coverage exclusions, and a ready-to-run `config.json`.
- Generation: `predictions.jsonl`, `cohort.json`, `plan.json`, `effective_config.json`, and `status.json`. Score after `status` is `complete`.
- Scoring: `metrics.json`, `scored_rows.json`, and an input/output hash manifest. Accuracies and effects are fractions; multiply by 100 for percentages or percentage points.

For declared missing images/relations, use `prepare_full_declared.json` with the source-row mask described in [PROTOCOL.md](PROTOCOL.md#missing-channels). Without a declaration, missing required channels are rejected during preparation or inference preflight.

Measurement conditions are absent, correct, constant, reassigned, and equal-length masked. Image/graph ablations use `run.channel_intervention`; object ablations use `object_intervention` in each condition. These controls preserve the trained host. The `full_` prefix in condition names describes object features; `host.model.host_mode` selects the object-only or full multimodal host.
