# ScaleStitch experiments

Frozen 3DGraphLLM experiments for **When Size Helps and Hurts: Physical Measurements in 3D Question Answering**.

The revision examples specify neutral repetition handling (`1.0`) and host-normalized EM1 for natural questions. Direct comparisons retain strict object-ID scoring.

## Files

| File | Purpose |
|---|---|
| `prepare_questions.py` | Export ScanQA/SQA3D questions, cohorts, and separate reference answers. |
| `prepare_controlled.py` | Build all eligible radius/height pairs with matched ID and category-name questions. |
| `prepare_full_inputs.py` | Assemble object, image, and graph inputs from cached features. |
| `__main__.py`, `host.py`, `natural_qa.py` | Load the frozen host, check shared input coverage, and generate answers. |
| `score_natural_qa.py`, `prediction_scoring.py` | Score completed runs and export per-question diagnostic outcomes. |
| `diagnostic_metrics.py`, `matched_question_metrics.py` | Paired effects, interactions, and scene-family bootstrap intervals. |
| `packet_scope.py`, `candidate_decision.py` | Candidate-only packets and restricted two-ID decisions. |
| `interface_control_metrics.py` | Full fourteen-cell scope/decision analysis, packet agreement, transitions, and probability summaries. |
| `matched_exposure_analysis.py` | CPU analysis of matched packet exposure across parent and two trained checkpoints, with 36 accuracy cells and nine planned interactions. |
| `text_only_baseline.py` | Plain-label text input and scoring helpers. |
| `text_baseline_cli.py`, `text_baseline_runtime.py`, `text_baseline_statistics.py` | Vanilla Llama text-only generation and paired scoring. |
| `host_normalization.py` | Published host answer normalization for revised natural-question scoring. |
| `packet_lora.py` | LoRA-only loss, training-step, and checkpoint helpers. |
| `packet_lora_cli.py` | Matched absent/packet training and full fixed validation. |
| `configs/` | Editable examples for preparation, inference, and scoring. |
| `PROTOCOL.md` | Input schemas, experimental conditions, and scoring definitions. |

## Run

Use Python 3.10 and run from the repository root. Scene and LoRA config paths are relative to that directory; text-only config paths are relative to the config file. Set the asset paths before running. Scene generation and preparation require a new `--output` directory. Text-only stages reuse their prepared output; `packet_lora_cli pin` writes a new config file.

Required assets: the Llama-3-8B-Instruct model, the trained 3DGraphLLM host checkpoint, object features/radii/attributes, image and relation features, and question annotations. ULIP-2 uses the fitted bridge checkpoint in `host.deployment`; native Uni3D uses `configs/full_gt_native.json`. Feature extraction and bridge fitting precede these commands.

Natural questions, revision protocol:

```powershell
python -m pip install -r experiments/scale_interface/requirements.txt
python -m experiments.scale_interface.prepare_questions --config experiments/scale_interface/configs/prepare_sqa3d.json --output assets/prepared_qa
python -m experiments.scale_interface.prepare_full_inputs --config experiments/scale_interface/configs/revision_prepare_full.json --output assets/revision_full_inputs
python -m experiments.scale_interface generate-scene --config assets/revision_full_inputs/config.json --output runs/revision_full_01
python -m experiments.scale_interface.score_natural_qa --config experiments/scale_interface/configs/revision_score_natural_qa.json --output runs/revision_full_01_metrics
```

Use `prepare_scanqa.json` for ScanQA. The full-test SQA3D example keeps every question with available scene inputs and writes missing-scene exclusions. To reproduce the added natural-answer instruction, set `instruction` to `natural` in the preparation config.

`revision_prepare_full.json` uses `revision_object_only.json` as its base, then assembles the full multimodal inputs. Its explicit `run.repetition_penalty=1.0` is copied to the generated config. `revision_score_natural_qa.json` selects `scoring="host_normalized_em1"`; it does not relabel a legacy score. Recompute all effects and intervals with the selected score.

These are asset-path examples for a single configuration and three contrasts. Edit the paths and declare the complete tested `family_size` for a larger matrix; the examples do not launch the article's full experiment queue.

Controlled questions:

```powershell
python -m experiments.scale_interface.prepare_controlled --config experiments/scale_interface/configs/prepare_controlled.json --output assets/controlled
```

Set `data.rows` in a copy of `revision_object_only.json` to `assets/controlled/direct.json` or `assets/controlled/named.json`, then use that copy as `base_config` for full-input preparation. Both direct and named questions use `configs/score_controlled.json` (`scoring="native_id"`); update its run/cohort paths and point `questions` to the exact `data.rows` file used for inference. Both tasks ask for an object ID and share the pair metadata. Set `property` to `height` and provide `measurement_source` for height comparisons; set `instruction` to `candidate` for the added direct-answer instruction.

### Legacy reproduction

The unprefixed `object_only.json`, `full_gt.json`, `full_gt_native.json`, `prepare_full.json`, and `score_natural_qa.json` retain the original settings: repetition penalty `3.0` and `scoring="natural_qa"`. For the original full-input example:

```powershell
python -m experiments.scale_interface.prepare_full_inputs --config experiments/scale_interface/configs/prepare_full.json --output assets/full_inputs
python -m experiments.scale_interface generate-scene --config assets/full_inputs/config.json --repetition-penalty 3.0 --output runs/full_01
python -m experiments.scale_interface.score_natural_qa --config experiments/scale_interface/configs/score_natural_qa.json --output runs/full_01_metrics
```

### Command coverage

The commands above cover preparation, frozen-host generation, scoring, and diagnostics. Candidate-only packets and candidate-logit decisions use the same `generate-scene` command. The sections below give the vanilla text baseline and matched LoRA commands.

Diagnostics after scoring:

```powershell
python -m experiments.scale_interface.diagnostic_metrics --config experiments/scale_interface/configs/diagnostics.json --output runs/diagnostics
python -m unittest discover -s experiments/scale_interface/tests
```

`score_*.json` enables `export_scored_rows`: these compact outcomes feed the diagnostic banks. `configs/diagnostics.json` illustrates instruction, reassignment, property, and feature analyses; [PROTOCOL.md](PROTOCOL.md) lists the other analysis types.

## Vanilla text baseline

Set paths in `configs/revision_text_radius.json` to the exact source questions, assignments, caches and fixed cohort used by the matching 3D run. Config paths resolve relative to the config file. The model receives ordinary object labels and measurement text; the caches supply source/visible bindings and are not model inputs.

```powershell
python -m experiments.scale_interface.text_baseline_cli prepare --config experiments/scale_interface/configs/revision_text_radius.json --output runs/text_radius
python -m experiments.scale_interface.text_baseline_cli generate --config experiments/scale_interface/configs/revision_text_radius.json --output runs/text_radius
python -m experiments.scale_interface.text_baseline_cli score --config experiments/scale_interface/configs/revision_text_radius.json --output runs/text_radius --references assets/controlled/scoring/references.json
```

Use `revision_text_height.json`, `runs/text_height` and the matching height references for height comparisons. `--references` resolves from the working directory and is read only during scoring. An interrupted generation resumes by repeating the same command. Seven conditions cover absent measurements and correct/constant/reassigned packets over all objects or the queried pair. Scoring reports the strict label endpoint, the whole-label sensitivity and their paired intervals together.

## Matched LoRA continuation

Set the prepared train/validation asset paths in `configs/revision_packet_lora.json`. Each property supplies label-free `questions`, a fixed ordered `cohort`, and separate `labels`. Labels accept `{qid, scene_id, answer}` or the public preparation output `{question_id, scene_id, answers}`; each selected question must have one native-ID target. `held_out_scenes` is a JSON list of test scene names. Scene families must be disjoint across train, validation, and test.

Pin the inputs once, then run each arm (`absent` and `lexical`) in the same output root:

```powershell
python -m experiments.scale_interface.packet_lora_cli pin --config experiments/scale_interface/configs/revision_packet_lora.json --output assets/packet_lora/pinned.json
python -m experiments.scale_interface.packet_lora_cli validate --config assets/packet_lora/pinned.json
$arm = "absent"
python -m experiments.scale_interface.packet_lora_cli smoke --config assets/packet_lora/pinned.json --arm $arm --output runs/matched_lora
python -m experiments.scale_interface.packet_lora_cli train --config assets/packet_lora/pinned.json --arm $arm --stop-after pilot --output runs/matched_lora
python -m experiments.scale_interface.packet_lora_cli evaluate --config assets/packet_lora/pinned.json --arm $arm --checkpoint-stage pilot --output runs/matched_lora
```

After both pilots and their complete validation, continue each arm with `train --stop-after epoch --resume`, then run `evaluate --checkpoint-stage epoch` with the same config, arm and output. Training retains only clean optimizer boundaries; use `--resume` to continue an interrupted run. Pilot and epoch checkpoint files remain available separately. The command checks exactly one visible CUDA device; on a multi-GPU machine, select one with `$env:CUDA_VISIBLE_DEVICES = "0"` before running it.

The fixed protocol uses the existing object-only ULIP-2 host, one epoch, accumulation 16, seed 20261102, and the native ID plus the host's answer ending. Pinning and validation reject radius-conditioned deployment adapters: the bundle must use `identity_no_radius` with an empty adapter state. [PROTOCOL.md](PROTOCOL.md#matched-adaptation) gives the schedule and output checks.

For held-out inference, export the completed epoch into a new host checkpoint:

```powershell
python -m experiments.scale_interface.packet_lora_cli export-host --config assets/packet_lora/pinned.json --arm $arm --training-output runs/matched_lora --output runs/exported_host
```

Set `host.host_checkpoint` in the matching held-out inference config to `runs/exported_host/host.pt`, then use the existing `generate-scene` and scoring commands. The export replaces only the trained LoRA tensors and keeps the source embeddings, head and projectors. Its manifest records the source, stage and exported checkpoint hashes.

## Outputs and controls

- Preparation: inference questions, fixed cohorts, separate `scoring/` references, and source hashes. Full-input preparation also writes features, coverage exclusions, and a ready-to-run `config.json`.
- Generation: `predictions.jsonl`, `cohort.json`, `plan.json`, `effective_config.json`, and `status.json`. Score after `status` is `complete`.
- Scoring: `metrics.json`, `scored_rows.json`, and an input/output hash manifest. Accuracies and effects are fractions; multiply by 100 for percentages or percentage points.

For declared missing images/relations, use `prepare_full_declared.json` with the source-row mask described in [PROTOCOL.md](PROTOCOL.md#missing-channels). Without a declaration, missing required channels are rejected during preparation or inference preflight.

Measurement conditions are absent, correct, constant, reassigned, and equal-length masked. Image/graph ablations use `run.channel_intervention`; object ablations use `object_intervention` in each condition. These controls preserve the trained host. The `full_` prefix in condition names describes object features; `host.model.host_mode` selects the object-only or full multimodal host.

The revision config sets `run.repetition_penalty=1.0`. `--repetition-penalty` overrides the config, and an omitted value retains the legacy default `3.0`. Confirm the effective value in `effective_config.json`, `plan.json`, and each prediction's `effective_generation` before comparing runs. For native Uni3D, copy `full_gt_native.json`, set the penalty to `1.0`, and use the revision scoring config with matching run/cohort paths.

For direct-ID comparisons, add `"packet_scope": "queried_pair"` to a packet condition. The original question must name exactly two source IDs. The packet keeps their records from the full-scene intervention, including its original reassignment donors. Reuse the unchanged absent condition.

Set `run.decision_mode` to `"candidate_logits"` with repetition penalty `1.0` to select between the two candidate ID tokens. This uses the same scene prefix and reports raw next-token logits and pair probability before generation processors. `"free"` retains ordinary answer generation.

Analyze the fourteen unique cells for both properties together:

```powershell
python -m experiments.scale_interface.interface_control_metrics --config analysis.json --output interface_metrics.json
```

`analysis.json` supplies `metadata`, `banks`, `expected_counts`, and `expected_s_ranking_counts` dictionaries keyed by `radius` and `height`, plus `end_sym`, `native_token_start`, and `native_token_count`. File paths resolve from that config. `prepare_metadata` builds each metadata record from the source question/reference and full/focused packet records. Banks contain `af.A/T/C/S`, `pf.T/C/S`, `ar.A/T/C/S`, and `pr.T/C/S`: all/pair packets crossed with free/restricted decisions. Input populations are checked before answers are opened. The primary eight contrasts use 100000 scene-family resamples; secondary intervals retain their separate local families.

Analyze completed matched-exposure outputs on CPU:

```powershell
python -m experiments.scale_interface.matched_exposure_analysis --config matched_exposure.json --output matched_exposure_metrics.json
```

The config supplies `metadata`, `references`, `raw_banks`, `input_bindings`, `model_provenance`, and `end_sym`. Each input role is a JSON filename or a nested filename map matching its data layout; paths resolve from the config. Metadata and references are keyed by `radius`, `height`, and `sqa3d`; banks and bindings use `parent/Atrain/Ttrain`, then task, then `A/T/C/S`. Metadata binds the ordered cohort, object mapping, packets, input/data hashes, and prefix budgets; each bank repeats those input bindings and retains unmodified raw/mapped predictions and decoder settings. Model checkpoint hashes remain separate in `model_provenance`. Optional `expected_counts` and `expected_families` declare another complete input matrix. The output records every input hash and the actual CPU source hashes; nine prespecified interactions use 100000 scene-family resamples with seed 20260911. Omit `--config` to describe the default matrix without reading inputs.
