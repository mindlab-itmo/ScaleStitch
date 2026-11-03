# Experimental protocol

## Model and inputs

The host is a frozen 3DGraphLLM with Llama-3-8B-Instruct. ULIP-2 features (1280 dimensions) pass through a fitted affine bridge to the host's 1024-dimensional input. Native Uni3D features use `host.feature_mode="native"` without a bridge. The original projectors, object identifiers, and LoRA weights are retained.

The revision examples explicitly set `run.repetition_penalty=1.0`. Legacy examples retain `3.0`, also the runtime fallback when no value is supplied. The CLI flag overrides the config. Effective decoder settings are recorded in the run artifacts and prediction rows; compare decoder-matched conditions rather than combining the two protocols.

`data.rows` contains `qid`, `scene_id`, `prompt`, and `assigned_ids`. The last field maps each visible identifier to a source feature-row index. Answers belong in the separate scoring references. Object attributes and feature caches must share the same row order.

`prepare_questions` accepts official ScanQA lists or SQA3D question/annotation wrappers. `scene_object_ids` maps each available scene to its complete row list `[0, 1, ...]`. `cohort.mode="all_available"` retains every question with scene inputs, without selecting by answer or model output. SQA3D preserves the original situation and question.

## Controlled pairs

`prepare_controlled` takes a scene-to-object-list JSON in feature-row order. Each radius object has `source_id`, `category`, `rho` (metres), and `radius_valid`; height objects use `height_m` and `height_valid`. The builder enumerates all pairs of valid objects whose categories are unique within the scene, excluding centimetre ties. It creates direct ID and category-name questions for each pair.

`direct.json` and `named.json` contain model inputs; `scoring/references.json` contains answers for both tasks. `scoring/pair_metadata.json` records roles, row identities, measurements, and targets. Pair selection and exclusions are recorded in `scoring/selection_summary.json`.

Height uses the full observed object's extent along the aligned Z axis. Its source JSON has `measurement="full_object_axis_aligned_height"`, `input_id` (SHA256 of the inference attributes), `source_reference`, and `verified_scenes` (scene-to-boolean mapping). Set `run.measurement_source` to this file and `measurement="height"` in the conditions. The declaration identifies an external geometry check; parsing it does not verify the coordinate transform.

## Measurement and feature conditions

Measurement records cover every visible object independently of the question. The default format is centimetres, appended after the question.

| Configuration | Condition |
|---|---|
| `representation="absent"` | No measurement packet. |
| `representation="lexical", mode="true"` | Correct object measurements. |
| `representation="lexical", mode="constant"` | Every object receives `run.constant_cm` (default 100). |
| `representation="lexical", mode="swapped"` | Cyclic reassignment across active source rows. |
| `representation="masked"` | Masked content with the correct packet's token count. |

The masked condition tests packet content at matched length; question-mark tokens are not assumed to be neutral. Radius on height questions is a separate property control, not an alternative height estimate.

`object_intervention` in each condition accepts `full`, `zero_object_slots`, `drop_object_features`, or `permute_object_slots`. Zeroing and permutation occur after projection and positional additions. The permutation follows the same cyclic donor map as reassigned measurements, so a joint permutation can align both channels to the wrong object. Dropping removes object-feature tokens; zeroing retains their slots.

`run.channel_intervention` accepts `original`, `zero_images`, or `zero_relations`. Channel zeroing is applied after projection, including projector bias. Object IDs and the other channels remain in place.

## Missing channels

Without a missing-channel declaration, absent required features are rejected during preparation or inference preflight; selected graph edges are checked against the scene neighborhood. For a predefined partial-coverage input, set both `missing_channel_policy="zero_projected_declared"` and `missing_channels` (JSON path) in the preparation config:

```json
{"scene0001_00": {"image_rows": [2], "relation_pairs": [[0, 2], [2, 0]]}}
```

Indices refer to source rows, not dataset object IDs. The declaration must match the missing source vectors. Preparation writes manifest v2 and sets the matching model policy. The host masks those slots after projection; a zero-valued vector present in a cache is not treated as missing.

## Instructions and scores

The candidate instruction is appended only to controlled direct questions. The natural-answer instruction is appended to natural questions. Each instruction remains identical across measurement conditions, including the absent baseline. Exact wording is defined in `controlled_protocol.py`.

Controlled direct and named scoring uses `scoring="native_id"`. Both the raw and mapped output must be a single `<OBJnnn>` identifier, allowing terminal `.`, `!`, or `?`; invalid IDs score zero. Visible-to-source mapping must agree with the recorded inference rows. Correctness requires the target object; candidate-pair compliance separately tests membership in either offered object. The scoring command checks the inference-row hash and the completed run artifacts.

The revised natural-question endpoint uses `scoring="host_normalized_em1"`. It applies the published 3DGraphLLM `clean_answer` function at commit `19d65cd9e220ec8e7704acc989aaef5e367f31dc` to the prediction and every reference, then tests nonempty exact equality. For example, `2` and `two` match under this cleaner. Blank and invalid-object-ID outputs remain errors, and every question in the fixed cohort stays in the denominator. The metric identifier is `host_normalized_em1_3dgraphllm_19d65cd9_v1`; the scoring manifest records its source and policy. This endpoint was selected during revision after identifying numeric-surface mismatches; it is not a claim of official SQA3D or leaderboard-equivalent evaluation.

Legacy `scoring="natural_qa"` uses protocol-normalized EM: lowercase, collapse whitespace, remove terminal `.`, `!`, and `?`, then remove one leading `a`, `an`, or `the`. Internal punctuation and numerical signs remain, so `2` and `two` do not match. The punctuation-only sensitivity retains articles. Raw and lowercase/whitespace EM are also exported. Keep this endpoint labelled as legacy and recompute paired estimates when changing the primary score.

## Paired analysis

All contrasts use complete fixed cohorts. Bootstrap resampling uses ScanNet scene families (the scene prefix before the final underscore), retains all paired question outcomes, and weights by the number of questions. The default is 10,000 resamples. `family_size` sets the Bonferroni-adjusted confidence level; an interval crossing zero does not establish equivalence.

`diagnostic_metrics` reads `scored_rows.json` banks with explicit condition aliases. Its analysis types are:

| `kind` | Required analysis fields | Comparison |
|---|---|---|
| `instruction` | `new`, `old`, `interaction_family_size` | Change in the correct-minus-absent effect. |
| `instruction_reassigned` | `at`, `continuation`, `old` | Change in correct-minus-reassigned separation. |
| `feature` | `new`, `old` | Object-feature control versus original features. |
| `between_features` | `zero`, `permute` | Difference between zeroed and permuted features. |
| `property` | `radius`, `height`, `radius_height`, `metadata` | Matched radius/height effects and the opposite-target subset. |
| `linear` | `bank`, `contrasts`, `family_size` | Explicit weighted contrasts within a bank. |

Instruction primary families contain 8 direct-task or 4 natural-task interactions; reassignment uses 8, and cross-property uses 4. Local instruction, property, and feature comparisons are labelled separately by the implementation. For other comparisons, pass the full tested family size, including contrasts distributed across runs. Property analysis joins by scene and source-object ID pair, and never selects pairs using model outcomes.

## Matched adaptation

`packet_lora_cli` trains the existing object-only ULIP-2 host under two arms: absent measurements and correct full-scene measurements. The deployment bundle must declare `architecture.adapter_kind="identity_no_radius"` and contain an empty `state.scalestitch`; pinning and validation check the actual CPU payload. The affine feature bridge and normalization remain. The input question order, initial checkpoint and optimizer schedule are shared. Only the existing 448 FP32 LoRA tensors (41,943,040 parameters) are trainable; base, projector, embedding and head parameters remain frozen. The base transformer uses FP16; the embeddings, head and LoRA tensors retain their loaded FP32 storage.

The fixed training cohort contains 35,321 radius and 35,330 height questions. A single permutation uses Python `random.Random(20261102).shuffle` on radius rows followed by height rows. One epoch uses accumulation 16, including the final 11-question group, and 4,416 successful updates. AdamW uses learning rate 0.00002, weight decay 0.01, gradient clipping 1.0, 133 linear warmup updates and linear decay to zero. Loss is teacher-forced cross-entropy on the native identifier plus the exact `host.end_sym`, averaged over answer tokens and then over examples in each group. No input-prefix position is supervised.

Each arm requires its own two-question backward/checkpoint smoke. The fixed pilot ends after 3,504 questions and 219 updates; full-epoch continuation requires its complete validation. Validation evaluates all 2,187 radius and 2,183 height questions under both absent and correct packets, with repetition penalty 1.0. Pilot and epoch selection is fixed before evaluation.

`pin` records every model/cohort input hash in a new config. `validate` checks these hashes, exact ordered cohorts, target identities and scene-family separation without loading a GPU model. Training checkpoints bind the config, source hashes, inputs, arm, optimizer, scaler and RNG. An AMP overflow retries the same group from its pre-group RNG with a reduced scale, at most eight times; it does not advance the question or update count. Interrupted training resumes at the last clean optimizer boundary. Immutable stage links preserve the exact pilot/epoch checkpoint bytes separately from the rotating `last.pt`/`previous.pt` resume slots.

Validation uses the actual loaded stage checkpoint and preserves one SQLite row per question and condition. Resume rechecks each stored question/scene/task/assignment binding and recomputes its strict native-ID hit before producing another answer. The terminal manifest seals the completed database hash. Test scene names are used only for the split-overlap check; this command never loads test answers.

`export-host` merges the completed immutable epoch into a new copy of the source host checkpoint. It requires exact source-to-trained LoRA name/shape coverage, all 448 finite FP32 tensors and the completed epoch's identity and hash. All other source tensors remain unchanged. The exported checkpoint is loaded by the existing frozen-host inference command, allowing the same held-out cohorts, conditions and scorers to be reused.

## Vanilla text baseline

`text_baseline_cli` uses unmodified Llama-3-8B-Instruct with ordinary labels (`A`, `B`, ..., `AA`, ...). Source questions, visible/source assignments, features and cohorts come from the matching 3D condition; no multimodal features enter the language model. All scene-wide packet interventions are constructed first, then the focused conditions retain the queried two records without changing reassignment donors. The seven conditions are absent, full true/constant/swapped and pair true/constant/swapped.

Preparation stores the exact base chat text, token IDs and input hashes. Generation uses the local model's `generation_config.json` EOS IDs, greedy decoding, 32 answer tokens and repetition penalty 1.0. Both strict-label scoring and the predeclared whole-label sensitivity are reported, with failures retained in every fixed denominator. Scoring reads references separately and computes 10,000 scene-family bootstrap resamples (seed 20260911): three full-packet contrasts, three focused-packet contrasts and two scope contrasts, with the corresponding family adjustments.
