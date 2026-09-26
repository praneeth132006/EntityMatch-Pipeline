# EntityMatch Pipeline

Offline business entity resolution for the ML Challenge 2026. The pipeline maps each
Source 1 reference entity to zero or more Source 2/3 records, using indexed candidate
retrieval followed by a CatBoost classifier.

**Status: all workflow code is implemented, including document generation, packaging,
and reproduction from an extracted ZIP. Tests use synthetic data. Full competition
training, evaluation, test predictions, and the final competition ZIP remain deferred
at the user's request. No competition score is claimed.** See `memory.md` for the next steps.

## Setup

Use Python 3.12 on macOS or Linux and a C++17 compiler (`c++`, or set `CXX`).
The native retriever is compiled automatically on first use. A little-endian CPU is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

`requirements.txt` pins direct runtime dependencies; `requirements-lock.txt` records
all packages in the tested environment, including test dependencies. CatBoost is
Apache-2.0 licensed (as recorded in the installed package metadata). There are no
pretrained model downloads. Project code and models trained with this project are
provided under the MIT license; third-party dependencies keep their own licenses.
The competition dataset is not included or relicensed.

## Complete workflow in one command

After freeing sufficient disk space and choosing the registered team details:

```bash
python src/pipeline.py run \
  --data-dir /Users/praneeth/Downloads/student_resource/dataset \
  --work-dir artifacts --output-dir output \
  --team 'Your registered team' --members 'Member One, Member Two'
```

This trains, predicts, validates, generates a two-page methodology, and packages the
submission. It writes the generated methodology to the selected output directory,
so it does not overwrite the repository's draft. The default ZIP destination is
`<output-dir>/EntityMatch_submission.zip`; override it with `--destination`.
The command checks required inputs, training settings, and the document dependency
before training. It is a full corpus run, not a smoke test. Do not run it until ready.

For a custom training budget, `run` accepts the same `--sample-mod`, `--sample-keep`,
`--top-k`, `--block-cap`, `--iterations`, and `--threads` flags as `train`. The exact
settings are saved, so reproduction does not silently revert to defaults.

## Reproduce the pipeline

The input directory must contain these supplied files:

```text
dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
```

Run from the repository root (or the extracted ZIP's `code/business_entity_resolution/`).
Replace the example dataset path with the actual path. All commands accept paths
containing spaces. Use the same work directory for training and prediction.

```bash
python src/pipeline.py train \
  --data-dir /Users/praneeth/Downloads/student_resource/dataset \
  --work-dir artifacts --report reports/evaluation.json

python src/pipeline.py predict \
  --data-dir /Users/praneeth/Downloads/student_resource/dataset \
  --work-dir artifacts --output-dir output

python src/pipeline.py validate \
  --data-dir /Users/praneeth/Downloads/student_resource/dataset \
  --output-dir output
```

Training defaults to a deterministic 2% sample of reference entities (about 44,000).
**All reference rows still determine blocking frequencies, and all target records
are scanned.** This is not a fast smoke test. Increase the sample using
`--sample-keep 50 --sample-mod 1000`, or use `--sample-keep 1 --sample-mod 1` for all
training entities only on a machine with sufficient memory. The default design targets
a 16 GB machine, but full-data peak memory and runtime have not yet been benchmarked.
Keep at least 8–10 GB of free disk space for environment files, outputs, packaging,
and operating-system headroom. `--work-dir` and `--output-dir` can point to an external disk.
The native retrieval stage is single-threaded; `--threads` controls CatBoost (default 4).

The training command writes `model.cbm`, `model_config.json`, `training_config.json`,
`evaluation.json`, and SHA-256 hashes of input files under the work directory. The evaluation report contains corpus counts, candidate reduction ratio, threshold,
held-out macro F0.5, pair precision/recall, blocking recall, country breakdowns,
feature importance, and example errors. Prediction records output hashes, model identity,
and test retrieval counts in `prediction_stats.json`. The final model is refit on all sampled
training entities after the held-out measurement; it never trains on test labels.

## Method

1. Normalize names and addresses, remove a small set of legal name suffixes, expand
   common address variants, and fold common Latin accents. Other UTF-8 bytes are
   preserved. Country is an arbitrary normalized string, with an explicit equality
   check; France and future country labels are supported without a whitelist.
2. Index Source 1 records by normalized names, sorted name tokens, addresses, and
   combinations of name-token prefixes with address-word prefixes or numeric tokens.
   Skip keys found in more than 64 reference records. This cap is applied before
   training reference sampling, so a sample does not make common keys appear rare.
3. Stream Source 2/3 records through the index. Deduplicate retrieved reference hits,
   discard pairs with name bigram Dice below 0.12 or weighted similarity below 0.27,
   and keep at most 12 candidates per reference. The weight is 0.65 name / 0.35 address.
   These deterministic operations are the final blocking stage, not classifier inference.
4. Compute 22 name/address/number similarity and missingness features. IDs, source
   numbering, and country identity are not classifier features. A depth-6 CatBoost
   model learns from true and false candidate pairs; no external data is used.
5. Hash reference IDs into 60% fitting / 20% threshold-tuning / 20% held-out evaluation
   groups, independently of the sampling hash. The tuning split controls early stopping
   and the probability threshold. The holdout is used only for the reported measurement.
6. Score every final candidate once. Write those exact IDs to `candidate_pairs.tsv`
   and above-threshold IDs to `matching_results.tsv`. Empty results remain empty.

The upper candidate count is `top_k * number_of_references`. The sorted posting index
and bounded lists avoid materializing the full reference-by-target Cartesian product.
Retrieval still scans the complete target corpus and requires an in-memory reference
index; this implementation is not a distributed billion-record system.

Both files retain the input reference order and contain exactly one row per Source 1
record. Candidates are deterministically ordered by ID. The validator checks schema,
coverage, duplicate IDs, match/candidate containment, and actual target-ID existence.
It holds distinct referenced target IDs in memory but streams the source files.

## Metric and limitations

For each reference entity, score `1.25 * TP / (predicted_count + 0.25 * true_count)`.
A correctly empty singleton scores 1; a singleton with any predicted match scores 0.
The macro score averages over every reference, including those with no candidates.
Unretrieved true matches remain false negatives in evaluation.

Country equality, fixed block caps, top-k truncation, and heavy name corruption can
reduce recall. Numeric-address features are approximate, not a country-specific postal
parser. Shared names/addresses can cause false merges. France has no labeled training
records, so its generalization accuracy cannot be measured on the supplied training set.
The full dataset must be evaluated before changing defaults or claiming performance.

## Final submission (after the deferred run)

1. Inspect `reports/evaluation.json` and the errors after training.
2. Generate the two-page methodology using the document command below (or use `run`,
   which does this automatically). Review `output/Documentation_template.md` and
   `output/pdf/Approach.pdf`, including actual metrics and registered team details.
   The repository-level draft remains marked as pending until deliberately updated.
3. Run the supplied organizer validator as an independent check:

```bash
python /Users/praneeth/Downloads/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir /Users/praneeth/Downloads/student_resource/dataset/test
```

The official validator skips target-ID membership by default; our validator includes it.
The official `--check-ids` option may need additional RAM.

4. Package only after validation and methodology completion:

```bash
python src/pipeline.py package \
  --data-dir /Users/praneeth/Downloads/student_resource/dataset \
  --work-dir artifacts --output-dir output \
  --methodology output/Documentation_template.md --approach-pdf output/pdf/Approach.pdf \
  --destination output/EntityMatch_submission.zip
```

Packaging revalidates the current files and refuses a methodology marked `NOT YET RUN`.
The archive contains `output/` with both TSVs, self-contained code under
`code/business_entity_resolution/`, and `Documentation_template.md` at its root.
The `--approach-pdf` file is included when explicitly supplied; `run` supplies its
newly generated PDF automatically. This avoids silently bundling a stale draft.
The archive also includes the trained model, exact training settings, input/output
hashes, evaluation reports, and a per-file SHA-256 `MANIFEST.json`. Packaging rejects
changed outputs or inconsistent model/report identities and verifies ZIP integrity
before replacing an existing archive.
Do not upload a code-only checkpoint ZIP as the final competition package.
Upload `matching_results.tsv` to the leaderboard and the full ZIP as the final package.

## Repository layout

- `src/retrieve.cpp`: native normalization, blocking, bounded retrieval, and features.
- `src/pipeline.py`: training, evaluation, prediction, strict validation, packaging, full-run, and reproduction CLI.
- `tests/test_pipeline.py`: metric, retrieval, output-contract, and synthetic integration tests.
- `Documentation_template.md`: methodology draft, later filled with measured results.
- `memory.md`: decisions, run status, commits, and continuation checklist.
- `reports/`: tracked verification summaries; competition evaluation is pending.
- `artifacts/`, `tmp/`, large generated TSVs/ZIPs: local outputs excluded from Git.

Generated submission artifacts larger than GitHub's normal file limits should be
published as release assets after they exist; raw competition inputs stay local.

To regenerate the two-page methodology after evaluation:

```bash
python -m pip install -r requirements-docs.txt
python src/build_document.py --team 'Your registered team' --members 'Member names' \
  --report artifacts/evaluation.json --prediction-stats output/prediction_stats.json \
  --markdown-output output/Documentation_template.md --output output/pdf/Approach.pdf
```

This writes the selected Markdown and PDF outputs from the measured evaluation.
Without that report, it produces an explicitly marked draft. A generated document
that exceeds two pages is rejected rather than silently violating the page limit.
Review the regenerated document before packaging. Prediction also verifies fingerprints
of the trained model and retrieval source so changed retrieval code requires retraining.


## Verify an extracted submission

From the extracted `code/business_entity_resolution/` directory, install the pinned
requirements and run:

```bash
python src/pipeline.py reproduce --data-dir /path/to/dataset --output-dir reproduced
```

This checks the supplied dataset against the packaged SHA-256 manifest, loads the
packaged model, regenerates both TSVs, validates them, and compares their bytes with
the original output hashes. To rebuild the model from the original training data
using the exact saved sample and training settings:

```bash
python src/pipeline.py reproduce --data-dir /path/to/dataset --output-dir reproduced --retrain
```

`--retrain` updates the model in the selected work directory. Both modes write
`reproduction.json`; any output mismatch produces a failing exit status. Differences
across library/platform environments are therefore detectable rather than hidden.
A full synthetic run, ZIP extraction, saved-model reproduction, and retraining
reproduction are exercised by the integration test.

For synthetic verification runs, set `--data-label synthetic` on `train` or `run`.
The generated report and document then explicitly label results as synthetic.

## Completion and input integrity

See `reports/code-completion.md` for the implemented requirement checklist. The final
audit suite includes singletons and multiple true matches in the full pipeline test.
`pytest.ini` limits discovery to `tests/`, including inside extracted submissions.

Training fingerprints only the required training files; prediction fingerprints the
actual test files it uses, even when they come from a different dataset directory.
Packaging checks that the supplied dataset matches both manifests. Extra unrelated
TSVs do not affect reproduction. Generated artifact paths must remain outside the
input dataset directory, including paths routed through symlinks.

Input headers are checked before a full scan. ID lists reject empty entries,
duplicates, invalid prefixes, whitespace, and control characters. A failed validation
removes a previous PASS report so it cannot be mistaken for a current result.
