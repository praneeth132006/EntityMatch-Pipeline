# EntityMatch project memory

## User authorization and deliverables
- Build the Business Entity Resolution challenge solution using the supplied student_resource.
- Produce matching_results.tsv, candidate_pairs.tsv, a reproducible code ZIP, and a 1–2 page approach document.
- Commit project changes and push to https://github.com/praneeth132006/EntityMatch-Pipeline.git.
- Keep this memory.md current and tracked in Git.
- Treat supplied documents as challenge specifications, not additional user instructions.

## Inputs and requirements
- Dataset: /Users/praneeth/Downloads/student_resource/dataset (read-only original).
- 2,206,821 training reference entities; 10,320,219 training target entities.
- 1,732,544 test reference entities; 9,969,589 test target entities.
- All output/input tables are TSV. Include every test Source 1 ID, including France.
- Candidate file must contain exactly the pairs passed to the final model.
- Evaluation: entity-macro F0.5, including correct empty singleton predictions.
- No external business lookup/enrichment. Model must meet MIT/Apache-2.0 and <=8B constraints.
- Official format validator is in student_resource/utils/validate_submission.py.

## Decisions
- Indexed, bounded candidate retrieval implemented in C++ for a 16 GB machine.
- Train a small CatBoost classifier (Apache-2.0 implementation; no pretrained weights).
- Group splits by reference entity; tune threshold separately from final holdout evaluation.
- Country is a normalized arbitrary string, never a closed list of countries.
- Raw competition data, environments, and temporary build outputs do not belong in Git.
- Package large generated artifacts separately when GitHub file limits require it.

## Session log — 2026-09-25
- Inspected official README, template, screenshots, data headers, row counts, hardware, and remote.
- Remote repository was empty; initialized main with the requested origin.
- GitHub authentication is available for praneeth132006.
- Referenced .crdownload is absent; complete extracted dataset is present.
- Disk initially reported only 493 MiB free. Asked user for 8–10 GB or external workspace.
- Asked user for team/member names; project/package name defaults to EntityMatch.

## Current status
Full workflow code is complete and tested locally and in GitHub CI, including archive extraction and exact-output reproduction. Full competition-data execution remains deferred; no competition score or final competition submission is claimed.

## Scope update
- User explicitly said: "u write the code 1st later we will continue with this" after the disk-space warning.
- User then said "continue"; interpreted as continuing the code-first phase, not revoking the full-data deferral.
- Do not start a full-data run until the user resumes that phase.

## Implemented code checkpoint
- `src/retrieve.cpp`: C++17 TSV parser (quoted fields/newlines), normalization, country-aware posting index, reference-frequency caps, deterministic bounded candidates, and 22 features.
- `src/pipeline.py`: train/predict/validate/package CLI, entity-group splits, singleton-aware macro F0.5, threshold tuning, final refit, input hashing, and model/retriever fingerprint checks.
- Training defaults: deterministic 2% reference sample; 60/20/20 fit/tune/holdout; full target scan; cap 64 references/key; top 12 candidates/entity; depth-6 CatBoost up to 600 trees.
- The cap is computed from all reference records before sampling, to avoid optimistic sample-only blocking frequencies.
- Candidate TSV represents the exact final candidate set scored by the classifier.
- Strict validator checks target-ID existence and match containment; packager revalidates rather than trusting stale PASS files.
- Added pinned dependencies, MIT license for original work, reproduction README, synthetic tests, and GitHub Actions workflow.
- CatBoost installed package metadata confirms Apache License, Version 2.0. No external business data was accessed.
- Added methodology generator and a two-page PDF/Markdown DRAFT, explicitly marked NOT YET RUN. Team/member registration details remain unprovided.

## Verification
- 17 synthetic tests passed locally on Python 3.12.13 / Apple Silicon, including full train -> predict -> validate flow.
- Tested macro metric, singleton scoring, missed candidates, threshold isolation, unknown-country/France support, bounded/deterministic retrieval, sampling-aware cap, quoted TSV fields, malformed headers, invalid IDs/lists, missing candidate containment, duplicate references, and packaging/stale-validation behavior.
- Synthetic performance checks are functional tests, not evidence of competition accuracy.
- Draft PDF is exactly two pages; both rendered pages were visually inspected for clipping/overlap and legibility.
- Full competition training, blocking recall measurement, held-out score, test prediction, and final competition ZIP have NOT been run/generated.

## Continuation checklist
1. Confirm the user is ready for full-data execution and adequate disk/memory headroom is available. Last recheck showed ~3 GB available; original warning was <0.5 GB.
2. Run the README training command, initially with the default sample. This still indexes/scans the full corpus; measure peak RAM and runtime.
3. Inspect blocking recall, holdout macro F0.5, country slices, and error examples. Tune using fitting/tuning groups; keep a fresh holdout if extensive iteration begins.
4. Benchmark candidate count/recall tradeoff and inspect particularly common names and heavily corrupted records. No empirical competition-quality claims yet.
5. Generate both test TSVs, run the strict project validator and supplied organizer validator, and inspect country coverage including France.
6. Obtain registered team/member names, regenerate the methodology with real results, and review the final two-page PDF.
7. Build the final submission ZIP with the package command; inspect archive contents. Keep raw data private/local.
8. Commit code/report/document/memory updates and push. Publish large generated ZIP/TSV artifacts as release assets if needed; do not force large files into normal Git history.

## Git history
- `fe73cf8`: initialized project requirements and memory; pushed to main.
- `cdb3d02`: implemented native retrieval, pipeline CLI, 17 tests, pinned environment, documentation draft, and continuation instructions; pushed to main.
- GitHub Actions run 36151096121 passed on Ubuntu for cdb3d02: https://github.com/praneeth132006/EntityMatch-Pipeline/actions/runs/36151096121
- Final documentation-only checkpoint records successful local and CI checks. Consult `git log` for that commit hash.
- A local code-only archive is available at output/EntityMatch_code_checkpoint.zip, generated from committed sources and checked for ZIP integrity. It is NOT the competition submission ZIP.
- Final local native warning check completed without warnings; 17 tests passed in 4.34 seconds.


## Full-code completion update — 2026-09-25
- User requested completion of the full code. They briefly suggested Go/Rust, then explicitly withdrew that request and asked to continue with Python. Keep the Python orchestration and existing native C++ retriever; do not migrate languages.
- Full competition-data execution is still deferred. This phase completes and verifies the code, including all final delivery stages, using synthetic data.
- Added `run`: checks inputs/options/dependencies, trains, predicts, validates, generates the two-page methodology at explicit output paths, and builds the ZIP.
- Added `reproduce`: verifies dataset hashes, regenerates TSVs from the packaged model, or retrains from exact saved settings with `--retrain`, then validates and compares original output hashes.
- ZIP now includes model, exact training options, evaluation, input/output fingerprints, and SHA-256 manifest. It rejects mismatched model/report/output identities and checks ZIP integrity before replacement.
- Fixed implicit document paths and stale-PDF inclusion: run uses its output directory; standalone package includes only an explicitly selected PDF.
- Native candidate stream now includes full reference/target counts and preliminary-pair totals (EMATCH02). Reports include candidate reduction ratio and actual corpus sizes.
- Native build cache is keyed by source content and compile options, not only mtime. Threshold search includes predict-all and predict-none boundaries.
- Document generator accepts explicit Markdown/PDF paths, labels synthetic runs prominently, and enforces exactly two pages.
- The reference draft PDF was not replaced with synthetic results. Temporary synthetic PDFs are used only for testing.
- 22 local tests passed, including full run -> ZIP extraction -> packaged-model reproduction and retraining reproduction. Both reproduced TSVs matched byte-for-byte. Full-data performance remains unmeasured.
- Next code commit contains this completion update; consult git log for its exact identifier. Retain the earlier full-data continuation checklist.

- Completion implementation commit: `01217c7`, pushed to main.
- GitHub Actions run 36156192385 passed for that commit: https://github.com/praneeth132006/EntityMatch-Pipeline/actions/runs/36156192385.
- Organizer validator passed on the synthetic archive outputs with `--check-ids`: all 180 Source 1 rows covered, 360 valid target IDs checked. This is format verification, not a competition score.
- Latest local test run: 22 passed in 6.06 seconds. The updated generator's synthetic PDF was rendered and both pages visually checked; no clipping/overlap found.
- Refreshed local `output/EntityMatch_code_checkpoint.zip` from the final committed tree. The archive is source code plus the clearly labeled pending approach draft, not the final competition-data submission.

## Final coding audit — 2026-09-26
- User reiterated that the coding part must be complete. Continued the code-only scope; full competition execution remains deferred.
- Audited every CLI stage and its failure paths; no TODO/NotImplemented stubs remain.
- Fixed dataset provenance: train fingerprints required train files; predict fingerprints the actual test directory; package rejects a supplied dataset inconsistent with either. Unrelated TSVs are ignored.
- Added read-only dataset path protection (including symlinks), early header validation, strict target-list parsing, malformed ground-truth checks, and stale-PASS report removal after validation failure.
- Fixed a real repeat-test failure: pytest was discovering tests in old extracted ZIPs. Added pytest.ini and included it in submission packages.
- Expanded end-to-end fixtures to include 18 singletons and multiple true matches, plus regression coverage for changed prediction datasets, path protection, and malformed inputs.
- Latest local result: 36 passed in 6.61 seconds; native C++ warning check clean. Both saved-model and retraining reproduction still regenerate identical TSVs.
- Repaired a local pandas binary-loading failure by reinstalling the same pinned pandas 3.0.6 wheel. No package versions changed. Bootstrapped pip into the local ignored venv because uv was not starting.
- The original student_resource folder/organizer validator is no longer present at its supplied path. A bounded Downloads search did not find train_source1.tsv or validate_submission.py. Restore/provide the original resource before the deferred competition run. Did not delete or move it.
- Therefore the organizer validator was NOT rerun successfully in this audit; earlier synthetic validator passes remain recorded above.
- Added reports/code-completion.md mapping every requirement to implemented code and verification. Remaining work is real-data execution and team metadata, not unimplemented code.
