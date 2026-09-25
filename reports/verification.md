# Code checkpoint verification — 2026-09-25

- Environment: macOS Apple Silicon, Python 3.12.13, C++17 Apple clang 21.
- Command: `.venv/bin/python -m pytest -q`.
- Result: **17 passed** on synthetic records only.
- Covered: native blocking and quoted TSV parsing; country handling; candidate caps;
  exact macro F0.5 and singletons; threshold/holdout isolation; full synthetic
  training/prediction/validation; invalid submission rejection; archive structure;
  refusal to package unfinished methodology or trust stale validation.
- Additional check: native source compiled with `-Wall -Wextra -Wpedantic` without warnings.
- GitHub Actions: Ubuntu workflow passed for code commit `cdb3d02`; run
  https://github.com/praneeth132006/EntityMatch-Pipeline/actions/runs/36151096121.
- Approach draft: two PDF pages; both visually reviewed.

Full competition execution is deferred at the user's request. There is no measured
competition score, test output, production runtime, or final submission ZIP yet.
The committed PDF is an approach draft, not a submission-ready report.


## Full workflow completion checks

- **22 local tests passed** after adding run/reproduce and artifact provenance checks.
- Integration test executes every stage on synthetic records, checks every archive
  manifest checksum, extracts the ZIP, and regenerates both TSVs from the extracted
  code using the saved model and again with `--retrain`.
- Both reproduction modes match the original TSVs byte-for-byte.
- New checks cover invalid training options, missing inputs, the predict-none threshold,
  actual retrieval counts/reduction ratio, and rejection of modified inference outputs.
- C++ warning check remains clean. All PDFs produced by these tests are explicitly
  marked synthetic and are not committed as competition results.

- GitHub CI passed for full-workflow commit `01217c7`:
  https://github.com/praneeth132006/EntityMatch-Pipeline/actions/runs/36156192385.
- Organizer-provided validator passed with `--check-ids` on the synthetic archive
  outputs: 180 required references, 360 valid target IDs. No competition data was run.
- Both pages of the updated document generator's synthetic output were visually reviewed.
