# Code checkpoint verification — 2026-09-25

- Environment: macOS Apple Silicon, Python 3.12.13, C++17 Apple clang 21.
- Command: `.venv/bin/python -m pytest -q`.
- Result: **17 passed** on synthetic records only.
- Covered: native blocking and quoted TSV parsing; country handling; candidate caps;
  exact macro F0.5 and singletons; threshold/holdout isolation; full synthetic
  training/prediction/validation; invalid submission rejection; archive structure;
  refusal to package unfinished methodology or trust stale validation.
- Additional check: native source compiled with `-Wall -Wextra -Wpedantic`.
- Approach draft: two PDF pages; both visually reviewed.

Full competition execution is deferred at the user's request. There is no measured
competition score, test output, production runtime, or final submission ZIP yet.
The committed PDF is an approach draft, not a submission-ready report.
