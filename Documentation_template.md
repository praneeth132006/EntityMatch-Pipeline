# ML Challenge 2026: Business Entity Resolution

**Status:** DRAFT - FULL DATA EVALUATION NOT YET RUN

Project/team: EntityMatch (registration details pending) | Members: Not yet supplied | Date: 2026-09-25

## 1. Executive summary

EntityMatch uses country-aware indexed blocking followed by a compact CatBoost classifier. It resolves every Source 1 reference against Source 2 and Source 3 while limiting the number of pairs sent to the classifier. All identity evidence comes from the provided files; there are no external lookups, geocoding calls, or pretrained model downloads.

## 2. Methodology

The supplied files contain 2,206,821 training references and 10,320,219 training targets, plus 1,732,544 test references and 9,969,589 test targets. Names and addresses contain abbreviations, reordered words, spelling noise, and missing fields. The implementation folds common Latin accents, removes common legal suffixes, and normalizes common address abbreviations. Country is an open string label; test-only France is retained. IDs are used for joining and deterministic splitting, never as classifier features.

## 3. Candidate generation

A sorted 64-bit posting index covers normalized names, sorted name tokens, full addresses, and name-token prefixes combined with address-word prefixes or numbers. Keys appearing in more than 64 reference records are excluded before training sampling. Targets are streamed through the index, with an explicit country equality check. Pairs with name Dice below 0.12 or weighted name/address similarity below 0.27 are removed. The final deterministic blocking stage retains at most 12 candidates per reference, ranked by 0.65 name Dice plus 0.35 address Dice. These defaults require full-data validation. Every retained pair is sent to the classifier and exported in candidate_pairs.tsv.

## 4. Matching model

The classifier uses 22 features: character-bigram Dice, token Jaccard and containment, token-sorted name similarity, exact agreement, length ratios, first-token/number agreement, numeric overlap/conflict, missingness flags, and name/address interactions. CatBoost uses depth 6, learning rate 0.06, a fixed seed, and up to 600 trees with early stopping. It is trained from scratch with an Apache-2.0 implementation; the project provides its original code and trained models under MIT. The model is far below the eight-billion-parameter limit.

## 5. Evaluation and error analysis

Training defaults to a deterministic 2% reference sample, with the complete target corpus available for retrieval. A separate hash partitions references into 60% fitting, 20% tuning, and 20% holdout groups. The tuning set selects tree count and a probability threshold maximizing entity-macro F0.5. The holdout remains unused for fitting or threshold selection. Unretrieved true links count as false negatives, and correctly empty singletons score 1. The final model is refit on all sampled training labels after evaluation.

## 6. Results and limitations

NOT YET RUN: no competition holdout score, candidate recall, runtime, or leaderboard result is available. Full-data execution was deferred at the user's request after a low-disk-space warning. Synthetic tests establish functional behavior only. Likely failure modes include true links lost through heavy corruption, country disagreements, frequent-key exclusion or top-k truncation, and false merges for shared names/addresses. France has no labeled training examples, so France accuracy cannot be inferred from a US/India holdout.

## 7. Reproduction and submission

src/pipeline.py provides train, predict, validate, and package commands; src/retrieve.cpp supplies native retrieval and features. The README gives exact commands and dependencies are pinned. Prediction writes one ordered row per reference to both TSVs. Validation checks complete coverage, unique IDs, source prefixes, real target-ID membership, and final-match containment in the scored candidate set. Packaging revalidates outputs and creates the required output/ and code/business_entity_resolution/ layout with this methodology. The raw dataset is supplied separately.

## 8. Conclusion

The implementation prioritizes compact candidate sets and precision-sensitive classification while preserving empty results and unseen-country coverage. Synthetic tests exercise the code and file contracts. Before a competition submission, run the supplied dataset, inspect measured blocking recall and matching errors, finalize team details, and regenerate the methodology and ZIP. Full-data memory, runtime, and France generalization remain to be assessed.
