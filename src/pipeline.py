#!/usr/bin/env python3
"""Offline business entity matching: retrieve, train, evaluate, predict, package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np

FEATURES = [
    "name_bigram_dice", "name_token_jaccard", "name_token_containment",
    "name_sorted_bigram_dice", "name_exact", "name_length_ratio", "name_first_token",
    "address_bigram_dice", "address_token_jaccard", "address_token_containment",
    "address_exact", "address_length_ratio", "number_jaccard", "first_number_equal",
    "postal_like_number_jaccard", "postal_like_number_conflict", "name_missing",
    "address_missing", "number_missing", "name_address_product",
    "name_address_minimum", "retrieval_score",
]
ROOT = Path(__file__).resolve().parents[1]


def stable_hash(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def build(work: Path) -> Path:
    if sys.byteorder != "little":
        raise RuntimeError("Candidate protocol requires a little-endian machine")
    work.mkdir(parents=True, exist_ok=True)
    binary = work / "retrieve"
    source = ROOT / "src" / "retrieve.cpp"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        compiler = os.environ.get("CXX", "c++")
        subprocess.run([compiler, "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)
    return binary


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError(f"Truncated retrieval stream: expected {size}, got {len(data)} bytes")
    return data


def read_u32(stream):
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_string(stream):
    size = read_u32(stream)
    if size > 1_000_000:
        raise RuntimeError("Corrupt candidate string length")
    return read_exact(stream, size).decode("utf-8")


def retrieve(data: Path, split: str, work: Path, top_k: int, sample_mod=1, sample_keep=1, block_cap=64):
    command = [str(build(work)), str(data / split), split, str(top_k),
               str(sample_mod), str(sample_keep), str(block_cap)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        stream = process.stdout
        if read_exact(stream, 8) != b"EMATCH01":
            raise RuntimeError("Invalid retrieval protocol")
        rows, n_features = read_u32(stream), read_u32(stream)
        if n_features != len(FEATURES):
            raise RuntimeError("Feature schema mismatch")
        for _ in range(rows):
            entity, country, count = read_string(stream), read_string(stream), read_u32(stream)
            if count > top_k:
                raise RuntimeError("Candidate bound exceeded")
            ids, features = [], np.empty((count, n_features), dtype=np.float32)
            for i in range(count):
                ids.append(read_string(stream))
                features[i] = np.frombuffer(read_exact(stream, 4 * n_features), dtype="<f4")
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate candidates for {entity}")
            yield entity, country, ids, features
        if stream.read(1):
            raise RuntimeError("Unexpected trailing retrieval data")
        if process.wait() != 0:
            raise RuntimeError("Retrieval subprocess failed")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdout.close()


def score_entities(groups, labels, probabilities, true_counts, threshold):
    """Exact entity-macro F0.5, including unretrieved positives and singletons."""
    selected = probabilities >= threshold
    n = len(true_counts)
    predicted = np.bincount(groups[selected], minlength=n)
    correct = np.bincount(groups[selected & (labels == 1)], minlength=n)
    denominator = predicted + 0.25 * true_counts
    scores = np.divide(1.25 * correct, denominator, out=np.zeros(n), where=denominator != 0)
    scores[(true_counts == 0) & (predicted == 0)] = 1.0
    return scores


def choose_threshold(groups, labels, probabilities, true_counts, mask):
    if not np.any(mask):
        raise ValueError("Threshold split has no entities")
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.10, 0.995, 180):
        score = float(score_entities(groups, labels, probabilities, true_counts, threshold)[mask].mean())
        if score >= best[0]:  # Conservative tie-break: prefer higher threshold.
            best = (score, float(threshold))
    return best[1], best[0]


def input_manifest(data):
    manifest = {}
    for split in ("train", "test"):
        for path in sorted((data / split).glob("*.tsv")):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            manifest[f"{split}/{path.name}"] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return manifest


def train(args):
    from catboost import CatBoostClassifier
    start = time.monotonic()
    work, data = Path(args.work_dir), Path(args.data_dir)
    refs, countries, targets, blocks, group_blocks = [], [], [], [], []
    for entity, country, ids, features in retrieve(data, "train", work, args.top_k,
                                                  args.sample_mod, args.sample_keep, args.block_cap):
        group_blocks.append(np.full(len(ids), len(refs), dtype=np.int32))
        refs.append(entity)
        countries.append(country)
        targets.extend(ids)
        blocks.append(features)
    if not refs:
        raise ValueError("No sampled reference entities")
    ref_lookup = {entity: i for i, entity in enumerate(refs)}
    truth = [None] * len(refs)
    with (data / "train" / "train_ground_truth.tsv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected ground truth header")
        for row in reader:
            i = ref_lookup.get(row["source1_entity_id"])
            if i is not None:
                if truth[i] is not None:
                    raise ValueError("Duplicate ground truth reference ID")
                truth[i] = set(filter(None, row["matched_entity_ids"].split(",")))
    if any(t is None for t in truth):
        raise ValueError("Ground truth does not cover sampled reference entities")
    x = np.concatenate(blocks)
    groups = np.concatenate(group_blocks)
    y = np.array([int(target in truth[group]) for target, group in zip(targets, groups)], dtype=np.uint8)
    counts = np.array([len(t) for t in truth], dtype=np.int32)
    # Split at entity level with a salt independent of the sampling hash.
    fold = np.array([stable_hash("evaluation:" + entity) % 10 for entity in refs])
    fitting, tuning, holdout = fold < 6, (fold >= 6) & (fold < 8), fold >= 8
    fit_pairs, tune_pairs = fitting[groups], tuning[groups]
    for name, mask in (("fitting", fit_pairs), ("tuning", tune_pairs)):
        if len(np.unique(y[mask])) < 2:
            raise ValueError(f"{name} split needs positive and negative candidate pairs; increase sample size")
    if not np.any(holdout):
        raise ValueError("No holdout entities; increase sample size")
    parameters = dict(iterations=args.iterations, depth=6, learning_rate=0.06,
                      loss_function="Logloss", random_seed=20260925, thread_count=args.threads,
                      l2_leaf_reg=5, allow_writing_files=False, verbose=False)
    model = CatBoostClassifier(**parameters)
    model.fit(x[fit_pairs], y[fit_pairs], eval_set=(x[tune_pairs], y[tune_pairs]),
              early_stopping_rounds=50, verbose=False)
    probabilities = model.predict_proba(x)[:, 1]
    threshold, tuning_score = choose_threshold(groups, y, probabilities, counts, tuning)
    scores = score_entities(groups, y, probabilities, counts, threshold)
    retrieved = np.bincount(groups, weights=y, minlength=len(refs))
    def describe(mask):
        pair_mask = mask[groups]
        selected = (probabilities >= threshold) & pair_mask
        tp = int(y[selected].sum())
        pred = int(selected.sum())
        total = int(counts[mask].sum())
        return {"entities": int(mask.sum()), "macro_f0_5": float(scores[mask].mean()),
                "pair_precision": tp / pred if pred else 0.0,
                "pair_recall": tp / total if total else 1.0,
                "blocking_pair_recall": float(retrieved[mask].sum() / total) if total else 1.0,
                "average_candidates": float(pair_mask.sum() / mask.sum()),
                "singleton_entities": int(((counts == 0) & mask).sum())}
    report = {
        "status": "measured held-out results, not leaderboard results",
        "sampled_entities": len(refs), "candidate_pairs": len(y), "positive_candidates": int(y.sum()),
        "threshold": threshold, "tuning_macro_f0_5": tuning_score,
        "holdout": describe(holdout),
        "holdout_by_country": {country: describe(holdout & (np.array(countries) == country))
                               for country in sorted(set(countries))
                               if np.any(holdout & (np.array(countries) == country))},
        "top_k": args.top_k, "block_cap": args.block_cap,
        "sample_mod": args.sample_mod, "sample_keep": args.sample_keep,
        "features": FEATURES, "iterations": model.tree_count_,
        "limitations": ["France has no labeled training examples; France accuracy cannot be estimated here.",
                        "Blocking may miss heavily corrupted names or addresses.",
                        "Final model is refit on the sampled training entities after holdout measurement."],
    }
    errors = []
    for i in np.flatnonzero(holdout & (scores < 1))[:40]:
        pair_indices = np.flatnonzero(groups == i)
        predicted = {targets[j] for j in pair_indices if probabilities[j] >= threshold}
        candidates = {targets[j] for j in pair_indices}
        errors.append({"source1_entity_id": refs[i], "country": countries[i],
                       "false_positive_ids": sorted(predicted - truth[i]),
                       "false_negative_ids": sorted(truth[i] - predicted),
                       "unretrieved_true_ids": sorted(truth[i] - candidates)})
    report["holdout_error_examples"] = errors
    importance = model.get_feature_importance()
    report["feature_importance"] = dict(zip(FEATURES, map(float, importance)))
    # Freeze evaluation results before fitting the final model on all sampled labels.
    parameters["iterations"] = max(1, model.tree_count_)
    final_model = CatBoostClassifier(**parameters)
    final_model.fit(x, y)
    final_model.save_model(str(work / "model.cbm"))
    report["elapsed_seconds"] = time.monotonic() - start
    dump_json(work / "model_config.json", {"threshold": threshold, "top_k": args.top_k,
              "block_cap": args.block_cap, "features": FEATURES, "parameters": parameters,
              "retriever_sha256": hashlib.sha256((ROOT / "src/retrieve.cpp").read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256((work / "model.cbm").read_bytes()).hexdigest()})
    dump_json(Path(args.report), report)
    dump_json(work / "input_manifest.json", input_manifest(data))
    print(json.dumps({"holdout": report["holdout"], "threshold": threshold}, indent=2))


def predict(args):
    from catboost import CatBoostClassifier
    work, output = Path(args.work_dir), Path(args.output_dir)
    config = json.loads((work / "model_config.json").read_text())
    if config["features"] != FEATURES:
        raise ValueError("Model feature schema mismatch")
    if config["retriever_sha256"] != hashlib.sha256((ROOT / "src/retrieve.cpp").read_bytes()).hexdigest():
        raise ValueError("Retriever changed since training; retrain before prediction")
    if config["model_sha256"] != hashlib.sha256((work / "model.cbm").read_bytes()).hexdigest():
        raise ValueError("Model file does not match its saved configuration")
    model = CatBoostClassifier()
    model.load_model(str(work / "model.cbm"))
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "matching_results.tsv", output / "candidate_pairs.tsv"]
    temporary = [path.with_suffix(".tsv.tmp") for path in paths]
    stats = {"entities": 0, "candidate_pairs": 0, "matched_pairs": 0, "by_country": {}}
    with temporary[0].open("w", encoding="utf-8", newline="") as match, temporary[1].open("w", encoding="utf-8", newline="") as candidate:
        match.write("source1_entity_id\tmatched_entity_ids\n")
        candidate.write("source1_entity_id\tcandidate_entity_ids\n")
        batch = []
        def flush():
            if not batch:
                return
            total = sum(len(row[2]) for row in batch)
            # Exactly these candidate pairs, with no additional pre-model pruning, are scored.
            p = model.predict_proba(np.concatenate([row[3] for row in batch]), thread_count=args.threads)[:, 1] if total else np.empty(0)
            offset = 0
            for entity, country, ids, _ in batch:
                chosen = [target for target, prob in zip(ids, p[offset:offset + len(ids)]) if prob >= config["threshold"]]
                offset += len(ids)
                candidate.write(entity + "\t" + ",".join(ids) + "\n")
                match.write(entity + "\t" + ",".join(chosen) + "\n")
                stats["entities"] += 1
                stats["candidate_pairs"] += len(ids)
                stats["matched_pairs"] += len(chosen)
                stat = stats["by_country"].setdefault(country, {"entities": 0, "candidates": 0, "matches": 0})
                stat["entities"] += 1
                stat["candidates"] += len(ids)
                stat["matches"] += len(chosen)
            batch.clear()
        for row in retrieve(Path(args.data_dir), "test", work, config["top_k"], block_cap=config["block_cap"]):
            batch.append(row)
            if len(batch) >= 2048:
                flush()
        flush()
    for temp, path in zip(temporary, paths):
        temp.replace(path)
    stats["average_candidates"] = stats["candidate_pairs"] / max(1, stats["entities"])
    dump_json(output / "prediction_stats.json", stats)
    print(json.dumps(stats, indent=2))


def validate(args):
    """Streaming validation plus exact target-ID membership using only referenced IDs."""
    data, output = Path(args.data_dir), Path(args.output_dir)
    requested = set()
    seen = set()
    rows = 0
    with (data / "test" / "test_source1.tsv").open(encoding="utf-8", newline="") as source, (output / "matching_results.tsv").open(encoding="utf-8", newline="") as mf, (output / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as cf:
        sr, mr, cr = (csv.reader(s, delimiter="\t") for s in (source, mf, cf))
        if next(sr) != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError("Unexpected Source 1 header")
        if next(mr) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Invalid matching header")
        if next(cr) != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Invalid candidate header")
        for source_row in sr:
            if len(source_row) != 4 or not source_row[0].startswith("S1-"):
                raise ValueError("Malformed Source 1 row")
            if source_row[0] in seen:
                raise ValueError("Duplicate Source 1 ID in input")
            seen.add(source_row[0])
            matching, candidates = next(mr, None), next(cr, None)
            if matching is None or candidates is None:
                raise ValueError("Missing Source 1 rows")
            if len(matching) != 2 or len(candidates) != 2:
                raise ValueError("Output must have exactly two TSV columns")
            if matching[0] != source_row[0] or candidates[0] != source_row[0]:
                raise ValueError("Source IDs must match input order exactly")
            lists = [r[1].split(",") if r[1] else [] for r in (matching, candidates)]
            for ids in lists:
                if len(ids) != len(set(ids)) or any(not x.startswith(("S2-", "S3-")) for x in ids):
                    raise ValueError("Duplicate or invalid target IDs")
            if not set(lists[0]).issubset(lists[1]):
                raise ValueError("A final match was not a scored candidate")
            requested.update(lists[1])
            rows += 1
        if next(mr, None) is not None or next(cr, None) is not None:
            raise ValueError("Extra output rows")
    referenced = len(requested)
    for source in (2, 3):
        with (data / "test" / f"test_source{source}.tsv").open(encoding="utf-8") as stream:
            next(stream)
            for row in csv.reader(stream, delimiter="\t"):
                requested.discard(row[0])
    if requested:
        raise ValueError(f"Unknown target IDs: {sorted(requested)[:5]}")
    result = {"status": "PASS", "source1_rows": rows, "distinct_referenced_targets": referenced,
              "checks": ["headers", "complete ordered coverage", "no duplicate list IDs", "S2/S3 only",
                         "matches subset of scored candidates", "all referenced target IDs exist"]}
    dump_json(output / "validation.json", result)
    print(json.dumps(result, indent=2))


def package(args):
    output = Path(args.output_dir)
    validation = output / "validation.json"
    if not validation.exists() or json.loads(validation.read_text())["status"] != "PASS":
        raise ValueError("Run validate successfully before packaging")
    # Always revalidate current files; an old PASS report must not authorize changed outputs.
    validate(args)
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    methodology = ROOT / "Documentation_template.md"
    if not methodology.exists() or "NOT YET RUN" in methodology.read_text():
        raise ValueError("Finalize the methodology with measured results before packaging")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            archive.write(output / name, "output/" + name)
        for path in sorted((ROOT / "src").glob("*")):
            if path.is_file() and path.suffix in {".py", ".cpp"}:
                archive.write(path, "code/business_entity_resolution/src/" + path.name)
        for name in ("README.md", "requirements.txt", "requirements-lock.txt", "requirements-docs.txt", "LICENSE", "memory.md"):
            archive.write(ROOT / name, "code/business_entity_resolution/" + name)
        archive.write(methodology, "Documentation_template.md")
        pdf = ROOT / "output" / "pdf" / "Approach.pdf"
        if pdf.exists():
            archive.write(pdf, "Approach.pdf")
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            archive.write(path, "code/business_entity_resolution/tests/" + path.name)
        archive.write(ROOT / "requirements-dev.txt", "code/business_entity_resolution/requirements-dev.txt")
    print(f"Created {destination} ({destination.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "predict", "validate", "package"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", required=True, help="Directory containing train/ and test/")
        p.add_argument("--work-dir", default="artifacts")
        p.add_argument("--output-dir", default="output")
        p.add_argument("--threads", type=int, default=4)
        p.set_defaults(func=globals()[command])
        if command == "train":
            p.add_argument("--sample-mod", type=int, default=1000)
            p.add_argument("--sample-keep", type=int, default=20)
            p.add_argument("--top-k", type=int, default=12)
            p.add_argument("--block-cap", type=int, default=64)
            p.add_argument("--iterations", type=int, default=600)
            p.add_argument("--report", default="reports/evaluation.json")
        if command == "package":
            p.add_argument("--destination", default="output/EntityMatch_submission.zip")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
