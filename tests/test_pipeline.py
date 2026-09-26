"""Contract and integration tests; all business records here are synthetic."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline", ROOT / "src" / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
HEADER = ["entity_id", "business_name", "business_address", "country"]


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("compiled")


def test_macro_metric_and_unretrieved_positives():
    # Entity 0: two true, one correct + one wrong => F0.5=.5.
    # Entity 1: correctly empty singleton => 1.
    # Entity 2: unretrieved true match => 0; entity 3: false singleton merge => 0.
    groups = np.array([0, 0, 3])
    scores = pipeline.score_entities(groups, np.array([1, 0, 0]), np.ones(3), np.array([2, 0, 1, 0]), .5)
    assert scores.tolist() == [.5, 1., 0., 0.]
    assert scores.mean() == .375


def test_official_metric_example():
    scores = pipeline.score_entities(np.array([0, 0, 0]), np.array([1, 1, 0]), np.ones(3), np.array([2]), .5)
    assert scores[0] == pytest.approx(5 / 7)


def test_threshold_never_uses_holdout_labels():
    groups = np.array([0, 0, 1])
    labels = np.array([1, 0, 1])
    probs = np.array([.8, .4, .7])
    counts = np.array([1, 1])
    mask = np.array([True, False])
    before = pipeline.choose_threshold(groups, labels, probs, counts, mask)
    labels[-1], counts[-1] = 0, 0
    assert pipeline.choose_threshold(groups, labels, probs, counts, mask) == before
    assert .4 < before[0] <= .8


def test_retrieval_country_unicode_empty_and_determinism(tmp_path, work):
    rows = [["S1-a", "Café Étoile SARL", "12 Rue des Fleurs", "France"],
            ["S1-b", "", "", "New Country"],
            ["S1-c", "Acme Corporation", "99 Main Road", "US"]]
    write_tsv(tmp_path / "test/test_source1.tsv", HEADER, rows)
    write_tsv(tmp_path / "test/test_source2.tsv", HEADER, [
        ["S2-a", "Cafe Etoile", "12 rue des fleurs", "France"],
        ["S2-wrong-country", "Cafe Etoile", "12 rue des fleurs", "US"],
        ["S2-c", "Acme Inc", "99 Main Rd", "US"]])
    write_tsv(tmp_path / "test/test_source3.tsv", HEADER, [])
    first = list(pipeline.retrieve(tmp_path, "test", work, 2))
    second = list(pipeline.retrieve(tmp_path, "test", work, 2))
    assert [r[:3] for r in first] == [r[:3] for r in second]
    assert first[0][2] == ["S2-a"]
    assert first[1][2] == []
    assert first[2][2] == ["S2-c"]
    assert first[0][3][0, 4] == 1  # Normalized name exact.
    assert first[2][3][0, 10] == 1  # Expanded address exact.
    assert all(np.isfinite(r[3]).all() for r in first)


def test_block_cap_counts_unsampled_references(tmp_path, work):
    refs = [[f"S1-{i}", "Same Company", "1 Same Avenue", "US"] for i in range(10)]
    write_tsv(tmp_path / "train/train_source1.tsv", HEADER, refs)
    write_tsv(tmp_path / "train/train_source2.tsv", HEADER, [["S2-1", "Same Company", "1 Same Avenue", "US"]])
    write_tsv(tmp_path / "train/train_source3.tsv", HEADER, [])
    rows = list(pipeline.retrieve(tmp_path, "train", work, 2, sample_mod=2, sample_keep=1, block_cap=2))
    assert rows
    assert all(not row[2] for row in rows)


def test_top_k_is_enforced(tmp_path, work):
    write_tsv(tmp_path / "test/test_source1.tsv", HEADER, [["S1-1", "Acme Labs", "1 Main Road", "US"]])
    write_tsv(tmp_path / "test/test_source2.tsv", HEADER, [[f"S2-{i}", "Acme Labs", "1 Main Road", "US"] for i in range(20)])
    write_tsv(tmp_path / "test/test_source3.tsv", HEADER, [])
    rows = list(pipeline.retrieve(tmp_path, "test", work, 3))
    assert rows[0][2] == sorted(f"S2-{i}" for i in range(20))[:3]


def validation_fixture(tmp_path, matching="S2-1", candidates="S2-1"):
    data, output = tmp_path / "dataset", tmp_path / "output"
    write_tsv(data / "test/test_source1.tsv", HEADER, [["S1-1", "Alpha", "1 Road", "France"]])
    write_tsv(data / "test/test_source2.tsv", HEADER, [["S2-1", "Alpha", "1 Road", "France"]])
    write_tsv(data / "test/test_source3.tsv", HEADER, [])
    write_tsv(output / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"], [["S1-1", matching]])
    write_tsv(output / "candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"], [["S1-1", candidates]])
    return argparse.Namespace(data_dir=str(data), output_dir=str(output))


@pytest.mark.parametrize("matching,candidates,error", [
    ("S2-1,S2-1", "S2-1", "Duplicate"),
    ("S2-1", "", "not a scored candidate"),
    ("S2-unknown", "S2-unknown", "Unknown target"),
    ("S1-1", "S1-1", "invalid target"),
])
def test_validator_rejects_invalid_outputs(tmp_path, matching, candidates, error):
    args = validation_fixture(tmp_path, matching, candidates)
    with pytest.raises(ValueError, match=error):
        pipeline.validate(args)


def test_validator_accepts_empty_lists(tmp_path):
    args = validation_fixture(tmp_path, "", "")
    pipeline.validate(args)
    assert json.loads((Path(args.output_dir) / "validation.json").read_text())["status"] == "PASS"


def test_truncated_protocol_fails():
    import io
    with pytest.raises(RuntimeError, match="Truncated"):
        pipeline.read_exact(io.BytesIO(b"abc"), 4)


def test_end_to_end_synthetic_training_and_prediction(tmp_path):
    data, work, output = tmp_path / "dataset", tmp_path / "work", tmp_path / "output"
    for split in ("train", "test"):
        refs, positives, negatives, truth = [], [], [], []
        for i in range(180):
            country = "France" if split == "test" else ("US" if i % 2 else "India")
            name = f"Synthetic venture {i} laboratories"
            address = f"{i + 100} Orchard road District{i}"
            refs.append([f"S1-{i}", name, address, country])
            matched = []
            if i % 10:  # Include real singletons in the complete workflow.
                positives.append([f"S2-{i}", name + " Inc", address.replace("road", "rd"), country])
                matched.append(f"S2-{i}")
                if i % 7 == 0:
                    negatives.append([f"S3-match-{i}", name + " Ltd", address, country])
                    matched.append(f"S3-match-{i}")
            negatives.append([f"S3-{i}", name, f"{i+9000} Distant lane Elsewhere", country])
            truth.append([f"S1-{i}", ",".join(matched)])
        write_tsv(data / split / f"{split}_source1.tsv", HEADER, refs)
        write_tsv(data / split / f"{split}_source2.tsv", HEADER, positives)
        write_tsv(data / split / f"{split}_source3.tsv", HEADER, negatives)
        if split == "train":
            write_tsv(data / split / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], truth)
    command = [sys.executable, str(ROOT / "src/pipeline.py")]
    common = ["--data-dir", str(data), "--work-dir", str(work), "--output-dir", str(output)]
    subprocess.run(command + ["run"] + common + ["--sample-mod", "1", "--sample-keep", "1", "--iterations", "30", "--report", str(tmp_path / "report.json"),
                   "--team", "Synthetic test team", "--members", "Synthetic test member", "--data-label", "synthetic"], check=True)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["holdout"]["entities"] > 0
    assert report["holdout"]["blocking_pair_recall"] == 1.0
    assert report["holdout"]["macro_f0_5"] >= .95
    stats = json.loads((output / "prediction_stats.json").read_text())
    assert stats["entities"] == 180
    assert stats["by_country"]["france"]["entities"] == 180
    assert stats["candidate_pairs"] > stats["matched_pairs"] >= 180
    assert report["holdout"]["singleton_entities"] > 0
    with (output / "matching_results.tsv").open() as stream:
        predicted = {row[0]: row[1] for row in list(csv.reader(stream, delimiter="\t"))[1:]}
    assert predicted["S1-0"] == ""
    assert {"S2-7", "S3-match-7"}.issubset(predicted["S1-7"].split(","))
    assert stats["retrieval"]["reference_count"] == 180
    assert stats["retrieval"]["target_count"] == len(positives) + len(negatives)
    assert 0 < stats["candidate_reduction_ratio"] < 1
    # The delivered code works independently of the original checkout.
    import zipfile
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(output / "EntityMatch_submission.zip") as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        import hashlib
        for name, entry in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry["sha256"]
        archive.extractall(extracted)
    code = extracted / "code/business_entity_resolution"
    script = code / "src/pipeline.py"
    generated = tmp_path / "reproduced"
    for extra in ([], ["--retrain"]):
        subprocess.run([sys.executable, str(script), "reproduce", "--data-dir", str(data),
                        "--output-dir", str(generated), *extra], cwd=code, check=True)
        assert json.loads((generated / "reproduction.json").read_text())["status"] == "PASS"
        for filename in ("matching_results.tsv", "candidate_pairs.tsv"):
            assert (generated / filename).read_bytes() == (output / filename).read_bytes()
    assert "SYNTHETIC VERIFICATION ONLY" in (output / "Documentation_template.md").read_text()
    # Valid-looking edits must still be rejected if they differ from model output.
    (output / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(f"S1-{i}\t\n" for i in range(180)))
    args = argparse.Namespace(data_dir=str(data), work_dir=str(work), output_dir=str(output),
                             destination=str(tmp_path / "bad.zip"),
                             methodology=str(output / "Documentation_template.md"), approach_pdf=None)
    with pytest.raises(ValueError, match="changed since model inference"):
        pipeline.package(args)
    # Prediction can use a newly provided test set; its provenance must follow it.
    import shutil
    alternate = tmp_path / "alternate-data"
    shutil.copytree(data, alternate)
    source = alternate / "test/test_source2.tsv"
    source.write_text(source.read_text().replace("Orchard", "Alternate", 1))
    alternate_output = tmp_path / "alternate-output"
    subprocess.run(command + ["predict", "--data-dir", str(alternate), "--work-dir", str(work),
                              "--output-dir", str(alternate_output)], check=True)
    assert json.loads((work / "input_manifest.json").read_text()) == pipeline.input_manifest(alternate)
    args.output_dir = str(alternate_output)
    with pytest.raises(ValueError, match="differs from the training/prediction manifest"):
        pipeline.package(args)




def test_quoted_tsv_fields(tmp_path, work):
    row = ["S1-1", 'Alpha "Research"\tLabs', "7 Long\nRoad", "France"]
    write_tsv(tmp_path / "test/test_source1.tsv", HEADER, [row])
    write_tsv(tmp_path / "test/test_source2.tsv", HEADER, [["S2-1", *row[1:]]])
    write_tsv(tmp_path / "test/test_source3.tsv", HEADER, [])
    result = list(pipeline.retrieve(tmp_path, "test", work, 2))
    assert result[0][2] == ["S2-1"]


def test_bad_source_header_fails(tmp_path, work):
    write_tsv(tmp_path / "test/test_source1.tsv", ["wrong"], [])
    with pytest.raises(RuntimeError, match="Truncated"):
        list(pipeline.retrieve(tmp_path, "test", work, 2))


def test_validator_rejects_duplicate_references(tmp_path):
    args = validation_fixture(tmp_path)
    path = Path(args.data_dir) / "test/test_source1.tsv"
    with path.open("a") as stream:
        stream.write("S1-1\tAlpha\t1 Road\tFrance\n")
    with pytest.raises(ValueError, match="Duplicate Source 1"):
        pipeline.validate(args)


def test_packaging_rejects_draft(tmp_path, monkeypatch):
    args = validation_fixture(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "Documentation_template.md").write_text("NOT YET RUN\n")
    monkeypatch.setattr(pipeline, "ROOT", project)
    args.destination = str(tmp_path / "submission.zip")
    args.work_dir = str(tmp_path / "work")
    args.methodology = None
    args.approach_pdf = None
    with pytest.raises(ValueError, match="Finalize the methodology"):
        pipeline.package(args)
    # A previously valid file must not authorize an invalid replacement.
    (Path(args.output_dir) / "matching_results.tsv").write_text("wrong\n")
    with pytest.raises(ValueError, match="Invalid matching header"):
        pipeline.package(args)


def test_threshold_can_predict_none():
    threshold, score = pipeline.choose_threshold(np.array([0]), np.array([0]), np.array([1.0]),
                                                 np.array([0]), np.array([True]))
    assert threshold > 1
    assert score == 1


@pytest.mark.parametrize("key,value", [("top_k", 0), ("iterations", -1), ("sample_keep", 1001)])
def test_invalid_training_settings_fail_before_retrieval(key, value):
    args = argparse.Namespace(sample_mod=1000, sample_keep=20, top_k=12, block_cap=64,
                              iterations=600, threads=4, data_label="synthetic")
    setattr(args, key, value)
    with pytest.raises(ValueError):
        pipeline.check_training_options(args)


def test_missing_input_fails_early(tmp_path):
    with pytest.raises(ValueError, match="Missing or empty"):
        pipeline.check_inputs(tmp_path)


@pytest.mark.parametrize("field", ["work_dir", "output_dir", "report", "destination"])
def test_cannot_write_into_original_dataset(tmp_path, field):
    data = tmp_path / "data"
    data.mkdir()
    args = argparse.Namespace(data_dir=str(data))
    setattr(args, field, str(data / "accidental-output"))
    with pytest.raises(ValueError, match="outside the input dataset"):
        pipeline.check_write_paths(args)


def test_symlink_cannot_bypass_dataset_protection(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(data, target_is_directory=True)
    args = argparse.Namespace(data_dir=str(data), output_dir=str(alias / "output"))
    with pytest.raises(ValueError, match="outside the input dataset"):
        pipeline.check_write_paths(args)


@pytest.mark.parametrize("text", ["S2-", "S2-1,", "S2-1,S2-1", "S2- 1", 'S3-a"', "S3-a\x00"])
def test_strict_target_id_lists(text):
    with pytest.raises(ValueError, match="invalid target"):
        pipeline.parse_target_ids(text)


def test_empty_submission_has_readable_error(tmp_path):
    args = validation_fixture(tmp_path)
    (Path(args.output_dir) / "matching_results.tsv").write_text("")
    with pytest.raises(ValueError, match="Invalid matching header"):
        pipeline.validate(args)


def test_invalid_source_header_clears_stale_pass(tmp_path):
    args = validation_fixture(tmp_path)
    pipeline.validate(args)
    report = Path(args.output_dir) / "validation.json"
    assert report.exists()
    (Path(args.data_dir) / "test/test_source2.tsv").write_text("wrong\n")
    with pytest.raises(ValueError, match="Unexpected Source 2 header"):
        pipeline.validate(args)
    assert not report.exists()


def test_manifest_tracks_only_required_files_and_selected_split(tmp_path):
    args = validation_fixture(tmp_path)
    data = Path(args.data_dir)
    before = pipeline.input_manifest(data, ("test",))
    assert len(before) == 3
    (data / "test/unrelated.tsv").write_text("notes\n")
    assert pipeline.input_manifest(data, ("test",)) == before
    path = data / "test/test_source1.tsv"
    path.write_text(path.read_text().replace("Alpha", "Beta"))
    assert pipeline.input_manifest(data, ("test",)) != before
