from pathlib import Path
import json
import numpy as np

from src.evaluation import judge_agreement


def test_create_item_key_stable():
    item = {"input": "a", "gold": "g", "prediction": "p"}
    key1 = judge_agreement.create_item_key(item)
    key2 = judge_agreement.create_item_key(item)
    assert key1 == key2
    assert len(key1) == 16


def test_align_and_filter_common_items():
    judge_results = {
        "judge1": {
            "file1": [{"input": "x", "gold": "g", "prediction": "p", "correct": "yes"}]
        },
        "judge2": {
            "file1": [{"input": "x", "gold": "g", "prediction": "p", "correct": "no"}]
        },
    }
    aligned, details = judge_agreement.align_judgments(judge_results)
    assert len(aligned) == 1
    key = next(iter(aligned.keys()))
    assert set(aligned[key].keys()) == {"judge1", "judge2"}
    assert details[key]["source_file"] == "file1"

    common = judge_agreement.filter_common_items(aligned, min_judges=2)
    assert len(common) == 1


def test_compute_cohen_kappa_basic():
    labels1 = ["yes", "no", "yes", "no"]
    labels2 = ["yes", "no", "no", "no"]
    kappa = judge_agreement.compute_cohen_kappa(labels1, labels2)
    assert isinstance(kappa, float)


def test_compute_fleiss_kappa_edge():
    ratings = np.array([[2, 0], [0, 2]])
    kappa = judge_agreement.compute_fleiss_kappa(ratings)
    assert round(kappa, 4) == 1.0


def test_load_evaluated_files(tmp_path):
    eval_dir = tmp_path / "eval"
    judge_dir = eval_dir / "judge1"
    judge_dir.mkdir(parents=True)

    payload = [{"input": "x", "gold": "g", "prediction": "p", "correct": "yes"}]
    file_path = judge_dir / "model__judge-judge1.json"
    file_path.write_text(json.dumps(payload))

    results = judge_agreement.load_evaluated_files(eval_dir)
    assert "judge1" in results
    assert "model" in results["judge1"]
    assert results["judge1"]["model"][0]["input"] == "x"


def test_normalize_human_annotations_and_merge():
    items = [
        {"input": "x", "gold": "g", "prediction": "p", "correct": True},
        {"input": "y", "gold": "g", "prediction": "p", "correct": "no"},
    ]
    normalized = judge_agreement.normalize_human_annotations(items, "source")
    assert "human" in normalized
    assert normalized["human"]["source"][0]["correct"] == "yes"

    base = {
        "judge1": {
            "source": [{"input": "x", "gold": "g", "prediction": "p", "correct": "yes"}]
        }
    }
    merged = judge_agreement.merge_judge_results(base, normalized)
    assert "judge1" in merged and "human" in merged


def test_pairwise_and_fleiss_kappa():
    aligned = {
        "item1": {"j1": "yes", "j2": "no"},
        "item2": {"j1": "yes", "j2": "yes"},
    }
    pairwise = judge_agreement.compute_pairwise_kappa(aligned, ["j1", "j2"])
    assert ("j1", "j2") in pairwise

    fleiss = judge_agreement.compute_overall_fleiss(aligned, ["j1", "j2"])
    assert fleiss["n_items"] == 2

    alpha = judge_agreement.compute_krippendorff_alpha(aligned, ["j1", "j2"])
    assert "alpha" in alpha


def test_generate_agreement_report_basic(tmp_path):
    eval_dir = tmp_path / "evaluated"
    judge1 = eval_dir / "judge1"
    judge2 = eval_dir / "judge2"
    judge1.mkdir(parents=True)
    judge2.mkdir(parents=True)

    payload = [{"input": "x", "gold": "g", "prediction": "p", "correct": "yes"}]
    (judge1 / "model__judge-judge1.json").write_text(json.dumps(payload))
    (judge2 / "model__judge-judge2.json").write_text(json.dumps(payload))

    report = judge_agreement.generate_agreement_report(eval_dir)
    assert report["n_judges"] == 2
    assert report["items_with_multiple_judges"] == 1
