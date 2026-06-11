import random
from collections import Counter

import pytest

from src import data_utils


def test_save_and_load(tmp_path):
    data = [{"input": "Agent A can guarantee p", "outputs": [{"formula": "<<A>>F p"}]}]
    p = tmp_path / "data.json"
    data_utils.save_data(data, str(p))
    loaded = data_utils.load_data(str(p))
    assert isinstance(loaded, list)
    assert loaded[0]["input"] == data[0]["input"]
    assert loaded[0]["outputs"] == ["<<A>>F p"]


def test_load_data_preserves_multiple_correct_outputs(tmp_path):
    data = [
        {
            "input": "Every server can guarantee that next p holds.",
            "outputs": [
                {"formula": "<<Server1,Server2>>X p"},
                {"formula": "<<Server1>>X p_1 && <<Server2>>X p_2"},
            ],
        }
    ]
    p = tmp_path / "data.json"
    data_utils.save_data(data, str(p))

    loaded = data_utils.load_data(str(p))

    assert loaded[0]["outputs"] == [
        "<<Server1,Server2>>X p",
        "<<Server1>>X p_1 && <<Server2>>X p_2",
    ]


def test_augment_data_preserves_and_expands():
    data = [
        {
            "input": "The system can guarantee that eventually p",
            "outputs": [{"formula": "<<S>>F p"}],
        }
    ]
    random.seed(123)
    augmented = data_utils.augment_data(data, augment_factor=3)
    assert len(augmented) == 3
    # ensure original present
    assert any(item["input"] == data[0]["input"] for item in augmented)
    # ensure outputs preserved
    assert all("outputs" in item for item in augmented)


def test_augment_data_seed_is_independent_of_global_random_state():
    data = [
        {
            "input": "The system can guarantee that eventually p",
            "outputs": [{"formula": "<<S>>F p"}],
        }
    ]

    random.seed(1)
    random.random()
    first = data_utils.augment_data(data, augment_factor=4, seed=123)

    random.seed(999)
    for _ in range(10):
        random.random()
    second = data_utils.augment_data(data, augment_factor=4, seed=123)

    assert [item["input"] for item in first] == [item["input"] for item in second]


def test_augment_data_no_change_when_factor_one():
    data = [
        {
            "input": "The system will always stay safe",
            "outputs": [{"formula": "<<S>>G safe"}],
        }
    ]
    augmented = data_utils.augment_data(data, augment_factor=1)
    assert augmented == [
        {
            "input": "The system will always stay safe",
            "outputs": ["<<S>>G safe"],
        }
    ]


def test_augment_data_preserves_multiple_correct_outputs():
    data = [
        {
            "input": "Every server can guarantee that next p holds.",
            "outputs": [
                {"formula": "<<Server1,Server2>>X p"},
                {"formula": "<<Server1>>X p_1 && <<Server2>>X p_2"},
            ],
        }
    ]

    augmented = data_utils.augment_data(data, augment_factor=2)

    assert len(augmented) == 2
    assert all(
        item["outputs"]
        == ["<<Server1,Server2>>X p", "<<Server1>>X p_1 && <<Server2>>X p_2"]
        for item in augmented
    )


def test_split_data_counts_without_stratification():
    data = []
    for i in range(10):
        data.append({"input": f"item a {i}", "outputs": [{"formula": "<<A>>F p"}]})
        data.append({"input": f"item b {i}", "outputs": [{"formula": "<<A>>G p"}]})

    train, val, test = data_utils.split_data(
        data, train_size=0.7, val_size=0.1, test_size=0.2, seed=123
    )

    assert len(train) == 14
    assert len(val) == 2
    assert len(test) == 4
    assert len(train) + len(val) + len(test) == len(data)
    assert sorted(item["input"] for item in train + val + test) == sorted(
        item["input"] for item in data
    )


def test_split_data_rejects_invalid_ratios():
    data = [{"input": "A", "outputs": [{"formula": "<<A>>F p"}]}]
    with pytest.raises(ValueError, match="sum to 1.0"):
        data_utils.split_data(data, train_size=0.7, val_size=0.2, test_size=0.2)


def _mixed_stratum_data():
    data = []
    for i in range(20):
        data.append({"input": f"single {i}", "outputs": [{"formula": "<<A>>F p"}]})
    for i in range(10):
        data.append(
            {
                "input": f"multi {i}",
                "outputs": [{"formula": "<<A>>F p"}, {"formula": "<<B>>G q"}],
            }
        )
    return data


def test_default_stratum_distinguishes_single_and_multi():
    single = {"input": "x", "outputs": [{"formula": "<<A>>F p"}]}
    multi = {"input": "y", "outputs": [{"formula": "a"}, {"formula": "b"}]}
    assert data_utils.default_stratum(single) == "single"
    assert data_utils.default_stratum(multi) == "multi"


def test_split_data_stratified_preserves_proportions_and_is_deterministic():
    data = _mixed_stratum_data()

    train, val, test = data_utils.split_data(
        data,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        seed=42,
        stratify_key=data_utils.default_stratum,
    )

    # No leakage and full coverage.
    assert len(train) + len(val) + len(test) == len(data)
    assert sorted(i["input"] for i in train + val + test) == sorted(
        i["input"] for i in data
    )

    # Every split carries both strata (proportional representation).
    for split in (train, val, test):
        labels = {data_utils.default_stratum(i) for i in split}
        assert labels == {"single", "multi"}

    # Deterministic for a fixed seed.
    again = data_utils.split_data(
        data,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        seed=42,
        stratify_key=data_utils.default_stratum,
    )
    assert [i["input"] for i in again[2]] == [i["input"] for i in test]


def test_split_data_test_set_changes_with_seed():
    data = _mixed_stratum_data()
    _, _, test_a = data_utils.split_data(data, seed=42)
    _, _, test_b = data_utils.split_data(data, seed=43)
    assert {i["input"] for i in test_a} != {i["input"] for i in test_b}


def test_stratified_folds_partition_data_and_balance_strata():
    data = _mixed_stratum_data()

    folds = data_utils.stratified_folds(data, n_folds=5, seed=42)

    assert len(folds) == 5
    # Disjoint cover of the whole dataset.
    all_inputs = [i["input"] for fold in folds for i in fold]
    assert sorted(all_inputs) == sorted(i["input"] for i in data)
    assert len(all_inputs) == len(set(all_inputs))

    # Each fold balances the strata (4 single + 2 multi here).
    for fold in folds:
        singles = sum(1 for i in fold if data_utils.default_stratum(i) == "single")
        multis = sum(1 for i in fold if data_utils.default_stratum(i) == "multi")
        assert singles == 4
        assert multis == 2


def test_stratified_folds_requires_at_least_two_folds():
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        data_utils.stratified_folds(_mixed_stratum_data(), n_folds=1)


def test_kfold_split_uses_held_out_fold_as_test():
    data = _mixed_stratum_data()

    test_sets = []
    for fold_index in range(5):
        train, val, test = data_utils.kfold_split(
            data, n_folds=5, fold_index=fold_index, val_size=0.1, seed=42
        )
        # train / val / test are disjoint and cover the dataset.
        ids = [i["input"] for i in train + val + test]
        assert sorted(ids) == sorted(i["input"] for i in data)
        assert len(ids) == len(set(ids))
        # Validation slice is carved relative to the whole dataset.
        assert len(val) == round(len(data) * 0.1)
        test_sets.append({i["input"] for i in test})

    # The five held-out test folds tile the dataset without overlap.
    union = set().union(*test_sets)
    assert union == {i["input"] for i in data}
    assert sum(len(t) for t in test_sets) == len(data)


def test_kfold_split_rejects_out_of_range_fold_index():
    with pytest.raises(ValueError, match="fold_index"):
        data_utils.kfold_split(_mixed_stratum_data(), n_folds=5, fold_index=5)


def test_stratified_sample_covers_both_strata_for_small_n():
    data = _mixed_stratum_data()  # 20 single + 10 multi

    sample = data_utils.stratified_sample(data, n=2, seed=42)

    assert len(sample) == 2
    strata = {data_utils.default_stratum(i) for i in sample}
    # A 2-item smoke sample includes one single- and one multi-formula example.
    assert strata == {"single", "multi"}


def test_stratified_sample_is_deterministic_and_bounded():
    data = _mixed_stratum_data()

    first = data_utils.stratified_sample(data, n=4, seed=42)
    second = data_utils.stratified_sample(data, n=4, seed=42)

    assert [i["input"] for i in first] == [i["input"] for i in second]
    assert len(first) == 4
    # Round-robin keeps the strata balanced (2 single + 2 multi).
    counts = Counter(data_utils.default_stratum(i) for i in first)
    assert counts["single"] == 2
    assert counts["multi"] == 2


def test_stratified_sample_returns_all_when_n_exceeds_size():
    data = _mixed_stratum_data()
    assert len(data_utils.stratified_sample(data, n=999, seed=42)) == len(data)
    assert data_utils.stratified_sample(data, n=None, seed=42) == data
    assert data_utils.stratified_sample(data, n=0, seed=42) == []



def test_load_data_rejects_missing_output(tmp_path):
    p = tmp_path / "data.json"
    data_utils.save_data([{"input": "Agent A can guarantee p"}], str(p))

    with pytest.raises(ValueError, match="missing a non-empty 'outputs'"):
        data_utils.load_data(str(p))


def test_apply_paraphrase_changes_phrase():
    random.seed(7)
    text = "The system can guarantee that eventually p"
    paraphrased = data_utils._apply_paraphrase(text)
    assert paraphrased != text
