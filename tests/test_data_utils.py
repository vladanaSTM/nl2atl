import random

import pytest

from src import data_utils


def test_save_and_load(tmp_path):
    data = [{"input": "Agent A can guarantee p", "output_2": "<<A>>F p"}]
    p = tmp_path / "data.json"
    data_utils.save_data(data, str(p))
    loaded = data_utils.load_data(str(p))
    assert isinstance(loaded, list)
    assert loaded[0]["input"] == data[0]["input"]
    assert loaded[0]["output"] == "<<A>>F p"
    assert loaded[0]["outputs"] == ["<<A>>F p"]


def test_load_data_preserves_multiple_correct_outputs(tmp_path):
    data = [
        {
            "input": "Every server can guarantee that next p holds.",
            "output_1": "<<Server1>>X p_1 && <<Server2>>X p_2",
            "output_2": "<<Server1,Server2>>X p",
        }
    ]
    p = tmp_path / "data.json"
    data_utils.save_data(data, str(p))

    loaded = data_utils.load_data(str(p))

    assert loaded[0]["output"] == "<<Server1,Server2>>X p"
    assert loaded[0]["outputs"] == [
        "<<Server1,Server2>>X p",
        "<<Server1>>X p_1 && <<Server2>>X p_2",
    ]


def test_augment_data_preserves_and_expands():
    data = [
        {"input": "The system can guarantee that eventually p", "output_2": "<<S>>F p"}
    ]
    random.seed(123)
    augmented = data_utils.augment_data(data, augment_factor=3)
    assert len(augmented) == 3
    # ensure original present
    assert any(item["input"] == data[0]["input"] for item in augmented)
    # ensure outputs preserved
    assert all("output" in item for item in augmented)


def test_augment_data_seed_is_independent_of_global_random_state():
    data = [
        {"input": "The system can guarantee that eventually p", "output_2": "<<S>>F p"}
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
    data = [{"input": "The system will always stay safe", "output_2": "<<S>>G safe"}]
    augmented = data_utils.augment_data(data, augment_factor=1)
    assert augmented == [
        {
            "input": "The system will always stay safe",
            "output_2": "<<S>>G safe",
            "outputs": ["<<S>>G safe"],
            "output": "<<S>>G safe",
        }
    ]


def test_augment_data_preserves_multiple_correct_outputs():
    data = [
        {
            "input": "Every server can guarantee that next p holds.",
            "output_1": "<<Server1>>X p_1 && <<Server2>>X p_2",
            "output_2": "<<Server1,Server2>>X p",
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
        data.append({"input": f"item a {i}", "output": "<<A>>F p"})
        data.append({"input": f"item b {i}", "output_2": "<<A>>G p"})

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
    data = [{"input": "A", "output": "<<A>>F p"}]
    with pytest.raises(ValueError, match="sum to 1.0"):
        data_utils.split_data(data, train_size=0.7, val_size=0.2, test_size=0.2)


def test_load_data_rejects_missing_output(tmp_path):
    p = tmp_path / "data.json"
    data_utils.save_data([{"input": "Agent A can guarantee p"}], str(p))

    with pytest.raises(ValueError, match="missing an output"):
        data_utils.load_data(str(p))


def test_apply_paraphrase_changes_phrase():
    random.seed(7)
    text = "The system can guarantee that eventually p"
    paraphrased = data_utils._apply_paraphrase(text)
    assert paraphrased != text
