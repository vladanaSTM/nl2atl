import random

from src import data_utils


def test_save_and_load(tmp_path):
    data = [{"input": "Agent A can guarantee p", "output_2": "<<A>>F p"}]
    p = tmp_path / "data.json"
    data_utils.save_data(data, str(p))
    loaded = data_utils.load_data(str(p))
    assert isinstance(loaded, list)
    assert loaded[0]["input"] == data[0]["input"]
    assert loaded[0]["output"] == "<<A>>F p"


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


def test_augment_data_no_change_when_factor_one():
    data = [{"input": "The system will always stay safe", "output_2": "<<S>>G safe"}]
    augmented = data_utils.augment_data(data, augment_factor=1)
    assert augmented == [
        {
            "input": "The system will always stay safe",
            "output_2": "<<S>>G safe",
            "output": "<<S>>G safe",
        }
    ]


def test_split_data_counts_without_stratification():
    data = []
    for i in range(10):
        data.append({"input": f"item a {i}", "output": "<<A>>F p"})
        data.append({"input": f"item b {i}", "output_2": "<<A>>G p"})

    train, val, test = data_utils.split_data(
        data, test_size=0.2, val_size=0.5, seed=123
    )

    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2
    assert len(train) + len(val) + len(test) == len(data)
    assert sorted(item["input"] for item in train + val + test) == sorted(
        item["input"] for item in data
    )


def test_apply_paraphrase_changes_phrase():
    random.seed(7)
    text = "The system can guarantee that eventually p"
    paraphrased = data_utils._apply_paraphrase(text)
    assert paraphrased != text
