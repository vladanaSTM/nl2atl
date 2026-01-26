import random

from src import data_utils


def test_save_and_load(tmp_path):
    data = [
        {"input": "Agent A can guarantee p", "output": "<<A>>F p", "difficulty": "easy"}
    ]
    p = tmp_path / "data.json"
    data_utils.save_data(data, str(p))
    loaded = data_utils.load_data(str(p))
    assert isinstance(loaded, list)
    assert loaded[0]["input"] == data[0]["input"]


def test_augment_data_preserves_and_expands():
    data = [
        {"input": "The system can guarantee that eventually p", "output": "<<S>>F p"}
    ]
    random.seed(123)
    augmented = data_utils.augment_data(data, augment_factor=3)
    assert len(augmented) == 3
    # ensure original present
    assert any(item["input"] == data[0]["input"] for item in augmented)
    # ensure outputs preserved
    assert all("output" in item for item in augmented)


def test_augment_data_no_change_when_factor_one():
    data = [{"input": "The system will always stay safe", "output": "<<S>>G safe"}]
    augmented = data_utils.augment_data(data, augment_factor=1)
    assert augmented == data


def test_split_data_stratified_counts():
    data = []
    for i in range(10):
        data.append({"input": f"easy {i}", "output": "<<A>>F p", "difficulty": "easy"})
        data.append({"input": f"hard {i}", "output": "<<A>>G p", "difficulty": "hard"})

    train, val, test = data_utils.split_data(
        data, test_size=0.2, val_size=0.5, seed=123
    )

    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2
    assert len(train) + len(val) + len(test) == len(data)


def test_apply_paraphrase_changes_phrase():
    random.seed(7)
    text = "The system can guarantee that eventually p"
    paraphrased = data_utils._apply_paraphrase(text)
    assert paraphrased != text
