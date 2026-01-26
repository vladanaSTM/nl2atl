import json

from src.experiment.data_manager import ExperimentDataManager


def test_prepare_data_splits_and_augments(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = []
    for i in range(10):
        data.append({"input": f"easy {i}", "output": "<<A>>F p", "difficulty": "easy"})
        data.append({"input": f"hard {i}", "output": "<<A>>G p", "difficulty": "hard"})
    dataset_path.write_text(json.dumps(data))

    manager = ExperimentDataManager(
        data_path=dataset_path,
        test_size=0.2,
        val_size=0.5,
        seed=1,
        augment_factor=2,
    )

    train_aug, val, test, full = manager.prepare_data()

    assert len(full) == 20
    assert len(val) == 2
    assert len(test) == 2
    assert len(train_aug) == 32
