import json

from src.experiment.data_manager import ExperimentDataManager


def test_prepare_data_splits_and_augments(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = []
    for i in range(10):
        data.append({"input": f"item a {i}", "output": "<<A>>F p"})
        data.append({"input": f"item b {i}", "output_2": "<<A>>G p"})
    dataset_path.write_text(json.dumps(data))

    manager = ExperimentDataManager(
        data_path=dataset_path,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        seed=1,
        augment_factor=2,
    )

    train_aug, val, test, full = manager.prepare_data()

    assert len(full) == 20
    assert len(val) == 2
    assert len(test) == 4
    assert len(train_aug) == 28
