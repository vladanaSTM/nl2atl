import json

from src.experiment.data_manager import ExperimentDataManager


def test_prepare_data_splits_and_augments(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = []
    for i in range(10):
        data.append({"input": f"item a {i}", "outputs": [{"formula": "<<A>>F p"}]})
        data.append({"input": f"item b {i}", "outputs": [{"formula": "<<A>>G p"}]})
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


def test_prepare_data_augmentation_uses_manager_seed(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = [
        {
            "input": f"item {index} can guarantee that eventually p",
            "outputs": [{"formula": "<<A>>F p"}],
        }
        for index in range(10)
    ]
    dataset_path.write_text(json.dumps(data))

    def prepare_inputs_after_global_noise():
        manager = ExperimentDataManager(
            data_path=dataset_path,
            train_size=0.7,
            val_size=0.1,
            test_size=0.2,
            seed=11,
            augment_factor=3,
        )
        train_aug, _, _, _ = manager.prepare_data()
        return [item["input"] for item in train_aug]

    first = prepare_inputs_after_global_noise()

    import random

    random.seed(999)
    for _ in range(20):
        random.random()

    second = prepare_inputs_after_global_noise()

    assert first == second
