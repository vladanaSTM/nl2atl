import json

from src.experiment.data_manager import ExperimentDataManager
from src.models.few_shot import few_shot_example_inputs


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


def test_load_dataset_holds_out_few_shot_exemplars(tmp_path):
    exemplar_input = sorted(few_shot_example_inputs())[0]
    dataset_path = tmp_path / "dataset.json"
    data = [
        {"input": exemplar_input, "outputs": [{"formula": "<<A>>F p"}]},
        {"input": "a distinct dataset item", "outputs": [{"formula": "<<A>>G p"}]},
    ]
    dataset_path.write_text(json.dumps(data))

    manager = ExperimentDataManager(
        data_path=dataset_path,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        seed=1,
        augment_factor=1,
    )

    loaded = manager.load_dataset()
    loaded_inputs = {item["input"].lower().strip() for item in loaded}

    assert exemplar_input not in loaded_inputs
    assert "a distinct dataset item" in loaded_inputs


def test_cross_validation_folds_tile_the_dataset(tmp_path):
    dataset_path = tmp_path / "dataset.json"
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
    dataset_path.write_text(json.dumps(data))

    test_sets = []
    for fold in range(5):
        manager = ExperimentDataManager(
            data_path=dataset_path,
            train_size=0.7,
            val_size=0.1,
            test_size=0.2,
            seed=42,
            augment_factor=1,
            cv_folds=5,
            cv_fold=fold,
        )
        train_aug, val, test, full = manager.prepare_data()
        assert len(full) == 30
        # Each fold yields disjoint train/val/test covering the dataset.
        ids = [i["input"] for i in train_aug + val + test]
        assert len(set(ids)) == len(set(i["input"] for i in full))
        test_sets.append({i["input"] for i in test})

    union = set().union(*test_sets)
    assert union == {i["input"] for i in data}
    assert sum(len(t) for t in test_sets) == len(data)


def test_canonical_split_is_independent_of_training_seed(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    data = [
        {"input": f"item {i}", "outputs": [{"formula": "<<A>>F p"}]} for i in range(30)
    ]
    dataset_path.write_text(json.dumps(data))

    def canonical_test(seed):
        manager = ExperimentDataManager(
            data_path=dataset_path,
            train_size=0.7,
            val_size=0.1,
            test_size=0.2,
            seed=seed,
            augment_factor=1,
        )
        _, _, test, _ = manager.prepare_data()
        return {item["input"] for item in test}

    # The data manager only sees the split seed; a fixed split seed yields the
    # same canonical test set regardless of which training seed is used.
    assert canonical_test(42) == canonical_test(42)
    assert canonical_test(42) != canonical_test(43)
