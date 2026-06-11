"""Data management for experiments."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..data_utils import (
    load_data,
    split_data,
    augment_data,
    kfold_split,
    default_stratum,
)
from ..models.few_shot import few_shot_example_inputs


class ExperimentDataManager:
    """Handles data loading, splitting, and caching for experiments."""

    def __init__(
        self,
        data_path: Path,
        train_size: float,
        test_size: float,
        val_size: float,
        seed: int,
        augment_factor: int,
        stratify: bool = True,
        cv_folds: int = 0,
        cv_fold: Optional[int] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.augment_factor = augment_factor
        self.stratify = stratify
        self.cv_folds = cv_folds
        self.cv_fold = cv_fold

        self.data: Optional[List[Dict[str, Any]]] = None
        self.train_data: Optional[List[Dict[str, Any]]] = None
        self.val_data: Optional[List[Dict[str, Any]]] = None
        self.test_data: Optional[List[Dict[str, Any]]] = None
        self.train_data_aug: Optional[List[Dict[str, Any]]] = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from configured path.

        Items whose input duplicates a curated few-shot exemplar are held out so
        the few-shot demonstrations never appear in any train/val/test split.
        """
        data = load_data(str(self.data_path))
        held_out = few_shot_example_inputs()
        self.data = [
            item
            for item in data
            if str(item.get("input", "")).lower().strip() not in held_out
        ]
        return self.data

    def split_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train/val/test.

        Uses a fixed, optionally stratified canonical split by default. When
        ``cv_folds >= 2`` and ``cv_fold`` is set, returns the requested
        stratified cross-validation fold instead.
        """
        stratify_key = default_stratum if self.stratify else None
        if self.cv_folds and self.cv_folds >= 2 and self.cv_fold is not None:
            train, val, test = kfold_split(
                dataset,
                n_folds=self.cv_folds,
                fold_index=self.cv_fold,
                val_size=self.val_size,
                seed=self.seed,
                stratify_key=stratify_key,
            )
        else:
            train, val, test = split_data(
                dataset,
                train_size=self.train_size,
                test_size=self.test_size,
                val_size=self.val_size,
                seed=self.seed,
                stratify_key=stratify_key,
            )
        self.train_data, self.val_data, self.test_data = train, val, test
        return train, val, test

    def augment_training_data(
        self, train_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Augment training data."""
        self.train_data_aug = augment_data(
            train_data,
            self.augment_factor,
            seed=self.seed,
        )
        return self.train_data_aug

    def prepare_data(
        self,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Load originals, split them, then augment the training split only."""
        dataset = self.load_dataset()
        train, val, test = self.split_dataset(dataset)
        train_aug = self.augment_training_data(train)
        return train_aug, val, test, dataset
