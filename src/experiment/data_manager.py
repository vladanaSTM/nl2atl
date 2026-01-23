"""Data management for experiments."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..data_utils import load_data, split_data, augment_data


class ExperimentDataManager:
    """Handles data loading, splitting, and caching for experiments."""

    def __init__(
        self,
        data_path: Path,
        test_size: float,
        val_size: float,
        seed: int,
        augment_factor: int,
    ) -> None:
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.augment_factor = augment_factor

        self.data: Optional[List[Dict[str, Any]]] = None
        self.train_data: Optional[List[Dict[str, Any]]] = None
        self.val_data: Optional[List[Dict[str, Any]]] = None
        self.test_data: Optional[List[Dict[str, Any]]] = None
        self.train_data_aug: Optional[List[Dict[str, Any]]] = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from configured path."""
        self.data = load_data(str(self.data_path))
        return self.data

    def split_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train/val/test."""
        train, val, test = split_data(
            dataset,
            test_size=self.test_size,
            val_size=self.val_size,
            seed=self.seed,
        )
        self.train_data, self.val_data, self.test_data = train, val, test
        return train, val, test

    def augment_training_data(
        self, train_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Augment training data."""
        self.train_data_aug = augment_data(train_data, self.augment_factor)
        return self.train_data_aug

    def prepare_data(
        self,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Load, split, and augment dataset."""
        dataset = self.load_dataset()
        train, val, test = self.split_dataset(dataset)
        train_aug = self.augment_training_data(train)
        return train_aug, val, test, dataset
