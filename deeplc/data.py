from __future__ import annotations

import numpy as np
import torch
from psm_utils import Peptidoform, PSMList
from torch.utils.data import Dataset, Subset

from deeplc._features import encode_peptidoform


class DeepLCDataset(Dataset):
    """Custom Dataset class for DeepLC used for loading features from peptide sequences."""

    def __init__(
        self,
        peptidoforms: list[Peptidoform | str],
        target_retention_times: np.ndarray | None = None,
        add_ccs_features: bool = False,
    ):
        self.peptidoforms = peptidoforms
        self.target_retention_times = target_retention_times
        self.add_ccs_features = add_ccs_features

    def __len__(self):
        return len(self.peptidoforms)

    def __getitem__(self, idx) -> tuple:
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)} instead.")
        features = encode_peptidoform(
            self.peptidoforms[idx], add_ccs_features=self.add_ccs_features
        )
        feature_tuples = (
            torch.from_numpy(features["matrix"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_sum"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_global"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_hc"]).to(dtype=torch.float32),
        )
        targets = (
            self.target_retention_times[idx]
            if self.target_retention_times is not None
            else torch.full_like(feature_tuples[0], fill_value=float("nan"), dtype=torch.float32)
        )
        return feature_tuples, targets
    
    @classmethod
    def from_psm_list(
        cls,
        psm_list: PSMList,
        add_ccs_features: bool = False,
    ) -> DeepLCDataset:
        """Create a DeepLCDataset from a PSMList."""
        peptidoforms = list(psm_list["peptidoform"])
        retention_times = psm_list["retention_time"]
        if None not in retention_times:
            target_retention_times = np.array(retention_times, dtype=np.float32)
        else:
            target_retention_times = None
        return cls(
            peptidoforms=peptidoforms,
            target_retention_times=target_retention_times,
            add_ccs_features=add_ccs_features,
        )


def split_datasets(
    train_data: Dataset,
    validation_data: Dataset | None,
    validation_split: float,
) -> tuple[Dataset, Dataset] | tuple[Subset, Subset]:
    """Split the dataset into training and validation sets."""
    if validation_data is None:
        if not hasattr(train_data, "__len__"):
            raise ValueError("Dataset must implement __len__ method for automatic splitting")
        dataset_len = len(train_data)  # type: ignore[arg-type]
        val_size = int(dataset_len * validation_split)
        train_size = dataset_len - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_data, [train_size, val_size]
        )
        return train_dataset, val_dataset
    else:
        return train_data, validation_data


def get_targets(psm_list: PSMList) -> torch.Tensor | None:
    retention_times = psm_list["retention_time"]
    if None not in retention_times:
        return torch.tensor(retention_times, dtype=torch.float32)
    else:
        return None
