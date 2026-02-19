from __future__ import annotations

import logging
from typing import TypeVar, overload

import numpy as np
import torch
from psm_utils import Peptidoform, PSMList
from torch.utils.data import Dataset, Subset

from deeplc._features import encode_peptidoform

_DatasetT = TypeVar("_DatasetT", bound=Dataset)

LOGGER = logging.getLogger(__name__)


class DeepLCDataset(Dataset):
    """Custom Dataset class for DeepLC used for loading features from peptide sequences."""

    def __init__(
        self,
        peptidoforms: list[Peptidoform | str],
        target_retention_times: np.ndarray | None = None,
        add_ccs_features: bool = False,
    ):
        """
        Initialize the DeepLCDataset.

        Parameters
        ----------
        peptidoforms
            A list of peptidoforms, which can be either Peptidoform objects or their string
            representations.
        target_retention_times
            An array of target retention times corresponding to the peptidoforms. If None, targets
            will be set to NaN.
        add_ccs_features
            Whether to include CCS features in the encoded representation. Default is False.

        Raises
        ------
        ValueError
            If ``target_retention_times`` is provided and its length does not match the number of
            peptidoforms.

        """
        self.peptidoforms = peptidoforms
        self.target_retention_times = target_retention_times
        self.add_ccs_features = add_ccs_features
        if self.target_retention_times is not None and len(self.target_retention_times) != len(
            self.peptidoforms
        ):
            raise ValueError(
                f"Length of target_retention_times ({len(self.target_retention_times)}) does not "
                f"match length of peptidoforms ({len(self.peptidoforms)})"
            )

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
        """
        Create a DeepLCDataset from a PSMList.

        Parameters
        ----------
        psm_list
            A PSMList containing the peptidoforms and their corresponding retention times.
        add_ccs_features
            Whether to include CCS features in the encoded representation. Default is False.

        Returns
        -------
        DeepLCDataset
            A DeepLCDataset instance created from the provided PSMList.

        """
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


@overload
def split_datasets(
    train_data: _DatasetT,
    validation_data: _DatasetT,
    validation_split: float,
) -> tuple[_DatasetT, _DatasetT]: ...


@overload
def split_datasets(
    train_data: _DatasetT,
    validation_data: None,
    validation_split: float,
) -> tuple[Subset[_DatasetT], Subset[_DatasetT]]: ...


def split_datasets(
    train_data: Dataset,
    validation_data: Dataset | None,
    validation_split: float,
) -> tuple[Dataset, Dataset] | tuple[Subset, Subset]:
    """
    Split the dataset into training and validation sets.

    Parameters
    ----------
    train_data
        The dataset to be split.
    validation_data
        If provided, this dataset will be used as the validation set. If None, the train_data will
        be split.
    validation_split
        The fraction of the dataset to be used as the validation set if validation_data is None.

    Returns
    -------
    tuple[Dataset, Dataset] | tuple[Subset, Subset]
        A tuple containing the training and validation datasets.

    Raises
    ------
    ValueError
        If validation_data is None and train_data does not implement ``__len__`` method.

    """
    # TODO: Implement stratified splitting based on stripped sequence
    if validation_data is None:
        if not hasattr(train_data, "__len__"):
            raise ValueError("Dataset must implement __len__ method for automatic splitting")
        dataset_len = len(train_data)  # type: ignore[arg-type]
        val_size = int(dataset_len * validation_split)
        train_size = dataset_len - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_data, [train_size, val_size]
        )
        LOGGER.info(
            "No validation data provided. Split training dataset into validation set of size "
            f"{len(val_dataset)} and training set of size {len(train_dataset)}"
        )
        return train_dataset, val_dataset
    else:
        LOGGER.info("Using provided validation dataset")
        return train_data, validation_data
