from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from deeplc import _model_ops
from deeplc.data import split_datasets


class _TinyDeepLCDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        features = (
            torch.zeros((60, 6), dtype=torch.float32),
            torch.zeros((30, 6), dtype=torch.float32),
            torch.zeros((55,), dtype=torch.float32),
            torch.zeros((60, 20), dtype=torch.float32),
        )
        target = torch.tensor(0.0, dtype=torch.float32)
        return features, target


class _DummyModel(torch.nn.Module):
    def forward(self, matrix, matrix_sum, matrix_global, matrix_hc):  # noqa: ARG002
        batch_size = matrix.shape[0]
        return torch.zeros((batch_size, 1), dtype=torch.float32)


def test_predict_returns_empty_tensor_for_empty_dataset():
    empty_data = _TinyDeepLCDataset(length=0)
    preds = _model_ops.predict(model=_DummyModel(), data=empty_data, show_progress=False)
    assert isinstance(preds, torch.Tensor)
    assert preds.numel() == 0


def test_split_datasets_rejects_too_small_dataset_without_validation_data():
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        split_datasets(
            train_data=_TinyDeepLCDataset(length=1),
            validation_data=None,
            validation_split=0.1,
        )


def test_train_rejects_empty_validation_loader():
    with pytest.raises(ValueError, match="Validation data loader is empty"):
        _model_ops.train(
            model=_DummyModel(),
            train_dataset=_TinyDeepLCDataset(length=2),
            validation_dataset=_TinyDeepLCDataset(length=0),
            epochs=1,
            batch_size=2,
            show_progress=False,
        )
