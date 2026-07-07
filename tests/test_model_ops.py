from __future__ import annotations

import sys

import pytest
import torch

from deeplc import _model_ops
from torch.utils.data import Dataset

from deeplc._architecture import DeepLCModel
from deeplc.core import DEFAULT_MODEL
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


def test_predict_raises_for_empty_dataset():
    empty_data = _TinyDeepLCDataset(length=0)
    with pytest.raises(ValueError, match="empty"):
        _model_ops.predict(model=DeepLCModel(n_heads=1), data=empty_data, show_progress=False)


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
            model=DeepLCModel(n_heads=1),
            train_dataset=_TinyDeepLCDataset(length=2),
            validation_dataset=_TinyDeepLCDataset(length=0),
            epochs=1,
            batch_size=2,
            show_progress=False,
        )


@pytest.mark.skipif(
    not DEFAULT_MODEL.exists(),
    reason="multitask model not bundled",
)
def test_load_multitask_model_without_prior_shim():
    """multitask_model.pt must load even when the legacy module is not pre-registered."""
    # Remove any previously registered shim so the test is self-contained.
    sys.modules.pop("multitask_model", None)

    model = _model_ops.load_model(DEFAULT_MODEL, device="cpu")

    assert isinstance(model, DeepLCModel)

    x_atom = torch.zeros(2, 60, 6)
    x_sum = torch.zeros(2, 30, 6)
    x_global = torch.zeros(2, 55)
    x_hc = torch.zeros(2, 60, 20)
    with torch.no_grad():
        out = model(x_atom, x_sum, x_global, x_hc)

    assert out.ndim == 2
    assert out.shape[0] == 2
    assert out.shape[1] > 1  # multiple heads
