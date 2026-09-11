from __future__ import annotations

import sys

import pytest
import torch

from deeplc import _model_ops
from torch.utils.data import Dataset

from deeplc._architecture import DeepLCModel
from deeplc.core import LEGACY_MULTITASK_MODEL
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
    not LEGACY_MULTITASK_MODEL.exists(),
    reason="multitask model not bundled",
)
def test_load_multitask_model_without_prior_shim():
    """multitask_model.pt must load even when the legacy module is not pre-registered."""
    # Remove any previously registered shim so the test is self-contained.
    sys.modules.pop("multitask_model", None)

    model = _model_ops.load_model(LEGACY_MULTITASK_MODEL, device="cpu")

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


def test_predict_output_matches_batchwise_concatenation():
    """The preallocated output must equal what concatenating the batches produced."""
    model, dataset = DeepLCModel(n_heads=1), _TinyDeepLCDataset(length=37)
    batched = _model_ops.predict(
        model, dataset, device="cpu", batch_size=8, show_progress=False, length_buckets=False
    )
    single = _model_ops.predict(
        model, dataset, device="cpu", batch_size=1000, show_progress=False, length_buckets=False
    )
    assert batched.shape == single.shape == (37, 1)
    torch.testing.assert_close(batched, single)


def test_allocate_output_reports_the_size_when_it_does_not_fit():
    """An output that cannot be allocated says how large it was and how to shrink it."""
    with pytest.raises(MemoryError, match=r"task_idx"):
        _model_ops._allocate_output(2**40, (6543,), torch.float32)


def test_predict_propagates_the_allocation_failure(monkeypatch):
    """The helper's message reaches the caller of predict() rather than a raw allocator error."""
    def _refuse(*args, **kwargs):
        raise MemoryError(_model_ops._output_hint(2**40, 6543, 4))

    monkeypatch.setattr(_model_ops, "_allocate_output", _refuse)
    with pytest.raises(MemoryError, match=r"task_idx"):
        _model_ops.predict(
            model=DeepLCModel(n_heads=1),
            data=_TinyDeepLCDataset(length=4),
            device="cpu",
            show_progress=False,
            length_buckets=False,
        )


def test_output_hint_mentions_the_size_in_gigabytes():
    """The hint names the size that failed, so a log line is enough to diagnose it."""
    hint = _model_ops._output_hint(2_587_932, 6543, 4)
    assert "63.1 GiB" in hint
    assert "task_idx" in hint
