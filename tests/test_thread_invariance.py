"""CPU inference and small-reference regressions for issue #136."""

import numpy as np
import pytest
import torch

from deeplc import _model_ops
from deeplc._architecture import FlexCNNMultitaskModel
from deeplc.calibration import MultiHeadRidgeCalibration
from deeplc.data import DeepLCDataset


@pytest.mark.parametrize("length_buckets", [False, True])
def test_cpu_predict_uses_the_same_kernel_threads_per_batch(monkeypatch, length_buckets):
    """The worker count controls scheduling, while arithmetic and row order stay fixed."""
    model = FlexCNNMultitaskModel(n_tasks=1)
    dataset = DeepLCDataset(["PEPTIDEK", "ACDK", "ACDEFGHIK", "PEPTIDEPEPTIDEK", "ACD"])
    worker_threads = []

    def thread_sensitive_forward(atoms, *_):
        # Model a thread-dependent numerical kernel in a backend-independent way.
        worker_threads.append(torch.get_num_threads())
        assert not torch.is_grad_enabled()
        return atoms.sum(dim=(1, 2))[:, None] + torch.get_num_threads()

    monkeypatch.setattr(model, "forward", thread_sensitive_forward)
    kwargs = {
        "batch_size": 2,
        "device": "cpu",
        "length_buckets": length_buckets,
        "show_progress": False,
    }

    prior_threads = torch.get_num_threads()
    try:
        sequential = _model_ops.predict(model, dataset, num_threads=1, **kwargs)
        parallel = _model_ops.predict(model, dataset, num_threads=4, **kwargs)
    finally:
        torch.set_num_threads(prior_threads)

    torch.testing.assert_close(sequential, parallel, atol=0, rtol=0)
    assert len(worker_threads) >= 6
    assert set(worker_threads) == {1}


def test_small_reference_floor_for_nonnegative_retention_times():
    """A negative linear extrapolation is outside a nonnegative RT scale."""
    source = np.array([[1.0], [2.0], [3.0]])
    target = np.array([10.0, 20.0, 30.0])
    query = np.array([[-1.0], [2.0], [4.0]])
    fit = MultiHeadRidgeCalibration()
    fit.fit(target, source)
    np.testing.assert_allclose(fit.transform(query), [0.0, 20.0, 40.0])

    # Refit to a signed, centred scale. The previous fit's floor must not carry over.
    fit.fit(target - 20, source)
    np.testing.assert_allclose(fit.transform(query), [-30.0, 0.0, 20.0], atol=1e-12)


def test_cpu_predict_restores_intraop_threads_if_forward_fails(monkeypatch):
    """A failing worker must leave the caller's Torch setting intact."""
    model = FlexCNNMultitaskModel(n_tasks=1)
    dataset = DeepLCDataset(["PEPTIDEK", "ACDEFGHIK", "SEQUENCEWITHK"])

    def fail(*_):
        raise ValueError("forward failed")

    monkeypatch.setattr(model, "forward", fail)
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with pytest.raises(ValueError, match="forward failed"):
            _model_ops.predict(
                model, dataset, device="cpu", batch_size=1, num_threads=4, show_progress=False
            )
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(prior_threads)


def test_cpu_batch_cap_preserves_padding_windows(monkeypatch):
    """Dividing a length bucket among workers must leave its encoded window intact."""
    model = FlexCNNMultitaskModel(n_tasks=1)
    peptides = ["ACDEFGH", "ACDEFGHI", "ACDEFGHIK", "ACDEFGHIKL"]
    dataset = DeepLCDataset(peptides)
    batch_shapes = []

    def observe(atoms, *_):
        batch, width, _ = atoms.shape
        batch_shapes.append((batch, width))
        return torch.zeros(batch, 1)

    monkeypatch.setattr(model, "forward", observe)
    monkeypatch.setattr(_model_ops, "_CPU_BATCH_SIZE", 2, raising=False)
    _model_ops.predict(
        model, dataset, device="cpu", batch_size=4, num_threads=1, show_progress=False
    )

    original_window = max(map(len, peptides)) + model.padding_reach
    assert batch_shapes == [(2, original_window), (2, original_window)]


def test_predict_outside_the_parallel_path_restores_intraop_threads(monkeypatch):
    """Models that skip the parallel CPU path also leave the caller's Torch setting intact."""

    class CustomModel(FlexCNNMultitaskModel):
        pass

    model = CustomModel(n_tasks=1)
    monkeypatch.setattr(model, "forward", lambda atoms, *_: torch.zeros(len(atoms), 1))
    dataset = DeepLCDataset(["PEPTIDEK", "ACDK"])
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        _model_ops.predict(model, dataset, device="cpu", num_threads=3, show_progress=False)
        assert torch.get_num_threads() == 2
        with pytest.raises(ValueError, match="num_threads"):
            _model_ops.predict(model, dataset, device="cpu", num_threads=-1, show_progress=False)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(prior_threads)
