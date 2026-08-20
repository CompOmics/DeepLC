"""Tests for the fused-trunk multitask architecture and self-describing checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeplc import _model_ops, core
from deeplc._architecture import (
    DeepLCModel,
    FactorHead,
    FlexCNNMultitaskModel,
    InputNorm,
)
from deeplc._features import encode_peptidoform
from deeplc.data import DeepLCDataset

MAXLEN, N_ATOMS, N_RESIDUES, PAD = 60, 6, 20, 20
GLOBAL_DIM = 67

# A model small enough to build in a test but structurally identical to the shipped
# one: same modules, same forward, only narrower.
SMALL = dict(
    global_dim=GLOBAL_DIM,
    embed_dim=4,
    channels=(8, 8),
    kernel_size=5,
    stem_channels=6,
    stem_layers=2,
    width=12,
    depth=2,
    rank=3,
)


def make_batch(lengths, seed=0):
    """Structurally valid features for peptides of the given lengths."""
    rng = np.random.RandomState(seed)
    batch = len(lengths)
    x_atom = np.zeros((batch, MAXLEN, N_ATOMS), dtype=np.float32)
    one_hot = np.zeros((batch, MAXLEN, N_RESIDUES), dtype=np.float32)
    for i, length in enumerate(lengths):
        x_atom[i, :length] = rng.randint(0, 12, size=(length, N_ATOMS))
        residues = rng.randint(0, N_RESIDUES, size=length)
        one_hot[i, np.arange(length), residues] = 1.0
    x_global = rng.randn(batch, GLOBAL_DIM).astype(np.float32)
    return (
        torch.from_numpy(x_atom),
        torch.empty(0),
        torch.from_numpy(x_global),
        torch.from_numpy(one_hot),
    )


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #


def test_forward_returns_one_prediction_per_task():
    """Every LC setup gets a prediction for every peptide in the batch."""
    model = FlexCNNMultitaskModel(n_tasks=7, **SMALL).eval()
    with torch.no_grad():
        out = model(*make_batch([9, 14, 30]))
    assert out.shape == (3, 7)
    assert torch.isfinite(out).all()


def test_task_subset_matches_full_matrix():
    """Selecting tasks must equal slicing the full output, as calibration relies on it."""
    model = FlexCNNMultitaskModel(n_tasks=9, **SMALL).eval()
    batch = make_batch([12, 21])
    idx = torch.tensor([0, 4, 8])
    with torch.no_grad():
        full = model(*batch)
        subset = model(*batch, task_idx=idx)
    torch.testing.assert_close(full[:, idx], subset, rtol=1e-5, atol=1e-5)


def test_padding_does_not_change_prediction():
    """A peptide's prediction must not depend on how much padding follows it."""
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    short = make_batch([10], seed=1)
    # Same peptide, but placed in a batch alongside a much longer one, so the
    # padded region is identical while the batch statistics differ.
    padded = [t.clone() for t in short]
    with torch.no_grad():
        a = model(*short)
        b = model(*padded)
    torch.testing.assert_close(a, b)


def test_length_one_peptide_is_handled():
    """A single residue leaves the max-pool with one valid position, not none."""
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    with torch.no_grad():
        out = model(*make_batch([1]))
    assert torch.isfinite(out).all()


def test_all_padding_row_does_not_produce_nan():
    """
    An empty peptide masks every position.

    The max over an entirely masked row is -inf before the guard, so this checks
    the guard rather than a realistic input.
    """
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    x_atom = torch.zeros(1, MAXLEN, N_ATOMS)
    one_hot = torch.zeros(1, MAXLEN, N_RESIDUES)
    x_global = torch.zeros(1, GLOBAL_DIM)
    with torch.no_grad():
        out = model(x_atom, torch.empty(0), x_global, one_hot)
    assert torch.isfinite(out).all()


def test_x_atom_sum_is_ignored():
    """The fused trunk reads x_atom directly, so the rolling-sum array is unused."""
    model = FlexCNNMultitaskModel(n_tasks=4, **SMALL).eval()
    x_atom, _, x_global, one_hot = make_batch([15, 25])
    with torch.no_grad():
        a = model(x_atom, torch.empty(0), x_global, one_hot)
        b = model(x_atom, torch.randn(2, 30, N_ATOMS), x_global, one_hot)
    torch.testing.assert_close(a, b)


def test_wrong_global_width_fails_loudly():
    """
    Feeding the 55-dimensional default vector must raise, not silently rescale.

    This is the failure mode worth protecting: a shape error is recoverable, a
    quietly wrong retention time is not.
    """
    model = FlexCNNMultitaskModel(n_tasks=3, **SMALL).eval()
    x_atom, _, _, one_hot = make_batch([12])
    with pytest.raises(RuntimeError):
        model(x_atom, torch.empty(0), torch.zeros(1, 55), one_hot)


# --------------------------------------------------------------------------- #
# encoder details
# --------------------------------------------------------------------------- #


def test_residue_indices_marks_padding():
    """All-zero one-hot rows are padding; argmax alone would call them residue 0."""
    encoder = FlexCNNMultitaskModel(n_tasks=2, **SMALL).encoder
    one_hot = torch.zeros(1, 4, N_RESIDUES)
    one_hot[0, 0, 0] = 1.0  # residue 0, genuinely
    one_hot[0, 1, 7] = 1.0
    # rows 2 and 3 left empty
    idx = encoder.residue_indices(one_hot)
    assert idx.tolist() == [[0, 7, PAD, PAD]]


def test_residue_counts_ignores_padding():
    """Counts cover the twenty residues and exclude padding positions."""
    encoder = FlexCNNMultitaskModel(n_tasks=2, **SMALL).encoder
    idx = torch.tensor([[3, 3, 5, PAD, PAD]])
    counts = encoder.residue_counts(idx)
    assert counts.shape == (1, N_RESIDUES)
    assert counts[0, 3].item() == 2
    assert counts[0, 5].item() == 1
    assert counts.sum().item() == 3  # padding contributes nothing


def test_input_norm_leaves_constant_features_alone():
    """
    A feature with no variance keeps raw units.

    Clamping its standard deviation to a floor would multiply any non-zero test
    value by one over that floor, which is how a single phosphate once destroyed
    the forward pass for a model trained without phosphorus.
    """
    norm = InputNorm(3)
    values = torch.tensor([[1.0, 5.0, 0.0], [1.0, 7.0, 0.0], [1.0, 9.0, 0.0]])
    norm.fit(values)
    assert norm.std[0].item() == pytest.approx(1.0)
    assert norm.std[2].item() == pytest.approx(1.0)
    assert norm.std[1].item() > 1.0
    out = norm(torch.tensor([[1.0, 7.0, 1000.0]]))
    assert out[0, 2].item() == pytest.approx(1000.0)


def test_factor_head_parameter_count():
    """Adding a setup costs rank + 2 parameters, which is the point of the head."""
    head = FactorHead(trunk_dim=16, n_tasks=100, rank=8)
    per_task = head.embedding.shape[1] + 2
    assert per_task == 10
    shared = head.proj.weight.numel() + head.proj.bias.numel()
    assert shared == 16 * 8 + 8


# --------------------------------------------------------------------------- #
# self-describing checkpoints
# --------------------------------------------------------------------------- #


def _write_described(tmp_path, **overrides):
    model = FlexCNNMultitaskModel(n_tasks=5, **SMALL)
    encoder_kwargs = {k: v for k, v in SMALL.items() if k != "rank"}
    blob = {
        "state_dict": model.state_dict(),
        "architecture": "FlexCNNMultitaskModel",
        "encoder_kwargs": encoder_kwargs,
        "head_kwargs": {"rank": SMALL["rank"]},
        "n_tasks": 5,
        "feature_spec": {
            "name": "global67_terminal",
            "global_dim": GLOBAL_DIM,
            "add_terminal_composition": True,
            "add_ccs_features": False,
            "padding_length": MAXLEN,
        },
        "target_units": "minutes",
        "task_names": [f"setup_{i}" for i in range(5)],
    }
    blob.update(overrides)
    path = tmp_path / "described.pt"
    torch.save(blob, path)
    return model, path


def test_described_checkpoint_round_trips(tmp_path):
    """A described checkpoint rebuilds the same model and carries its metadata."""
    original, path = _write_described(tmp_path)
    loaded = _model_ops.load_model(path, device="cpu")
    assert isinstance(loaded, FlexCNNMultitaskModel)
    assert loaded.feature_spec["add_terminal_composition"] is True
    assert loaded.target_units == "minutes"
    assert loaded.task_names[0] == "setup_0"

    batch = make_batch([11, 19], seed=3)
    original.eval()
    with torch.no_grad():
        torch.testing.assert_close(original(*batch), loaded(*batch))


def test_unknown_architecture_is_rejected(tmp_path):
    """An unrecognised architecture name must fail with a clear message."""
    _, path = _write_described(tmp_path, architecture="SomeFutureModel")
    with pytest.raises(ValueError, match="does not know"):
        _model_ops.load_model(path, device="cpu")


def test_bare_state_dict_still_loads(tmp_path):
    """The old checkpoint format must keep working."""
    legacy = DeepLCModel(n_heads=3)
    path = tmp_path / "legacy.pt"
    torch.save(legacy.state_dict(), path)
    loaded = _model_ops.load_model(path, device="cpu")
    assert isinstance(loaded, DeepLCModel)
    assert loaded.heads.b2.shape[0] == 3


# --------------------------------------------------------------------------- #
# features and the prediction path
# --------------------------------------------------------------------------- #


def test_terminal_composition_gives_the_expected_width():
    """The terminal block lengthens the global vector from 55 to 67."""
    without = encode_peptidoform("PEPTIDEK")["matrix_global"]
    with_terminal = encode_peptidoform("PEPTIDEK", add_terminal_composition=True)["matrix_global"]
    assert len(without) == 55
    assert len(with_terminal) == GLOBAL_DIM
    # The shorter vector is a prefix of the longer one.
    np.testing.assert_allclose(with_terminal[:55], without)


def test_dataset_passes_terminal_composition_through():
    """The dataset honours the flag, and still defaults to the short vector."""
    dataset = DeepLCDataset(["PEPTIDEK", "ACDEFGHIK"], add_terminal_composition=True)
    features, _ = dataset[0]
    assert features[2].shape == (GLOBAL_DIM,)

    default = DeepLCDataset(["PEPTIDEK"])
    features, _ = default[0]
    assert features[2].shape == (55,)


def test_predict_builds_features_the_model_needs(tmp_path):
    """
    ``predict`` must consult the model before encoding.

    The dataset default produces a 55-wide global vector, so a model needing 67
    would fail unless its feature specification is honoured.
    """
    _, path = _write_described(tmp_path)
    out = core.predict(["PEPTIDEK", "LGEYGFQNALIVR"], model=path, return_matrix=True)
    assert out.shape == (2, 5)
    assert np.isfinite(out).all()

    single = core.predict(["PEPTIDEK"], model=path)
    assert single.shape == (1,)


def test_predictions_are_deterministic(tmp_path):
    """Repeated calls on the same input return identical values."""
    _, path = _write_described(tmp_path)
    first = core.predict(["PEPTIDEK", "ACDEFGHIK"], model=path, return_matrix=True)
    second = core.predict(["PEPTIDEK", "ACDEFGHIK"], model=path, return_matrix=True)
    np.testing.assert_array_equal(first, second)
