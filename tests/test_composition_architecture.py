"""Tests for the composition-encoder multitask model."""

from pathlib import Path

import numpy as np
import pytest
import torch

from deeplc._composition_architecture import CompositionMultitaskModel
from deeplc._model_ops import load_model
from deeplc.core import predict

MODEL_PATH = Path(__file__).parent.parent / "deeplc" / "package_data" / "models" / (
    "composition_multitask_model.pt"
)


def test_model_declares_its_feature_requirement():
    """The model needs terminal composition, and says so."""
    model = CompositionMultitaskModel(n_tasks=4)
    assert model.requires_terminal_composition is True


def test_forward_shape_and_ignored_sum_matrix():
    """Output is one prediction per LC setup; matrix_sum does not affect it."""
    model = CompositionMultitaskModel(n_tasks=7).eval()
    x_atom = torch.rand(3, 60, 6)
    x_global = torch.rand(3, 67)
    one_hot = torch.zeros(3, 60, 20)
    one_hot[:, :9, 5] = 1.0

    with torch.no_grad():
        first = model(x_atom, torch.zeros(3, 30, 6), x_global, one_hot)
        second = model(x_atom, torch.rand(3, 30, 6), x_global, one_hot)

    assert first.shape == (3, 7)
    assert torch.allclose(first, second), "matrix_sum must not change the prediction"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="packaged model not available")
def test_packaged_model_loads_and_predicts():
    """The shipped checkpoint round-trips through load_model and predict."""
    model = load_model(MODEL_PATH, device="cpu")
    assert isinstance(model, CompositionMultitaskModel)
    assert model.requires_terminal_composition is True

    peptides = ["PEPTIDEK", "[Acetyl]-PEPTIDEK", "ELVISLIVESK", "M[Oxidation]ACGHTR"]
    predictions = predict(
        peptides,
        model=MODEL_PATH,
        return_matrix=True,
        predict_kwargs={"show_progress": False, "device": "cpu"},
    )

    assert predictions.shape[0] == len(peptides)
    assert predictions.shape[1] > 1, "multitask model should return one column per LC setup"
    assert np.isfinite(predictions).all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="packaged model not available")
def test_terminal_modification_changes_the_prediction():
    """An N-terminal acetyl must not predict identically to the unmodified peptide.

    This is what the terminal-composition features exist for; without them the two
    encode the same and the model cannot tell them apart.
    """
    predictions = predict(
        ["PEPTIDEK", "[Acetyl]-PEPTIDEK"],
        model=MODEL_PATH,
        return_matrix=True,
        predict_kwargs={"show_progress": False, "device": "cpu"},
    )
    assert not np.allclose(predictions[0], predictions[1])
