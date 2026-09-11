"""The factored prediction matrix must equal the matrix it stands for."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeplc import _model_ops, core
from deeplc._architecture import DeepLCModel
from deeplc._factored import FactoredPredictionMatrix
from deeplc.core import DEFAULT_MODEL

PREDICT_KWARGS = {"device": "cpu", "show_progress": False}
PEPTIDES = [
    "PEPTIDEK", "LVVVGAGGVGK", "GPNGPWSVMK", "YPLQLAELLK", "VVEEAVDLFK",
    "AAELALR", "SLIDLLQK", "ELVISLIVESK", "AAAAAAAAAAK", "WWWWK",
]


def _matrices():
    """Return the same predictions as factors and as a dense matrix."""
    lazy = core.predict(
        PEPTIDES, model=None, predict_kwargs={**PREDICT_KWARGS, "factored": True},
        return_matrix=True,
    )
    dense = core.predict(
        PEPTIDES, model=None, predict_kwargs=PREDICT_KWARGS, return_matrix=True
    )
    return lazy, dense


requires_bundled_model = pytest.mark.skipif(
    not DEFAULT_MODEL.exists(), reason="multitask model not bundled"
)


@requires_bundled_model
def test_factored_is_returned_only_when_asked_for():
    """The default stays an ndarray; the factors come only on request."""
    lazy, dense = _matrices()
    assert isinstance(lazy, FactoredPredictionMatrix)
    assert isinstance(dense, np.ndarray)
    assert lazy.shape == dense.shape
    assert lazy.ndim == dense.ndim == 2
    assert len(lazy) == len(PEPTIDES)


@requires_bundled_model
@pytest.mark.parametrize(
    "index",
    [
        np.s_[:],
        np.s_[np.array([0, 3, 7])],
        np.s_[2],
        np.s_[:, 2016],
        np.s_[:, [12, 2016, 4000]],
        np.s_[np.array([0, 3, 7]), 2016],
        np.s_[3, 2016],
    ],
    ids=["all", "rows", "one-row", "one-column", "columns", "rows-and-column", "scalar"],
)
def test_every_indexing_form_matches_the_dense_matrix(index):
    """Shape and values must agree, so a caller cannot tell which it was handed."""
    lazy, dense = _matrices()
    from_factors = np.asarray(lazy[index])
    from_dense = np.asarray(dense[index])
    assert from_factors.shape == from_dense.shape
    np.testing.assert_allclose(from_factors, from_dense, atol=1e-3)


@requires_bundled_model
def test_asarray_gives_the_dense_matrix():
    """Code that genuinely needs the matrix still gets it."""
    lazy, dense = _matrices()
    np.testing.assert_allclose(np.asarray(lazy), dense, atol=1e-3)


@requires_bundled_model
def test_factors_are_smaller_than_the_matrix_they_stand_for():
    """The point of the exercise: a run's factors cost a fraction of its predictions."""
    lazy, _ = _matrices()
    # Per peptide, rank floats instead of n_tasks floats.
    per_peptide_factored = lazy.rank * lazy.dtype.itemsize
    per_peptide_dense = lazy.shape[1] * lazy.dtype.itemsize
    assert per_peptide_dense / per_peptide_factored > 100


@requires_bundled_model
def test_calibration_accepts_it_as_a_head_source():
    """`as_head_matrix` must pass it through rather than densifying it."""
    from deeplc.calibration.multihead import as_head_matrix

    lazy, _ = _matrices()
    assert lazy.is_head_source is True
    assert as_head_matrix(lazy) is lazy


def test_single_task_models_are_not_factorable():
    """Only a multitask FactorHead has factors to keep."""
    assert _model_ops.supports_factored(DeepLCModel(n_heads=1)) is False


@requires_bundled_model
def test_a_head_finetuned_onto_one_setup_is_not_factorable():
    """`add_task` makes the head return one column, so there is nothing to factor."""
    model = _model_ops.load_model(DEFAULT_MODEL, device="cpu")
    assert _model_ops.supports_factored(model) is True
    model.head.add_task(torch.zeros(4))
    assert _model_ops.supports_factored(model) is False


def test_rank_mismatch_is_rejected():
    """Factors that cannot multiply are caught at construction, not at indexing."""
    with pytest.raises(ValueError, match="rank"):
        FactoredPredictionMatrix(
            np.zeros((4, 8), dtype=np.float32),
            np.zeros((3, 16), dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
