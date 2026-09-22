from collections.abc import Sequence

import numpy as np
from psm_utils import PSM, PSMList

import deeplc.core

# A small set of real peptides with varied lengths and compositions to ensure
# model predictions span a range (needed for calibration to be non-degenerate).
_PEPTIDES = [
    "AAGPSLSHTSGGTQSK/2",
    "AGFAGDDAPR/2",
    "AIQEYNQDK/2",
    "AAYFGILEK/2",
    "ADTQLDESSEQIDEEELTSK/2",
    "AHQVVEDGYEFFAK/2",
    "ALDQFVNFSEQK/2",
    "AAPFSPAEK/2",
    "VGAHAGEYGAEALER/2",
    "LNLSPLGEEMR/2",
]


def _make_psm_list(
    peptides: list[str],
    rts: Sequence[float] | None = None,
    qvalues: Sequence[float | None] | None = None,
) -> PSMList:
    psms = []
    for i, pf in enumerate(peptides):
        psms.append(
            PSM(
                spectrum_id=str(i),
                peptidoform=pf,
                retention_time=rts[i] if rts else None,
                qvalue=qvalues[i] if qvalues else None,
            )
        )
    return PSMList(psm_list=psms)


def test_predict_and_calibrate_with_explicit_reference():
    psm_list = _make_psm_list(_PEPTIDES)
    reference = _make_psm_list(
        _PEPTIDES,
        rts=[float(i * 3) for i in range(len(_PEPTIDES))],
    )
    result = deeplc.core.predict_and_calibrate(psm_list, psm_list_reference=reference)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(_PEPTIDES),)


def test_predict_and_calibrate_returns_calibrated_values_differ_from_raw():
    raw = deeplc.core.predict(_make_psm_list(_PEPTIDES))

    reference = _make_psm_list(
        _PEPTIDES,
        # Observed RTs are 10x model output range — forces a non-trivial calibration
        rts=[float(i * 10) for i in range(len(_PEPTIDES))],
    )
    calibrated = deeplc.core.predict_and_calibrate(
        _make_psm_list(_PEPTIDES), psm_list_reference=reference
    )
    assert not np.allclose(raw, calibrated)


def test_predict_returns_matrix_when_flag_set():
    result = deeplc.core.predict(_make_psm_list(_PEPTIDES), return_matrix=True)
    assert result.ndim == 2
    assert result.shape[0] == len(_PEPTIDES)
    assert result.shape[1] > 1


def test_default_model_is_the_flexcnn_multitask_model():
    """Since 4.1.1 the fused-trunk model of 6,543 setups is what a bare call loads."""
    assert deeplc.core.DEFAULT_MODEL == deeplc.core.FLEXCNN_MULTITASK_MODEL
    assert deeplc.core.DEFAULT_MODEL.name == "multitask_flexcnn_model.pt"
    assert deeplc.core.LEGACY_MULTITASK_MODEL.name == "multitask_model.pt"
    assert deeplc.core.LEGACY_MULTITASK_MODEL.exists()


def test_uncalibrated_predict_reports_the_default_setup():
    """
    A bare ``predict`` returns the column of :data:`DEFAULT_TASK_NAME`, not head 0.

    The default model lists thousands of setups, and head 0 is whichever sorted first.
    The PXD005573 setup keeps uncalibrated output on the gradient DeepLC 1.x to 3.x
    reported, so downstream code that never calibrated sees comparable numbers.
    """
    model = deeplc.core._model_ops.load_model(deeplc.core.DEFAULT_MODEL, device="cpu")
    idx = list(model.task_names).index(deeplc.core.DEFAULT_TASK_NAME)
    assert idx != 0

    psm_list = _make_psm_list(_PEPTIDES)
    single = deeplc.core.predict(psm_list, predict_kwargs={"device": "cpu"})
    matrix = deeplc.core.predict(psm_list, return_matrix=True, predict_kwargs={"device": "cpu"})
    np.testing.assert_allclose(single, matrix[:, idx], rtol=1e-5, atol=1e-4)
    assert not np.allclose(single, matrix[:, 0])
    assert np.isfinite(single).all()


def test_legacy_multitask_model_still_loads_and_predicts():
    """The 4.0 default remains bundled and usable when pinned explicitly."""
    result = deeplc.core.predict(
        _make_psm_list(_PEPTIDES),
        model=deeplc.core.LEGACY_MULTITASK_MODEL,
        predict_kwargs={"device": "cpu"},
    )
    assert result.shape == (len(_PEPTIDES),)
    assert np.isfinite(result).all()


def test_predict_and_calibrate_auto_selects_reference():
    # 200 PSMs cycling through _PEPTIDES; 100 with qvalue<=0.01, 100 with qvalue=1.0.
    # auto-selection picks the 100 low-qvalue PSMs as reference.
    n = 200
    peptides = [_PEPTIDES[i % len(_PEPTIDES)] for i in range(n)]
    qvalues = [0.001 if i < 100 else 1.0 for i in range(n)]
    rts = [float(i) for i in range(n)]

    psm_list = _make_psm_list(peptides, rts=rts, qvalues=qvalues)
    result = deeplc.core.predict_and_calibrate(psm_list)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n,)


def test_predict_deduplicates_repeated_peptidoforms():
    """A peptidoform repeated across PSMs gets one prediction, repeated in place."""
    repeated = [p for p in _PEPTIDES for _ in range(3)]
    predictions = deeplc.core.predict(_make_psm_list(repeated))

    assert predictions.shape == (len(repeated),)
    once = deeplc.core.predict(_make_psm_list(_PEPTIDES))
    np.testing.assert_allclose(predictions, np.repeat(once, 3), rtol=0, atol=1e-4)


def test_predict_is_unchanged_when_every_peptidoform_is_distinct():
    """The deduplication must not disturb the ordinary path."""
    psm_list = _make_psm_list(_PEPTIDES)
    predictions = deeplc.core.predict(psm_list)

    assert predictions.shape == (len(_PEPTIDES),)
    assert len(set(np.round(predictions, 6))) > 1


def test_predict_shares_a_prediction_across_charge_states():
    """Charge reaches no feature the model reads, so charge states are one peptidoform."""
    charges = ["AGFAGDDAPR", "AGFAGDDAPR/2", "AGFAGDDAPR/3", "AIQEYNQDK/2"]
    predictions = deeplc.core.predict(_make_psm_list(charges))

    assert predictions[0] == predictions[1] == predictions[2]
    assert predictions[3] != predictions[0]


def test_predict_matrix_keeps_the_callers_rows_when_deduplicating():
    """``return_matrix`` still reports one row per PSM, in the caller's order."""
    repeated = ["AGFAGDDAPR/2", "AIQEYNQDK/2", "AGFAGDDAPR/2", "AAYFGILEK/2", "AIQEYNQDK/2"]
    matrix = np.asarray(deeplc.core.predict(_make_psm_list(repeated), return_matrix=True))

    assert matrix.shape[0] == len(repeated)
    np.testing.assert_array_equal(matrix[0], matrix[2])
    np.testing.assert_array_equal(matrix[1], matrix[4])
    assert not np.array_equal(matrix[0], matrix[1])
