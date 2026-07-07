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
