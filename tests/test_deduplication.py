"""Reference deduplication: one PSM per peptidoform, the first observation."""

from __future__ import annotations

import logging

import numpy as np
from psm_utils import PSM, PSMList

from deeplc import core
from deeplc._reference_selection import deduplicate_psms

_PEPTIDES = [
    "AAGPSLSHTSGGTQSK",
    "AGFAGDDAPR",
    "AIQEYNQDK",
    "AAYFGILEK",
    "ADTQLDESSEQIDEEELTSK",
    "AHQVVEDGYEFFAK",
    "ALDQFVNFSEQK",
    "AAPFSPAEK",
    "VGAHAGEYGAEALER",
    "LNLSPLGEEMR",
]


def _psms(pairs: list[tuple[str, float | None]], charge: int = 2) -> PSMList:
    """Build a PSMList from (peptide, retention time) pairs."""
    return PSMList(
        psm_list=[
            PSM(spectrum_id=str(i), peptidoform=f"{seq}/{charge}", retention_time=rt)
            for i, (seq, rt) in enumerate(pairs)
        ]
    )


def test_keeps_the_first_observation_of_each_peptidoform():
    """Repeats are dropped and the retention time of the first one survives."""
    psm_list = _psms([("PEPTIDEK", 10.0), ("PEPTIDEK", 40.0), ("ACDEFGHIK", 20.0)])

    deduplicated = deduplicate_psms(psm_list)

    assert [str(p.peptidoform) for p in deduplicated] == ["PEPTIDEK/2", "ACDEFGHIK/2"]
    assert [p.retention_time for p in deduplicated] == [10.0, 20.0]


def test_order_is_preserved():
    """The kept PSMs stay in the order they were given in."""
    psm_list = _psms([(s, float(i)) for i, s in enumerate(_PEPTIDES)])
    assert [str(p.peptidoform) for p in deduplicate_psms(psm_list)] == [
        str(p.peptidoform) for p in psm_list
    ]


def test_idempotent():
    """Deduplicating an already deduplicated list changes nothing."""
    psm_list = _psms([("PEPTIDEK", 10.0), ("PEPTIDEK", 40.0), ("ACDEFGHIK", 20.0)])
    once = deduplicate_psms(psm_list)
    assert len(deduplicate_psms(once)) == len(once)


def test_charge_states_are_duplicates_by_default():
    """Retention time does not depend on charge, so charge states are repeats."""
    psm_list = PSMList(
        psm_list=[
            PSM(spectrum_id="1", peptidoform="PEPTIDEK/2", retention_time=10.0),
            PSM(spectrum_id="2", peptidoform="PEPTIDEK/3", retention_time=11.0),
        ]
    )

    assert len(deduplicate_psms(psm_list)) == 1
    assert len(deduplicate_psms(psm_list, ignore_charge=False)) == 2


def test_modified_peptidoforms_are_not_duplicates():
    """A modification makes a different peptidoform, which elutes at a different time."""
    psm_list = PSMList(
        psm_list=[
            PSM(spectrum_id="1", peptidoform="PEPTM[Oxidation]IDEK/2", retention_time=10.0),
            PSM(spectrum_id="2", peptidoform="PEPTMIDEK/2", retention_time=12.0),
        ]
    )
    assert len(deduplicate_psms(psm_list)) == 2


def test_warns_when_repeats_disagree_on_the_retention_time(caplog):
    """
    A disagreement of the order of the gradient is a data problem, not jitter.

    Deduplication silently fixes the fit, so the case a user needs to hear about is when the
    repeated peptidoform was not the same elution event at all.
    """
    pairs = [(s, float(i)) for i, s in enumerate(_PEPTIDES)]
    pairs.append((_PEPTIDES[0], 500.0))  # same peptidoform, 500 minutes later

    with caplog.at_level(logging.WARNING, logger="deeplc._reference_selection"):
        deduplicate_psms(_psms(pairs))

    assert any("disagreed on the observed retention time" in r.message for r in caplog.records)


def test_missing_retention_times_do_not_break_the_report(caplog):
    """PSMs without an observed retention time are deduplicated like any other."""
    psm_list = _psms([("PEPTIDEK", None), ("PEPTIDEK", None), ("ACDEFGHIK", 20.0)])
    with caplog.at_level(logging.INFO, logger="deeplc._reference_selection"):
        assert len(deduplicate_psms(psm_list)) == 2


def _reference_with_duplicates() -> PSMList:
    """Ten peptidoforms on a clean gradient, each repeated once at a wrong retention time."""
    pairs = [(s, 5.0 + 3.0 * i) for i, s in enumerate(_PEPTIDES)]
    pairs += [(s, 100.0 - 2.0 * i) for i, s in enumerate(_PEPTIDES)]
    return _psms(pairs)


def test_calibrate_uses_the_first_observations_by_default():
    """
    ``calibrate`` fits on the deduplicated reference by default.

    The reference holds each peptidoform twice: once on a clean 5 to 32 minute gradient and
    once at a contradictory 82 to 100 minutes. A fit on the first observations must reproduce
    the clean gradient, and a fit on every PSM must sit well above it, pulled by the repeats.
    """
    reference = _reference_with_duplicates()
    targets = _psms([(s, None) for s in _PEPTIDES])
    predicted = core.predict(targets, return_matrix=True)

    def fitted_mean(deduplicate: bool) -> float:
        calibration = core.calibrate(
            reference, predict_kwargs={"device": "cpu"}, deduplicate_reference=deduplicate
        )
        head = calibration.selected_model_head or 0
        calibrated = calibration.transform(predicted[:, head])
        assert np.isfinite(calibrated).all()
        return float(calibrated.mean())

    clean_low, clean_high = 5.0, 5.0 + 3.0 * (len(_PEPTIDES) - 1)  # the first observations
    on, off = fitted_mean(True), fitted_mean(False)

    assert clean_low <= on <= clean_high, f"deduplicated fit {on:.1f} left the clean gradient"
    assert off > clean_high, f"fit on all PSMs {off:.1f} was not pulled above the gradient"


def test_predict_and_calibrate_forwards_the_parameter():
    """Both settings run end to end and give one prediction per input PSM."""
    psm_list = _psms([(s, None) for s in _PEPTIDES])
    reference = _reference_with_duplicates()

    on = core.predict_and_calibrate(
        psm_list, psm_list_reference=reference, predict_kwargs={"device": "cpu"}
    )
    off = core.predict_and_calibrate(
        psm_list,
        psm_list_reference=reference,
        predict_kwargs={"device": "cpu"},
        deduplicate_reference=False,
    )

    assert on.shape == off.shape == (len(_PEPTIDES),)
    assert not np.allclose(on, off)
