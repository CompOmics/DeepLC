import numpy as np
import pytest
from psm_utils import PSM, PSMList

from deeplc._reference_selection import (
    MAX_REFERENCE_PSMS,
    MIN_REFERENCE_PSMS,
    Q_VALUE_THRESHOLD,
    TOP_SCORE_FRACTION,
    _filter_targets_with_rt,
    _select_by_qvalue,
    _select_by_score,
    select_reference_psms,
)
from deeplc.exceptions import ReferenceSelectionError


def _make_psm_list(
    n,
    *,
    rts=True,
    scores=None,
    qvalues=None,
    decoy_indices=None,
    is_decoy_none=False,
) -> PSMList:
    """Build a PSMList of n PSMs.

    decoy_indices: set of indices that are decoys (is_decoy=True).
    is_decoy_none: if True, all PSMs get is_decoy=None instead.
    """
    decoy_indices = set(decoy_indices or [])
    psms = []
    for i in range(n):
        is_decoy = None if is_decoy_none else i in decoy_indices
        psms.append(
            PSM(
                spectrum_id=str(i),
                peptidoform="PEPTIDE/2",
                retention_time=float(i) if rts else None,
                score=scores[i] if scores is not None else None,
                qvalue=qvalues[i] if qvalues is not None else None,
                is_decoy=is_decoy,
            )
        )
    return PSMList(psm_list=psms)


class TestFilterTargetsWithRT:
    def test_excludes_decoys(self):
        psm_list = _make_psm_list(5, decoy_indices={2, 4})
        result = _filter_targets_with_rt(psm_list)
        assert len(result) == 3

    def test_excludes_psms_without_rt(self):
        psm_list = _make_psm_list(5, rts=False)
        result = _filter_targets_with_rt(psm_list)
        assert len(result) == 0

    def test_excludes_psms_with_nan_rt(self):
        psms = [
            PSM(spectrum_id="0", peptidoform="PEPTIDE/2", retention_time=float("nan")),
            PSM(spectrum_id="1", peptidoform="PEPTIDE/2", retention_time=5.0),
        ]
        result = _filter_targets_with_rt(PSMList(psm_list=psms))
        assert len(result) == 1

    def test_includes_psms_where_is_decoy_is_none(self):
        psm_list = _make_psm_list(5, is_decoy_none=True)
        result = _filter_targets_with_rt(psm_list)
        assert len(result) == 5


class TestSelectByQvalue:
    def test_selects_below_threshold(self):
        qvalues = [0.001, 0.005, 0.02, 0.5]
        psm_list = _make_psm_list(4, qvalues=qvalues)
        result = _select_by_qvalue(psm_list)
        assert len(result) == 2

    def test_selects_at_threshold_boundary(self):
        # Confirms the <= fix (was strict <)
        qvalues = [Q_VALUE_THRESHOLD, Q_VALUE_THRESHOLD + 1e-9]
        psm_list = _make_psm_list(2, qvalues=qvalues)
        result = _select_by_qvalue(psm_list)
        assert len(result) == 1
        assert result[0].qvalue == pytest.approx(Q_VALUE_THRESHOLD)

    def test_none_qvalue_treated_as_inf(self):
        psm_list = _make_psm_list(3, qvalues=[0.001, None, 0.005])
        result = _select_by_qvalue(psm_list)
        assert len(result) == 2


class TestSelectByScore:
    def test_selects_top_fraction(self):
        n = 1000
        scores = list(range(n))
        psm_list = _make_psm_list(n, scores=scores)
        result = _select_by_score(psm_list)
        expected = max(MIN_REFERENCE_PSMS, min(MAX_REFERENCE_PSMS, int(n * TOP_SCORE_FRACTION)))
        assert len(result) == expected

    def test_clamps_to_min(self):
        # Small list: result should be clamped to MIN_REFERENCE_PSMS
        n = MIN_REFERENCE_PSMS + 1
        scores = list(range(n))
        psm_list = _make_psm_list(n, scores=scores)
        result = _select_by_score(psm_list)
        assert len(result) == MIN_REFERENCE_PSMS

    def test_clamps_to_max(self):
        # Large list: result should be clamped to MAX_REFERENCE_PSMS
        n = MAX_REFERENCE_PSMS * 10
        scores = list(range(n))
        psm_list = _make_psm_list(n, scores=scores)
        result = _select_by_score(psm_list)
        assert len(result) == MAX_REFERENCE_PSMS

    def test_selects_highest_scores(self):
        scores = [1.0, 5.0, 3.0, 2.0, 4.0] + [0.0] * 200
        psm_list = _make_psm_list(len(scores), scores=scores)
        result = _select_by_score(psm_list)
        result_scores = sorted(
            [psm.score for psm in result.psm_list if psm.score is not None], reverse=True
        )
        # Top score must be 5.0
        assert result_scores[0] == pytest.approx(5.0)


class TestSelectReferencePSMs:
    def test_strategy_qvalues(self):
        # 100 targets with low qvalues, enough to pass MIN threshold
        qvalues = [0.001] * 60 + [0.5] * 40
        psm_list = _make_psm_list(100, qvalues=qvalues)
        result = select_reference_psms(psm_list)
        assert len(result) >= MIN_REFERENCE_PSMS
        assert all(
            psm.qvalue is not None and psm.qvalue <= Q_VALUE_THRESHOLD for psm in result.psm_list
        )

    def test_strategy_tda(self):
        # Targets score 1.0–0.5, decoys score 0.4–0.0 — no overlap, all targets get q-value=0
        n_targets = 100
        n_decoys = 20
        target_scores = list(np.linspace(1.0, 0.5, n_targets))
        decoy_scores = list(np.linspace(0.4, 0.0, n_decoys))
        scores = target_scores + decoy_scores
        decoy_indices = set(range(n_targets, n_targets + n_decoys))
        psm_list = _make_psm_list(n_targets + n_decoys, scores=scores, decoy_indices=decoy_indices)
        result = select_reference_psms(psm_list)
        assert len(result) >= MIN_REFERENCE_PSMS
        assert all(not psm.is_decoy for psm in result.psm_list)

    def test_strategy_scores_only(self):
        # is_decoy=None so has_decoy_labels=False, routing to score-only path
        n = 1000
        scores = list(np.linspace(1.0, 0.0, n))
        psm_list = _make_psm_list(n, scores=scores, is_decoy_none=True)
        result = select_reference_psms(psm_list)
        assert MIN_REFERENCE_PSMS <= len(result) <= MAX_REFERENCE_PSMS

    def test_raises_when_no_scores_or_qvalues(self):
        psm_list = _make_psm_list(100)
        with pytest.raises(ReferenceSelectionError):
            select_reference_psms(psm_list)

    def test_raises_when_no_targets_with_rt(self):
        psm_list = _make_psm_list(100, rts=False, qvalues=[0.001] * 100)
        with pytest.raises(ReferenceSelectionError):
            select_reference_psms(psm_list)

    def test_raises_when_too_few_psms_survive(self):
        # Only 5 PSMs pass the qvalue threshold — below MIN_REFERENCE_PSMS
        qvalues = [0.001] * 5 + [0.5] * 95
        psm_list = _make_psm_list(100, qvalues=qvalues)
        with pytest.raises(ReferenceSelectionError):
            select_reference_psms(psm_list)
