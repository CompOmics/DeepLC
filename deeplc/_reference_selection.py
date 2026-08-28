"""Automatic reference PSM selection for auto-calibration."""

import logging

import numpy as np
from psm_utils import PSMList

from deeplc.exceptions import ReferenceSelectionError

LOGGER = logging.getLogger(__name__)

Q_VALUE_THRESHOLD = 0.01
MIN_REFERENCE_PSMS = 50
MAX_REFERENCE_PSMS = 500
TOP_SCORE_FRACTION = 0.2


def select_reference_psms(psm_list: PSMList) -> PSMList:
    """
    Select the best PSMs from a PSMList to use as calibration reference.

    Selection strategy (in priority order):

    1. q-values already present: filter targets with observed RT by q-value threshold.
    2. Target-decoy labels and scores present: compute q-values, then filter by threshold.
    3. Scores only: take top fraction by score, clamped to [MIN, MAX] count.
    4. None of the above: raise ReferenceSelectionError.

    Parameters
    ----------
    psm_list
        PSMList to select reference PSMs from. Must contain PSMs with observed retention
        times and either q-values, scores + target-decoy labels, or scores.

    Returns
    -------
    PSMList
        Subset of target PSMs with observed RT suitable for calibration.

    Raises
    ------
    ReferenceSelectionError
        If the PSMList lacks the required fields for any selection strategy.

    """
    # Filter to targets with observed RT
    candidates = _filter_targets_with_rt(psm_list)
    if len(candidates) == 0:
        raise ReferenceSelectionError(
            "No target PSMs with observed retention times found. "
            "Auto-calibration requires PSMs with retention times."
        )

    # Determine strategy
    has_qvalues = any(psm.qvalue is not None for psm in psm_list.psm_list)
    has_scores = any(psm.score is not None for psm in psm_list.psm_list)
    has_decoy_labels = any(psm.is_decoy is not None for psm in psm_list.psm_list)

    if has_qvalues:
        LOGGER.info("Selecting reference PSMs by pre-computed q-values.")
        reference = _select_by_qvalue(candidates)
    elif has_decoy_labels and has_scores:
        LOGGER.info("Computing q-values from target-decoy competition.")
        reference = _select_by_computed_qvalue(psm_list)
    elif has_scores:
        LOGGER.info("Selecting reference PSMs by top scores.")
        reference = _select_by_score(candidates)
    else:
        raise ReferenceSelectionError(
            "Cannot auto-select reference PSMs: input lacks q-values, scores, and "
            "target-decoy labels. Provide a separate reference file instead."
        )

    if len(reference) < MIN_REFERENCE_PSMS:
        raise ReferenceSelectionError(
            f"Only {len(reference)} reference PSMs selected, but at least "
            f"{MIN_REFERENCE_PSMS} are required for calibration."
        )

    LOGGER.info("Selected %d reference PSMs.", len(reference))
    return reference


def _filter_targets_with_rt(psm_list: PSMList) -> PSMList:
    """Filter to non-decoy PSMs with observed retention times."""
    is_decoy = np.array(
        [psm.is_decoy if psm.is_decoy is not None else False for psm in psm_list.psm_list]
    )
    has_rt = np.array(
        [
            psm.retention_time is not None and not np.isnan(psm.retention_time)
            for psm in psm_list.psm_list
        ]
    )
    mask = ~is_decoy & has_rt
    return psm_list[mask]


def _select_by_qvalue(candidates: PSMList) -> PSMList:
    """Select PSMs with q-value below threshold."""
    qvalues = np.array(
        [psm.qvalue if psm.qvalue is not None else np.inf for psm in candidates.psm_list]
    )
    mask = qvalues <= Q_VALUE_THRESHOLD
    return candidates[mask]


def _select_by_computed_qvalue(psm_list: PSMList) -> PSMList:
    """
    Compute q-values from target-decoy competition, then select by threshold.

    Note: this modifies ``psm_list`` in-place by assigning computed q-values.
    """
    psm_list.calculate_qvalues(reverse=True)
    candidates = _filter_targets_with_rt(psm_list)
    return _select_by_qvalue(candidates)


def _select_by_score(candidates: PSMList) -> PSMList:
    """Select top fraction of PSMs by score, clamped to [MIN, MAX] count."""
    scores = np.array(
        [psm.score if psm.score is not None else -np.inf for psm in candidates.psm_list]
    )
    n_total = len(candidates)
    n_select = int(n_total * TOP_SCORE_FRACTION)
    n_select = max(MIN_REFERENCE_PSMS, min(MAX_REFERENCE_PSMS, n_select))
    n_select = min(n_select, n_total)

    top_indices = np.argsort(scores)[::-1][:n_select]
    return candidates[top_indices]


def deduplicate_psms(psm_list: PSMList, ignore_charge: bool = True) -> PSMList:
    """
    Keep one PSM per peptidoform, the first occurrence in the list.

    A reference set built from a search result usually contains the same peptidoform
    identified in many spectra, with a different observed retention time each time. Those
    repeats do not add information about the gradient: they give the calibration one x value
    with several conflicting y values, and their number is what a spline fit weighs, so a
    peptidoform seen 261 times counts 261 times while one seen once counts once. On a
    reported MS2Rescore case, 6,331 reference PSMs collapsed to 2,623 peptidoforms, and the
    observed retention times of one repeated peptidoform differed by two thirds of the whole
    observed range.

    Only the first observation is kept, which is what a caller who has already sorted or
    filtered its PSMs expects, and it makes the result independent of how many times a
    peptidoform happened to be identified.

    Parameters
    ----------
    psm_list
        PSMs to deduplicate.
    ignore_charge
        Treat charge states of the same peptidoform as duplicates (default). Retention time
        does not depend on precursor charge, so the charge states of one peptidoform are
        repeats of the same measurement. Set to False to keep one PSM per peptidoform *and*
        charge.

    Returns
    -------
    PSMList
        The first PSM of every peptidoform, in the original order.

    """
    seen: set[str] = set()
    keep = np.zeros(len(psm_list), dtype=bool)
    for i, psm in enumerate(psm_list.psm_list):
        key = str(psm.peptidoform)
        if ignore_charge:
            key = key.rsplit("/", 1)[0]
        if key not in seen:
            seen.add(key)
            keep[i] = True

    n_dropped = int((~keep).sum())
    if n_dropped:
        LOGGER.info(
            "Deduplicated the reference: %d of %d PSMs are repeats of a peptidoform already "
            "in the set and were dropped, leaving %d. Pass deduplicate_reference=False to "
            "keep them.",
            n_dropped,
            len(psm_list),
            int(keep.sum()),
        )
        _warn_on_conflicting_retention_times(psm_list, keep, ignore_charge)
    return psm_list[keep]


def _warn_on_conflicting_retention_times(
    psm_list: PSMList, keep: np.ndarray, ignore_charge: bool
) -> None:
    """
    Report the largest retention-time disagreement among the dropped repeats.

    A small spread is ordinary chromatographic jitter; a spread of the order of the gradient
    means the repeats are not the same elution event, so the reference was built from PSMs
    that a search would normally not put in one calibration set (decoys, low-scoring hits, or
    several runs pooled into one file). Worth saying out loud, because deduplication then
    hides a data problem rather than solving it.
    """
    by_key: dict[str, list[float]] = {}
    for psm in psm_list.psm_list:
        rt = psm.retention_time
        if rt is None or np.isnan(rt):
            continue
        key = str(psm.peptidoform)
        if ignore_charge:
            key = key.rsplit("/", 1)[0]
        by_key.setdefault(key, []).append(float(rt))

    spreads = [max(v) - min(v) for v in by_key.values() if len(v) > 1]
    if not spreads:
        return
    worst = max(spreads)
    observed = [
        float(psm.retention_time)
        for psm in psm_list.psm_list
        if psm.retention_time is not None and not np.isnan(psm.retention_time)
    ]
    span = (max(observed) - min(observed)) if observed else 0.0
    LOGGER.log(
        logging.WARNING if span and worst > 0.25 * span else logging.INFO,
        "Repeated peptidoforms disagreed on the observed retention time by up to %.2f "
        "(median %.2f) against an observed range of %.2f. A large disagreement means the "
        "repeats are not the same elution event: check whether the reference mixes runs or "
        "includes low-confidence PSMs.",
        worst,
        float(np.median(spreads)),
        span,
    )
