"""
Prediction reports: provenance and uncertainty next to every retention time.

A plain prediction is a number with no way to tell whether the model has seen the peptidoform,
merely something like it, or nothing like it, and no statement of how far off it may be. The
report answers those three questions per PSM:

- **membership**: is the peptidoform an exact match to the reference the calibration or
  fine-tuning used, and, when a training index is available, to the corpus the bundled model
  was trained on, or to the training sets of the setups the calibration selected;
- **novelty**: the Levenshtein distance from the stripped sequence to the closest reference
  sequence (and to the closest training sequence, when the index is available);
- **uncertainty**: a conformal prediction interval calibrated on the reference.

The interval comes from cross-fitted split-conformal prediction: the reference is split into
folds, each fold is predicted by a calibration fitted on the other folds, and the interval
half-width is a finite-sample quantile of those honest |residuals|, taken per predicted-RT bin
because peak width varies along a gradient. On eight PRIDE setups no DeepLC model was trained
on, the empirical coverage of the 90 % interval was 0.88 to 0.96 per setup (median 0.91), with
widths from 4 % of the gradient on well-behaved setups to 79 % on a run that pools several
fractions, which is what an honest interval looks like there. Coverage is marginal, not
per-peptide: on average over peptides like the reference, not for each one individually.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from psm_utils import PSM, Peptidoform, PSMList

from deeplc import core
from deeplc._reference_selection import deduplicate_psms, select_reference_psms
from deeplc.calibration import Calibration, SplineTransformerCalibration

LOGGER = logging.getLogger(__name__)

#: Bins for the RT-dependent interval width, and the minimum honest residuals a bin needs
#: before it is trusted over the global quantile.
_N_RT_BINS = 5
_MIN_RESIDUALS_PER_BIN = 40
_N_FOLDS = 5


def canonical_peptidoform_key(peptidoform: Peptidoform | str) -> str:
    """
    Build the identifier under which a peptidoform appears in the multitask training corpus.

    ``SEQUENCE|`` followed by position-sorted ``pos|U:<unimod id>`` pairs, positions in peprec
    convention (1-based, 0 for N-terminal, -1 for C-terminal). A modification without a Unimod
    accession contributes its lowercased name, matching how the corpus was built: an unmapped
    name still matches itself across sources instead of silently merging with another.
    """
    if isinstance(peptidoform, str):
        peptidoform = Peptidoform(peptidoform)

    def token(mod) -> str:
        accession = getattr(mod, "id", None)
        if accession is not None and str(accession).isdigit():
            return f"U:{accession}"
        name = getattr(mod, "name", None) or str(mod)
        return str(name).lower()

    pairs: list[tuple[int, str]] = []
    n_term = peptidoform.properties.get("n_term")
    if n_term:
        pairs += [(0, token(mod)) for mod in n_term]
    c_term = peptidoform.properties.get("c_term")
    if c_term:
        pairs += [(-1, token(mod)) for mod in c_term]
    for position, (_, mods) in enumerate(peptidoform.parsed_sequence, start=1):
        if mods:
            pairs += [(position, token(mod)) for mod in mods]
    pairs.sort()
    mods_text = "|".join(f"{position}|{tok}" for position, tok in pairs)
    return f"{peptidoform.sequence}|{mods_text}"


class TrainingIndex:
    """
    Memory-mapped index of the corpus behind the bundled multitask model.

    Built offline from the training cache (10,105,640 canonical peptidoform keys and their
    65,139,832 peptidoform-setup observations over 6,543 setups) and loaded from a directory:
    ``key_hashes.npy`` (sorted xxh3-64 of the canonical keys), ``task_indptr.npy`` /
    ``task_cols.npy`` (which setups each peptidoform was observed in), ``sequences.npy`` /
    ``seq_lengths.npy`` (unique stripped sequences, for edit distances) and ``meta.json``.

    Everything is memory-mapped, so opening the index costs nothing until it is used.
    """

    def __init__(self, path: PathLike | str) -> None:
        """Open a training index directory."""
        self.path = Path(path)
        meta_file = self.path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"{self.path} is not a training index (no meta.json). It is built offline "
                "from the training cache and distributed separately from the package."
            )
        self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self._hashes = np.load(self.path / "key_hashes.npy", mmap_mode="r")
        self._indptr = np.load(self.path / "task_indptr.npy", mmap_mode="r")
        self._cols = np.load(self.path / "task_cols.npy", mmap_mode="r")
        self._sequences: np.ndarray | None = None
        self._seq_lengths: np.ndarray | None = None

    @staticmethod
    def _hash(keys: list[str]) -> np.ndarray:
        try:
            from xxhash import xxh3_64_intdigest as digest
        except ImportError:
            from hashlib import blake2b

            def digest(text: str) -> int:
                return int.from_bytes(blake2b(text.encode(), digest_size=8).digest(), "little")

        return np.array([digest(key) for key in keys], dtype=np.uint64)

    def _rows(self, keys: list[str]) -> np.ndarray:
        """Index of each key in the sorted hash array, or -1 when absent."""
        hashes = self._hash(keys)
        position = np.searchsorted(self._hashes, hashes)
        position = np.clip(position, 0, len(self._hashes) - 1)
        found = self._hashes[position] == hashes
        return np.where(found, position, -1)

    def contains(self, keys: list[str]) -> np.ndarray:
        """Whether each canonical key occurs anywhere in the training corpus."""
        return self._rows(keys) >= 0

    def contains_in_tasks(self, keys: list[str], task_idx: np.ndarray) -> np.ndarray:
        """
        Whether each key was observed in at least one of the given setups.

        Setup ids the index does not know (a model with more heads than the corpus the index
        was built from) are ignored: they cannot contribute a membership either way.
        """
        n_tasks = int(self.meta["n_tasks"])
        task_idx = np.asarray(task_idx, dtype=int)
        known = task_idx[(task_idx >= 0) & (task_idx < n_tasks)]
        if len(known) < len(task_idx):
            LOGGER.warning(
                "%d of %d selected setups are outside this training index (%d setups); "
                "does the index belong to this model?",
                len(task_idx) - len(known),
                len(task_idx),
                n_tasks,
            )
        wanted = np.zeros(n_tasks, dtype=bool)
        wanted[known] = True
        rows = self._rows(keys)
        out = np.zeros(len(keys), dtype=bool)
        for i, row in enumerate(rows):
            if row < 0:
                continue
            cols = self._cols[self._indptr[row] : self._indptr[row + 1]]
            out[i] = bool(wanted[cols].any())
        return out

    def distance_to_training(self, sequences: list[str], max_distance: int = 10) -> np.ndarray:
        """
        Levenshtein distance from each stripped sequence to the closest training sequence.

        Distances are exact up to ``max_distance`` and reported as ``max_distance + 1`` beyond
        it. The cap is what keeps this fast: exact matches are a set lookup, near matches a
        length-banded cutoff search, and the expensive unbounded scan over millions of
        sequences never runs. Beyond ten edits the distance carries no usable signal anyway;
        on held-out setups the prediction error is flat in this distance.
        """
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz.process import cdist

        if self._sequences is None:
            blob = (self.path / "sequences.txt").read_bytes().decode("ascii")
            self._sequences = np.array(blob.split(chr(10)), dtype=object)
            self._seq_lengths = np.load(self.path / "seq_lengths.npy")
        unique, inverse = np.unique(np.asarray(sequences, dtype=object), return_inverse=True)
        exact = np.isin(unique, self._sequences)
        per_unique = np.full(len(unique), -1, dtype=np.int32)
        per_unique[exact] = 0
        todo = np.flatnonzero(~exact)
        if len(todo) == 0:
            return per_unique[inverse]
        lengths = np.array([len(unique[i]) for i in todo])
        band = (self._seq_lengths >= lengths.min() - max_distance) & (
            self._seq_lengths <= lengths.max() + max_distance
        )
        candidates = self._sequences[band]
        distance = cdist(
            [unique[i] for i in todo],
            candidates.tolist(),
            scorer=Levenshtein.distance,
            score_cutoff=max_distance,
            workers=-1,
        )
        # rapidfuzz reports cutoff + 1 for everything above the cutoff, which is exactly the
        # capped value this method promises
        per_unique[todo] = distance.min(axis=1)
        return per_unique[inverse]


@dataclass
class _ConformalInterval:
    """RT-binned conformal half-widths, fitted on honest reference residuals."""

    coverage: float
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    half_width: np.ndarray = field(default_factory=lambda: np.array([]))

    @staticmethod
    def _finite_sample_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
        n = len(abs_residuals)
        rank = min(int(np.ceil((n + 1) * coverage)), n)
        return float(np.sort(abs_residuals)[rank - 1])

    @classmethod
    def fit(
        cls, predicted: np.ndarray, residuals: np.ndarray, coverage: float
    ) -> _ConformalInterval:
        """Per-RT-bin conformal quantiles with a global fallback for thin bins."""
        absolute = np.abs(residuals)
        overall = cls._finite_sample_quantile(absolute, coverage)
        edges = np.quantile(predicted, np.linspace(0, 1, _N_RT_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bins = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, _N_RT_BINS - 1)
        half_width = np.full(_N_RT_BINS, overall)
        for b in range(_N_RT_BINS):
            mask = bins == b
            if int(mask.sum()) >= _MIN_RESIDUALS_PER_BIN:
                half_width[b] = cls._finite_sample_quantile(absolute[mask], coverage)
        return cls(coverage=coverage, edges=edges, half_width=half_width)

    def widths(self, predicted: np.ndarray) -> np.ndarray:
        """Interval half-width for each prediction."""
        bins = np.clip(np.searchsorted(self.edges, predicted, side="right") - 1, 0, _N_RT_BINS - 1)
        return self.half_width[bins]


def _crossfit_residuals(
    y_reference: np.ndarray,
    matrix_reference: np.ndarray,
    calibration_template: Calibration,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Honest reference residuals: each fold predicted by a calibration fitted without it.

    Returns (cross-fitted predictions, residuals), aligned with the reference order. The
    template is re-instantiated per fold with ``type(...)()`` semantics via a deep copy of its
    construction parameters, so a fitted calibration is never reused across folds.
    """
    import copy

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_reference))
    folds = np.array_split(order, min(_N_FOLDS, max(2, len(y_reference) // 25)))
    predicted = np.empty(len(y_reference))
    for i, fold in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        calibration = copy.deepcopy(calibration_template)
        if getattr(calibration, "uses_all_heads", False):
            calibration.fit(target=y_reference[train], source=matrix_reference[train])
            predicted[fold] = calibration.transform(matrix_reference[fold])
        else:
            head = core._best_correlating_head(matrix_reference[train], y_reference[train])
            calibration.selected_model_head = head
            calibration.fit(
                target=y_reference[train].astype(np.float32),
                source=matrix_reference[train][:, head].astype(np.float32),
            )
            predicted[fold] = np.asarray(
                calibration.transform(matrix_reference[fold][:, head].astype(np.float32)),
                dtype=float,
            )
    return predicted, y_reference - predicted


def prediction_report(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList | list[PSM | Peptidoform | str] | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | None = None,
    coverage: float = 0.90,
    training_index: TrainingIndex | PathLike | str | None = None,
    predict_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Predict with calibration and report provenance and uncertainty per PSM.

    Parameters
    ----------
    psm_list
        PSMs to predict retention times for.
    psm_list_reference
        Reference for calibration; auto-selected from ``psm_list`` when None, as in
        :func:`deeplc.predict_and_calibrate`.
    model
        Trained model or path; the bundled multitask model when None.
    calibration
        Unfitted calibration to use; :class:`SplineTransformerCalibration` when None. Pass
        :class:`~deeplc.calibration.MultiHeadRidgeCalibration` to combine setups, in which case
        the membership column covers every selected head.
    coverage
        Nominal coverage of the conformal interval (marginal, on peptides exchangeable with
        the reference). 0.90 by default.
    training_index
        A :class:`TrainingIndex` or a path to one. Without it, the columns about the training
        corpus are omitted and the report is limited to the reference.
    predict_kwargs
        Passed to the prediction function (``{"device": "cpu"}`` and the like).

    Returns
    -------
    pd.DataFrame
        One row per input PSM, in order: ``peptidoform``, ``predicted_rt``, ``ci_lower``,
        ``ci_upper`` (conformal at ``coverage``), ``observed_rt`` (when present),
        ``in_reference``, ``dist_to_reference`` and, with a training index,
        ``in_training``, ``dist_to_training`` and ``in_selected_heads_training``.

    """
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    parsed = core._parse_psms(psm_list)
    if psm_list_reference is None:
        reference = select_reference_psms(parsed)
    else:
        reference = core._parse_psms(psm_list_reference)
    reference = deduplicate_psms(reference)

    if calibration is None:
        calibration = SplineTransformerCalibration()
    if calibration.is_fitted:
        raise ValueError(
            "prediction_report fits the calibration itself (it also needs cross-fitted "
            "residuals for the interval); pass an unfitted calibration."
        )

    # one matrix for the reference, one for the queries; everything below reuses them
    matrix_reference = core.predict(
        reference, model=model, predict_kwargs=predict_kwargs, return_matrix=True
    ).astype(np.float64)
    matrix_query = core.predict(
        parsed, model=model, predict_kwargs=predict_kwargs, return_matrix=True
    ).astype(np.float64)
    y_reference = np.array(reference["retention_time"], dtype=np.float64)

    import copy

    template = copy.deepcopy(calibration)
    if getattr(calibration, "uses_all_heads", False):
        calibration.fit(target=y_reference, source=matrix_reference)
        predicted = calibration.transform(matrix_query)
        selected_heads = np.asarray(calibration._head_idx, dtype=int)
    else:
        head = core._best_correlating_head(matrix_reference, y_reference)
        calibration.selected_model_head = head
        calibration.fit(
            target=y_reference.astype(np.float32),
            source=matrix_reference[:, head].astype(np.float32),
        )
        predicted = np.asarray(
            calibration.transform(matrix_query[:, head].astype(np.float32)), dtype=float
        )
        selected_heads = np.array([head], dtype=int)

    cross_predicted, residuals = _crossfit_residuals(y_reference, matrix_reference, template)
    interval = _ConformalInterval.fit(cross_predicted, residuals, coverage)
    half_width = interval.widths(np.asarray(predicted, dtype=float))

    # membership and novelty against the reference
    reference_keys = {canonical_peptidoform_key(psm.peptidoform) for psm in reference.psm_list}
    query_keys = [canonical_peptidoform_key(psm.peptidoform) for psm in parsed.psm_list]
    in_reference = np.array([key in reference_keys for key in query_keys])

    reference_sequences = sorted({psm.peptidoform.sequence for psm in reference.psm_list})
    query_sequences = [psm.peptidoform.sequence for psm in parsed.psm_list]
    dist_to_reference = cdist(
        query_sequences, reference_sequences, scorer=Levenshtein.distance, workers=-1
    ).min(axis=1)

    observed = [psm.retention_time for psm in parsed.psm_list]
    frame = pd.DataFrame(
        {
            "peptidoform": [str(psm.peptidoform) for psm in parsed.psm_list],
            "predicted_rt": np.asarray(predicted, dtype=float),
            "ci_lower": np.asarray(predicted, dtype=float) - half_width,
            "ci_upper": np.asarray(predicted, dtype=float) + half_width,
            "observed_rt": [rt if rt is not None else np.nan for rt in observed],
            "in_reference": in_reference,
            "dist_to_reference": dist_to_reference.astype(int),
        }
    )
    frame.attrs["coverage"] = coverage
    frame.attrs["selected_heads"] = selected_heads.tolist()

    if training_index is not None:
        if not isinstance(training_index, TrainingIndex):
            training_index = TrainingIndex(training_index)
        frame["in_training"] = training_index.contains(query_keys)
        frame["in_selected_heads_training"] = training_index.contains_in_tasks(
            query_keys, selected_heads
        )
        frame["dist_to_training"] = training_index.distance_to_training(query_sequences)
        LOGGER.info(
            "%d of %d peptidoforms are in the training corpus, %d in the %d selected setups.",
            int(frame["in_training"].sum()),
            len(frame),
            int(frame["in_selected_heads_training"].sum()),
            len(selected_heads),
        )
    return frame
