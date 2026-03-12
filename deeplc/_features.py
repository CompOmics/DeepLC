"""Feature extraction for DeepLC."""

# TODO: Consider ProForma fixed modifications (that are not applied yet) for feature extraction.

from __future__ import annotations

import logging
import warnings
from re import sub

import numpy as np
from psm_utils import Peptidoform
from pyteomics import mass

logger = logging.getLogger(__name__)


# fmt: off
DEFAULT_POSITIONS: set[int] = {0, 1, 2, 3, -1, -2, -3, -4}
DEFAULT_POSITIONS_POS: set[int] = {0, 1, 2, 3}
DEFAULT_POSITIONS_NEG: set[int] = {-1, -2, -3, -4}
DEFAULT_DICT_AA: dict[str, int] = {
    "K": 0, "R": 1, "P": 2, "T": 3, "N": 4, "A": 5, "Q": 6, "V": 7, "S": 8, "G": 9, "I": 10,
    "L": 11, "C": 12, "M": 13, "H": 14, "F": 15, "Y": 16, "W": 17, "E": 18, "D": 19,
}
DEFAULT_DICT_INDEX_POS: dict[str, int] = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}
DEFAULT_DICT_INDEX: dict[str, int] = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}
# fmt: on


def encode_peptidoform(
    peptidoform: Peptidoform | str,
    add_ccs_features: bool = False,
    padding_length: int = 60,
    positions: set[int] | None = None,
    positions_pos: set[int] | None = None,
    positions_neg: set[int] | None = None,
    dict_aa: dict[str, int] | None = None,
    dict_index_pos: dict[str, int] | None = None,
    dict_index: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract features from a single peptidoform.

    Parameters
    ----------
    peptidoform
        The peptidoform to encode, either as a Peptidoform object or a string.
    add_ccs_features
        Whether to include CCS features. Default is False.
    padding_length
        The maximum length of the sequence after padding. Default is 60.
    positions
        The positions to consider for feature extraction. Default is DEFAULT_POSITIONS.
    positions_pos
        The positive positions to consider for feature extraction. Default is
        DEFAULT_POSITIONS_POS.
    positions_neg
        The negative positions to consider for feature extraction. Default is
        DEFAULT_POSITIONS_NEG.
    dict_aa
        A dictionary mapping amino acids to indices. Default is DEFAULT_DICT_AA.
    dict_index_pos
        A dictionary mapping atoms to indices for the positional matrix. Default is
        DEFAULT_DICT_INDEX_POS.
    dict_index
        A dictionary mapping atoms to indices. Default is DEFAULT_DICT_INDEX.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary of Numpy arrays containing the extracted features.

    """
    positions = positions or DEFAULT_POSITIONS
    positions_pos = positions_pos or DEFAULT_POSITIONS_POS
    positions_neg = positions_neg or DEFAULT_POSITIONS_NEG
    dict_aa = dict_aa or DEFAULT_DICT_AA
    dict_index_pos = dict_index_pos or DEFAULT_DICT_INDEX_POS
    dict_index = dict_index or DEFAULT_DICT_INDEX

    if isinstance(peptidoform, str):
        peptidoform = Peptidoform(peptidoform)
    seq = peptidoform.sequence
    charge = peptidoform.precursor_charge
    seq, seq_len = _truncate_sequence(seq, padding_length)

    std_matrix = _fill_standard_matrix(seq, padding_length, dict_index)
    onehot_matrix = _fill_onehot_matrix(peptidoform.parsed_sequence, padding_length, dict_aa)
    pos_matrix = _fill_pos_matrix(
        seq, seq_len, positions_pos, positions_neg, dict_index, dict_index_pos
    )
    _apply_modifications(
        std_matrix,
        pos_matrix,
        peptidoform.parsed_sequence,
        seq_len,
        dict_index,
        dict_index_pos,
        positions,
    )
    _apply_terminal_modifications(
        std_matrix,
        pos_matrix,
        peptidoform,
        seq_len,
        dict_index,
        dict_index_pos,
        positions,
    )

    matrix_all = np.sum(std_matrix, axis=0)
    matrix_all = np.append(matrix_all, seq_len)
    if add_ccs_features:
        if not charge:
            raise ValueError(f"Peptidoform has no charge: {peptidoform}")
        matrix_all = np.append(matrix_all, (seq.count("H")) / seq_len)
        matrix_all = np.append(
            matrix_all, (seq.count("F") + seq.count("W") + seq.count("Y")) / seq_len
        )
        matrix_all = np.append(matrix_all, (seq.count("D") + seq.count("E")) / seq_len)
        matrix_all = np.append(matrix_all, (seq.count("K") + seq.count("R")) / seq_len)
        matrix_all = np.append(matrix_all, charge)

    matrix_sum = _compute_rolling_sum(std_matrix.T, n=2)[:, ::2].T

    matrix_global = np.concatenate([matrix_all, pos_matrix.flatten()])

    return {
        "matrix": std_matrix,
        "matrix_sum": matrix_sum,
        "matrix_global": matrix_global,
        "matrix_hc": onehot_matrix,
    }


def _truncate_sequence(seq: str, max_length: int) -> tuple[str, int]:
    """Truncate the sequence if it exceeds the max_length."""
    if len(seq) > max_length:
        warnings.warn(f"Truncating peptide (too long): {seq}", stacklevel=2)
        seq = seq[:max_length]
    return seq, len(seq)


def _fill_standard_matrix(seq: str, padding_length: int, dict_index: dict[str, int]) -> np.ndarray:
    """Fill the standard composition matrix using mass.std_aa_comp."""
    mat = np.zeros((padding_length, len(dict_index)), dtype=np.float16)
    for i, aa in enumerate(seq):
        for atom, value in mass.std_aa_comp[aa].items():
            try:
                mat[i, dict_index[atom]] = value
            except (KeyError, IndexError):
                warnings.warn(f"Skipping atom {atom} at pos {i}", stacklevel=2)
    return mat


def _fill_onehot_matrix(
    parsed_seq: list, padding_length: int, dict_aa: dict[str, int]
) -> np.ndarray:
    """Fill a one-hot matrix based on the parsed sequence tokens."""
    onehot = np.zeros((padding_length, len(dict_aa)), dtype=np.float16)
    for i, token in enumerate(parsed_seq):
        try:
            onehot[i, dict_aa[token[0]]] = 1.0
        except (KeyError, IndexError):
            warnings.warn(f"One-hot skip: {i} {token}", stacklevel=2)
    return onehot


def _fill_pos_matrix(
    seq: str,
    seq_len: int,
    positions_pos: set[int],
    positions_neg: set[int],
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
) -> np.ndarray:
    """Fill positional matrix for atoms at specific positions."""
    pos_total = positions_pos.union(positions_neg)
    pos_mat = np.zeros((max(pos_total) - min(pos_total) + 1, len(dict_index)), dtype=np.float16)
    # For positive positions
    for pos in positions_pos:
        try:
            aa = seq[pos]
        except Exception:
            warnings.warn(f"Unable to get pos {pos}", stacklevel=2)
            continue
        for atom, value in mass.std_aa_comp[aa].items():
            try:
                # shift index for matrix row since positions may be negative.
                pos_mat[pos - min(pos_total), dict_index_pos[atom]] = value
            except (KeyError, IndexError):
                warnings.warn(f"Pos matrix skip: {atom} at pos {pos}", stacklevel=2)
    # For negative positions
    for pos in positions_neg:
        try:
            aa = seq[seq_len + pos]
        except Exception:
            warnings.warn(f"Unable to get pos {pos}", stacklevel=2)
            continue
        for atom, value in mass.std_aa_comp[aa].items():
            try:
                pos_mat[pos - min(pos_total), dict_index_pos[atom]] = value
            except (KeyError, IndexError):
                warnings.warn(f"Pos matrix skip: {atom} at neg pos {pos}", stacklevel=2)
    return pos_mat


def _apply_composition_to_matrices(
    mat: np.ndarray,
    pos_mat: np.ndarray,
    composition: mass.Composition,
    i: int,
    seq_len: int,
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
    positions: set[int],
) -> None:
    """Apply a composition delta to the standard and positional matrices."""
    for atom_comp, change in composition.items():
        try:
            mat[i, dict_index[atom_comp]] += change
            if i in positions:
                pos_mat[i, dict_index_pos[atom_comp]] += change
            elif (i - seq_len) in positions:
                pos_mat[i - seq_len, dict_index_pos[atom_comp]] += change
        except KeyError:
            try:
                warnings.warn(f"Replacing pattern for atom: {atom_comp}", stacklevel=2)
                atom_comp_clean = sub(r"\[.*?\]", "", atom_comp)
                mat[i, dict_index[atom_comp_clean]] += change
                if i in positions:
                    pos_mat[i, dict_index_pos[atom_comp_clean]] += change
                elif (i - seq_len) in positions:
                    pos_mat[i - seq_len, dict_index_pos[atom_comp_clean]] += change
            except KeyError:
                warnings.warn(f"Ignoring atom {atom_comp} at pos {i}", stacklevel=2)
                continue
        except IndexError:
            warnings.warn(f"Index error for atom {atom_comp} at pos {i}", stacklevel=2)


def _apply_modifications(
    mat: np.ndarray,
    pos_mat: np.ndarray,
    parsed_seq: list,
    seq_len: int,
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
    positions: set[int],
) -> None:
    """Apply modification changes to the matrices."""
    for i, token in enumerate(parsed_seq):
        if token[1] is None:
            continue
        try:
            mod_comp = token[1][0].composition
        except Exception:
            warnings.warn(
                f"Skipping modification without known composition: {token[1]}", stacklevel=2
            )
            continue
        _apply_composition_to_matrices(
            mat,
            pos_mat,
            mod_comp,
            i,
            seq_len,
            dict_index,
            dict_index_pos,
            positions,
        )


def _apply_terminal_modifications(
    mat: np.ndarray,
    pos_mat: np.ndarray,
    peptidoform: Peptidoform,
    seq_len: int,
    dict_index: dict[str, int],
    dict_index_pos: dict[str, int],
    positions: set[int],
) -> None:
    """Apply N- and C-terminal modification changes to the matrices."""
    terminal_mods = [
        (0, peptidoform.properties.get("n_term")),  # N-terminus at position 0
        (seq_len - 1, peptidoform.properties.get("c_term")),  # C-terminus at last position
    ]
    for i, mods in terminal_mods:
        if not mods:
            continue
        for tag in mods:
            try:
                mod_comp = tag.composition
            except Exception:
                warnings.warn(
                    f"Skipping terminal modification without known composition: {tag}",
                    stacklevel=2,
                )
                continue
            _apply_composition_to_matrices(
                mat,
                pos_mat,
                mod_comp,
                i,
                seq_len,
                dict_index,
                dict_index_pos,
                positions,
            )


def _compute_rolling_sum(matrix: np.ndarray, n: int = 2) -> np.ndarray:
    """Compute a rolling sum over the matrix."""
    ret = np.cumsum(matrix, axis=1, dtype=np.float32)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1 :]
