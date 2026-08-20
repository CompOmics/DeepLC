"""Tests for deeplc._features.encode_peptidoform."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from psm_utils import Peptidoform
from pyteomics import mass

from deeplc._features import (
    DEFAULT_DICT_AA,
    DEFAULT_DICT_INDEX,
    DEFAULT_DICT_INDEX_POS,
    DEFAULT_POSITIONS,
    DEFAULT_POSITIONS_NEG,
    DEFAULT_POSITIONS_POS,
    encode_peptidoform,
)

# HELPERS

PADDING = 60

# Number of rows in pos_matrix: max(positions) - min(positions) + 1
# DEFAULT_POSITIONS = {0,1,2,3,-1,-2,-3,-4}  →  3 - (-4) + 1 = 8
_POS_ROWS = max(DEFAULT_POSITIONS) - min(DEFAULT_POSITIONS) + 1
# matrix_global = sum(std_matrix, axis=0) [6] + seq_len [1] + pos_matrix.flatten() [8*6]
_GLOBAL_BASE_LEN = len(DEFAULT_DICT_INDEX) + 1 + _POS_ROWS * len(DEFAULT_DICT_INDEX_POS)
# _compute_rolling_sum(std_matrix.T, n=2)[:, ::2].T → (30, 6)
_SUM_ROWS = (PADDING - 1) // 2  # == 29 for n=2, stride 2 on 59 cols


class TestReturnStructure:
    """Tests that encode_peptidoform returns the expected keys and shapes."""

    def test_returns_four_keys(self):
        result = encode_peptidoform("ACDE")
        assert set(result.keys()) == {"matrix", "matrix_sum", "matrix_global", "matrix_hc"}

    def test_matrix_shape(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix"].shape == (PADDING, len(DEFAULT_DICT_INDEX))

    def test_matrix_hc_shape(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix_hc"].shape == (PADDING, len(DEFAULT_DICT_AA))

    def test_matrix_global_shape_no_ccs(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix_global"].shape == (_GLOBAL_BASE_LEN,)

    def test_matrix_global_shape_with_ccs(self):
        result = encode_peptidoform("ACDE/2", add_ccs_features=True)
        # add_ccs_features appends 5 extra values (H%, FWY%, DE%, KR%, charge)
        assert result["matrix_global"].shape == (_GLOBAL_BASE_LEN + 5,)

    def test_matrix_sum_shape(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix_sum"].ndim == 2
        assert result["matrix_sum"].shape[1] == len(DEFAULT_DICT_INDEX)

    def test_matrix_dtype(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix"].dtype == np.float16

    def test_matrix_hc_dtype(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix_hc"].dtype == np.float16


class TestStringInput:
    """Tests that both str and Peptidoform inputs are accepted and equivalent."""

    def test_str_and_peptidoform_are_equivalent(self):
        str_result = encode_peptidoform("ACDE")
        pf_result = encode_peptidoform(Peptidoform("ACDE"))
        for key in str_result:
            np.testing.assert_array_equal(str_result[key], pf_result[key])


class TestPaddingAndSeqLen:
    """Tests that padding and sequence length are handled correctly."""

    def test_padded_rows_are_zero(self):
        seq = "ACDE"
        result = encode_peptidoform(seq)
        # Rows beyond seq length should be all zeros in standard matrix
        assert np.all(result["matrix"][len(seq) :] == 0)

    def test_padded_rows_are_zero_onehot(self):
        seq = "ACDE"
        result = encode_peptidoform(seq)
        assert np.all(result["matrix_hc"][len(seq) :] == 0)

    def test_seq_len_encoded_in_matrix_global(self):
        seq = "ACDE"
        result = encode_peptidoform(seq)
        # matrix_global[len(DEFAULT_DICT_INDEX)] holds seq_len
        assert result["matrix_global"][len(DEFAULT_DICT_INDEX)] == len(seq)

    def test_truncation_warns(self):
        long_seq = "A" * (PADDING + 5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = encode_peptidoform(long_seq)
            assert any("Truncating" in str(warning.message) for warning in w)
        # After truncation seq_len == PADDING
        assert result["matrix_global"][len(DEFAULT_DICT_INDEX)] == PADDING


class TestOneHotEncoding:
    """Tests the one-hot (matrix_hc) component."""

    def test_first_residue_one_hot(self):
        # "A" is index 5 in DEFAULT_DICT_AA
        result = encode_peptidoform("ACDE")
        assert result["matrix_hc"][0, DEFAULT_DICT_AA["A"]] == 1.0

    def test_second_residue_one_hot(self):
        result = encode_peptidoform("ACDE")
        assert result["matrix_hc"][1, DEFAULT_DICT_AA["C"]] == 1.0

    def test_each_residue_has_exactly_one_hot(self):
        seq = "ACDE"
        result = encode_peptidoform(seq)
        for i in range(len(seq)):
            assert result["matrix_hc"][i].sum() == 1.0

    def test_padded_rows_are_zero_and_no_hot(self):
        seq = "AC"
        result = encode_peptidoform(seq)
        assert result["matrix_hc"][2:].sum() == 0.0


class TestStandardMatrixComposition:
    """Tests that atomic composition in std_matrix is correct."""

    def test_glycine_carbon_count(self):
        # Glycine (G): C2 H3 N1 O1 — check carbon at index 0
        result = encode_peptidoform("G")
        c_idx = DEFAULT_DICT_INDEX["C"]
        expected_c = mass.std_aa_comp["G"]["C"]
        assert result["matrix"][0, c_idx] == expected_c

    def test_unmodified_and_modified_differ_in_affected_residue(self):
        # Oxidized methionine adds one O
        unmod = encode_peptidoform("ACMDE")
        mod = encode_peptidoform("ACM[Oxidation]DE")
        o_idx = DEFAULT_DICT_INDEX["O"]
        assert mod["matrix"][2, o_idx] > unmod["matrix"][2, o_idx]

    def test_modification_does_not_affect_other_residues(self):
        unmod = encode_peptidoform("ACMDE")
        mod = encode_peptidoform("ACM[Oxidation]DE")
        for i in [0, 1, 3, 4]:
            np.testing.assert_array_equal(mod["matrix"][i], unmod["matrix"][i])


class TestNTerminalModification:
    """Tests that N-terminal modifications are applied to position 0."""

    def test_nterm_mod_changes_position_zero(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("[Acetyl]-ACDE")
        # Acetyl adds C2H2O to position 0; at least carbon should increase
        c_idx = DEFAULT_DICT_INDEX["C"]
        assert mod["matrix"][0, c_idx] > unmod["matrix"][0, c_idx]

    def test_nterm_mod_does_not_affect_other_positions(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("[Acetyl]-ACDE")
        for i in range(1, 4):
            np.testing.assert_array_equal(mod["matrix"][i], unmod["matrix"][i])

    def test_nterm_mod_reflected_in_matrix_global(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("[Acetyl]-ACDE")
        # matrix_global contains the column sums so modification must change it
        assert not np.array_equal(mod["matrix_global"], unmod["matrix_global"])

    def test_nterm_mod_reflected_in_pos_matrix_part(self):
        # Position 0 is in DEFAULT_POSITIONS_POS so pos_matrix row 0 must change
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("[Acetyl]-ACDE")
        # pos_matrix is concatenated at the end of matrix_global after the base part
        base = len(DEFAULT_DICT_INDEX) + 1  # col sums + seq_len
        pos_flat_unmod = unmod["matrix_global"][base:]
        pos_flat_mod = mod["matrix_global"][base:]
        assert not np.array_equal(pos_flat_unmod, pos_flat_mod)


class TestCTerminalModification:
    """Tests that C-terminal modifications are applied to the last residue position."""

    def test_cterm_mod_changes_last_residue_position(self):
        seq = "ACDE"
        unmod = encode_peptidoform(seq)
        mod = encode_peptidoform("ACDE-[Amidation]")
        last = len(seq) - 1
        # Amidation changes N count (replaces O with NH2)
        assert not np.array_equal(mod["matrix"][last], unmod["matrix"][last])

    def test_cterm_mod_does_not_affect_other_positions(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("ACDE-[Amidation]")
        for i in range(0, 3):
            np.testing.assert_array_equal(mod["matrix"][i], unmod["matrix"][i])

    def test_cterm_mod_reflected_in_matrix_global(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("ACDE-[Amidation]")
        assert not np.array_equal(mod["matrix_global"], unmod["matrix_global"])


class TestBothTerminalModifications:
    """Tests a peptide carrying both N- and C-terminal modifications."""

    def test_both_term_mods_change_both_ends(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("[Acetyl]-ACDE-[Amidation]")
        assert not np.array_equal(mod["matrix"][0], unmod["matrix"][0])
        assert not np.array_equal(mod["matrix"][3], unmod["matrix"][3])

    def test_middle_residues_unchanged(self):
        unmod = encode_peptidoform("ACDE")
        mod = encode_peptidoform("[Acetyl]-ACDE-[Amidation]")
        for i in [1, 2]:
            np.testing.assert_array_equal(mod["matrix"][i], unmod["matrix"][i])


class TestCCSFeatures:
    """Tests the add_ccs_features flag."""

    def test_ccs_features_requires_charge(self):
        with pytest.raises(ValueError, match="no charge"):
            encode_peptidoform("ACDE", add_ccs_features=True)

    def test_ccs_features_appends_five_values(self):
        base = encode_peptidoform("ACDE/2")
        ccs = encode_peptidoform("ACDE/2", add_ccs_features=True)
        assert ccs["matrix_global"].shape[0] == base["matrix_global"].shape[0] + 5

    def test_ccs_charge_value_position(self):
        # matrix_global layout with CCS:
        # [col_sums(6), seq_len(1), H%(1), FWY%(1), DE%(1), KR%(1), charge(1), pos_flat(48)]
        charge = 3
        result = encode_peptidoform(f"ACDE/{charge}", add_ccs_features=True)
        charge_idx = len(DEFAULT_DICT_INDEX) + 1 + 4  # 6 col sums + seq_len + 4 ratios
        assert result["matrix_global"][charge_idx] == charge


class TestShortPeptide:
    """Tests edge cases for short peptides."""

    def test_single_residue(self):
        result = encode_peptidoform("A")
        assert result["matrix"].shape == (PADDING, len(DEFAULT_DICT_INDEX))
        assert result["matrix_hc"][0, DEFAULT_DICT_AA["A"]] == 1.0

    def test_two_residues_no_crash(self):
        result = encode_peptidoform("AC")
        assert result["matrix_global"].shape == (_GLOBAL_BASE_LEN,)


def test_positional_delta_uses_same_row_as_base_residue():
    """A modification delta must land on the row of the residue it modifies.

    Regression test: ``_fill_pos_matrix`` offsets rows by ``min(positions)`` so
    that row 0 means position -4, while modification deltas were written with
    raw indexing, putting an N-terminal delta four residues from the other end.
    """
    from deeplc._features import DEFAULT_POSITIONS, encode_peptidoform

    order = sorted(DEFAULT_POSITIONS)
    plain = encode_peptidoform("PEPTIDEK")["matrix_global"][7:55].reshape(8, 6)
    n_term = encode_peptidoform("[Acetyl]-PEPTIDEK")["matrix_global"][7:55].reshape(8, 6)
    c_term = encode_peptidoform("PEPTIDEK-[Amidated]")["matrix_global"][7:55].reshape(8, 6)

    changed_n = [order[r] for r in range(8) if (n_term[r] - plain[r]).any()]
    changed_c = [order[r] for r in range(8) if (c_term[r] - plain[r]).any()]

    assert changed_n == [0], f"N-terminal delta landed at {changed_n}, expected position 0"
    assert changed_c == [-1], f"C-terminal delta landed at {changed_c}, expected position -1"


def test_terminal_composition_is_opt_in_and_separates_terminal_from_side_chain():
    """``[Acetyl]-PEPTIDEK`` and ``P[Acetyl]EPTIDEK`` are chemically different."""
    import numpy as np

    from deeplc._features import encode_peptidoform

    default_width = encode_peptidoform("PEPTIDEK")["matrix_global"].shape[0]
    assert default_width == 55, "default global width must not change"

    terminal = encode_peptidoform("[Acetyl]-PEPTIDEK", add_terminal_composition=True)
    side_chain = encode_peptidoform("P[Acetyl]EPTIDEK", add_terminal_composition=True)

    assert terminal["matrix_global"].shape[0] == 67
    assert not np.allclose(terminal["matrix_global"], side_chain["matrix_global"])
    # the acetyl composition C2H2O appears in the N-terminal block only when terminal
    assert terminal["matrix_global"][55:61].tolist() == [2, 2, 0, 1, 0, 0]
    assert side_chain["matrix_global"][55:61].tolist() == [0, 0, 0, 0, 0, 0]


# LEGACY ENCODING, FOR MODELS TRAINED BEFORE 4.0.1

#: Non-zero entries of ``matrix_global`` as produced by a real v4.0.0 checkout.
#: Captured by running v4.0.0 rather than derived, so this pins the compatibility
#: path to what those models were actually trained against. Only modified
#: peptidoforms are listed, since unmodified ones encode identically either way.
V400_GLOBAL_NONZERO: dict[str, dict[int, float]] = {
    "AC[UNIMOD:4]DEK": {
        0: 23,
        1: 37,
        2: 7,
        3: 10,
        4: 1,
        6: 5,
        7: 3,
        8: 5,
        9: 1,
        10: 1,
        11: 1,
        13: 6,
        14: 8,
        15: 2,
        16: 4,
        19: 5,
        20: 7,
        21: 1,
        22: 3,
        25: 6,
        26: 12,
        27: 2,
        28: 1,
        31: 3,
        32: 5,
        33: 1,
        34: 1,
        37: 3,
        38: 5,
        39: 1,
        40: 1,
        41: 1,
        43: 4,
        44: 5,
        45: 1,
        46: 3,
        49: 5,
        50: 7,
        51: 1,
        52: 3,
    },
    "[UNIMOD:737]-PEPTIDEK": {
        0: 52,
        1: 83,
        2: 11,
        3: 17,
        6: 8,
        7: 18,
        8: 31,
        9: 3,
        10: 3,
        13: 4,
        14: 5,
        15: 1,
        16: 3,
        19: 5,
        20: 7,
        21: 1,
        22: 3,
        25: 6,
        26: 12,
        27: 2,
        28: 1,
        31: 5,
        32: 7,
        33: 1,
        34: 1,
        37: 5,
        38: 7,
        39: 1,
        40: 3,
        43: 5,
        44: 7,
        45: 1,
        46: 1,
        49: 4,
        50: 7,
        51: 1,
        52: 2,
    },
    "PEPTIDEK-[UNIMOD:2]": {
        0: 40,
        1: 64,
        2: 10,
        3: 14,
        6: 8,
        7: 6,
        8: 11,
        9: 1,
        10: 1,
        13: 4,
        14: 5,
        15: 1,
        16: 3,
        19: 5,
        20: 7,
        21: 1,
        22: 3,
        25: 6,
        26: 12,
        27: 2,
        28: 1,
        31: 5,
        32: 7,
        33: 1,
        34: 1,
        37: 5,
        38: 7,
        39: 1,
        40: 3,
        43: 5,
        44: 7,
        45: 1,
        46: 1,
        49: 4,
        50: 8,
        51: 2,
        52: 1,
    },
    "M[UNIMOD:35]EEPTIDEK": {
        0: 45,
        1: 72,
        2: 10,
        3: 19,
        4: 1,
        6: 9,
        7: 6,
        8: 11,
        9: 1,
        10: 2,
        13: 4,
        14: 5,
        15: 1,
        16: 3,
        19: 5,
        20: 7,
        21: 1,
        22: 3,
        25: 6,
        26: 12,
        27: 2,
        28: 1,
        31: 5,
        32: 9,
        33: 1,
        34: 1,
        35: 1,
        37: 5,
        38: 7,
        39: 1,
        40: 3,
        43: 5,
        44: 7,
        45: 1,
        46: 3,
        49: 5,
        50: 7,
        51: 1,
        52: 1,
    },
    "PEPS[UNIMOD:21]TIDEK": {
        0: 43,
        1: 69,
        2: 10,
        3: 20,
        5: 1,
        6: 9,
        7: 6,
        8: 11,
        9: 1,
        10: 1,
        13: 4,
        14: 5,
        15: 1,
        16: 3,
        19: 5,
        20: 7,
        21: 1,
        22: 3,
        25: 6,
        26: 13,
        27: 2,
        28: 4,
        30: 1,
        31: 5,
        32: 7,
        33: 1,
        34: 1,
        37: 5,
        38: 7,
        39: 1,
        40: 3,
        43: 5,
        44: 7,
        45: 1,
        46: 1,
        49: 3,
        50: 5,
        51: 1,
        52: 2,
    },
    "AC[UNIMOD:4]DEKR": {
        0: 29,
        1: 49,
        2: 11,
        3: 11,
        4: 1,
        6: 6,
        7: 4,
        8: 5,
        9: 1,
        10: 3,
        13: 7,
        14: 10,
        15: 2,
        16: 4,
        19: 6,
        20: 12,
        21: 2,
        22: 1,
        25: 6,
        26: 12,
        27: 4,
        28: 1,
        31: 3,
        32: 5,
        33: 1,
        34: 1,
        37: 3,
        38: 5,
        39: 1,
        40: 1,
        41: 1,
        43: 4,
        44: 5,
        45: 1,
        46: 3,
        49: 5,
        50: 7,
        51: 1,
        52: 3,
    },
}


@pytest.mark.parametrize("proforma", sorted(V400_GLOBAL_NONZERO))
def test_legacy_positional_deltas_reproduces_v400(proforma):
    """
    The compatibility path must match v4.0.0 exactly, not approximately.

    IM2Deep's CCS models and every DeepLC checkpoint from before 4.0.1 were trained
    against the pre-fix placement, so any deviation here silently changes their
    predictions on modified peptides.
    """
    result = encode_peptidoform(proforma, legacy_positional_deltas=True)["matrix_global"]
    expected = np.zeros_like(result)
    for index, value in V400_GLOBAL_NONZERO[proforma].items():
        expected[index] = value
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("proforma", ["PEPTIDEK", "ACDEK", "LGEYGFQNALIVR", "A" * 70])
def test_legacy_flag_is_a_no_op_without_modifications(proforma):
    """An unmodified peptidoform has no deltas to place, so both paths agree."""
    for key, legacy in encode_peptidoform(proforma, legacy_positional_deltas=True).items():
        np.testing.assert_array_equal(legacy, encode_peptidoform(proforma)[key], err_msg=key)


def test_legacy_and_corrected_placement_differ_on_modified_peptidoforms():
    """Guards against the flag being silently wired to nothing."""
    legacy = encode_peptidoform("[Acetyl]-PEPTIDEK", legacy_positional_deltas=True)
    corrected = encode_peptidoform("[Acetyl]-PEPTIDEK")

    assert not np.array_equal(legacy["matrix_global"], corrected["matrix_global"])
    # The defect was positional only: the per-residue matrix is unaffected, so the
    # difference must be confined to the positional block of matrix_global.
    np.testing.assert_array_equal(legacy["matrix"], corrected["matrix"])
    np.testing.assert_array_equal(legacy["matrix_global"][:7], corrected["matrix_global"][:7])


def test_dataset_defaults_to_the_encoding_released_models_expect():
    """
    The default must be the pre-4.0.1 placement, not the corrected one.

    This is what lets a downstream package holding a model trained before the
    correction keep working without changing its call. IM2Deep reaches DeepLC only
    through ``from_psm_list(psm_list, add_ccs_features=True)``, so that exact call
    is what is checked here.
    """
    from deeplc.data import DeepLCDataset

    peptidoforms = ["AC[UNIMOD:4]DEK/2", "[UNIMOD:737]-PEPTIDEK/2"]
    default = DeepLCDataset.from_psm_list(_psm_list(peptidoforms), add_ccs_features=True)
    corrected = DeepLCDataset.from_psm_list(
        _psm_list(peptidoforms), add_ccs_features=True, legacy_positional_deltas=False
    )

    for index, proforma in enumerate(peptidoforms):
        legacy_expected = encode_peptidoform(
            proforma, add_ccs_features=True, legacy_positional_deltas=True
        )["matrix_global"].astype(np.float32)
        # The dataset stores float32 while matrix_global is float64, so compare at
        # float32 precision rather than exactly.
        np.testing.assert_array_equal(default[index][0][2].numpy(), legacy_expected)
        assert not np.array_equal(default[index][0][2].numpy(), corrected[index][0][2].numpy()), (
            "the corrected path must still be reachable"
        )


def _psm_list(peptidoforms):
    """Build a PSMList over ``peptidoforms``; from_psm_list does not take strings."""
    from psm_utils import PSM, PSMList

    return PSMList(
        psm_list=[
            PSM(peptidoform=Peptidoform(p), spectrum_id=str(i), retention_time=float(i))
            for i, p in enumerate(peptidoforms)
        ]
    )


def test_undescribed_checkpoint_resolves_to_the_legacy_encoding():
    """
    Every checkpoint DeepLC has released is a bare state dict with no spec.

    Those models predate the correction, so an absent specification has to mean the
    old placement or their predictions on modified peptides change silently.
    """
    from deeplc.core import _feature_kwargs_from_spec

    for spec in (None, {}):
        assert _feature_kwargs_from_spec(spec)["legacy_positional_deltas"] is True


def test_described_checkpoint_resolves_to_the_corrected_encoding():
    """A recorded specification is only written by versions that carry the fix."""
    from deeplc.core import _feature_kwargs_from_spec

    resolved = _feature_kwargs_from_spec({"padding_length": 60, "global_dim": 67})
    assert resolved["legacy_positional_deltas"] is False
    assert resolved["padding_length"] == 60


def test_described_checkpoint_may_request_the_legacy_encoding():
    """A model trained on the old placement can say so and be believed."""
    from deeplc.core import _feature_kwargs_from_spec

    spec = {"padding_length": 60, "legacy_positional_deltas": True}
    assert _feature_kwargs_from_spec(spec)["legacy_positional_deltas"] is True


def test_bundled_multitask_model_declares_its_encoding():
    """The shipped model was trained after the correction, so it must say so."""
    import torch

    from deeplc.core import FLEXCNN_MULTITASK_MODEL, _feature_kwargs_from_spec

    blob = torch.load(FLEXCNN_MULTITASK_MODEL, map_location="cpu", weights_only=False)
    spec = blob["feature_spec"]
    assert spec["legacy_positional_deltas"] is False
    assert _feature_kwargs_from_spec(spec)["legacy_positional_deltas"] is False
