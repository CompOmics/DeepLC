"""A prediction matrix held as its low-rank factors rather than materialised."""

from __future__ import annotations

import numpy as np

__all__ = ["FactoredPredictionMatrix"]


class FactoredPredictionMatrix:
    """
    A prediction matrix kept as its low-rank factors.

    Holds the ``(n_peptides, n_tasks)`` output of a
    :class:`~deeplc._architecture.FactorHead` as the factors it is computed from.

    That head is low rank by construction::

        pred[:, j] = (proj(trunk).embedding[j]) * scale[j] + shift[j]

    so the whole matrix is determined by ``proj(trunk)``, which is ``(n_peptides, rank)``,
    together with the head's own parameters. At rank 64 and 6,543 setups that is 102 times
    smaller than the matrix it stands for: 2.6 M peptides need 0.6 GiB of factors instead of
    63 GiB of predictions.

    It indexes like the matrix. ``m[rows]``, ``m[:, heads]`` and ``m[rows, head]`` each
    evaluate only what was asked for, so a caller that reads a few thousand rows to choose a
    head and then one column per run never builds the rest. ``np.asarray(m)`` still gives the
    dense matrix for code that genuinely needs it, at its full size.

    Parameters
    ----------
    projections
        ``proj(trunk)`` for every peptide, shape ``(n_peptides, rank)``.
    embedding
        Per-setup embedding, shape ``(n_tasks, rank)``.
    scale, shift
        Per-setup affine map, shape ``(n_tasks,)``.

    """

    #: Marks this as a source a calibration may index instead of a materialised matrix.
    is_head_source = True

    def __init__(
        self,
        projections: np.ndarray,
        embedding: np.ndarray,
        scale: np.ndarray,
        shift: np.ndarray,
    ) -> None:
        self._projections = np.asarray(projections)
        self._embedding = np.asarray(embedding, dtype=self._projections.dtype)
        self._scale = np.asarray(scale, dtype=self._projections.dtype)
        self._shift = np.asarray(shift, dtype=self._projections.dtype)
        if self._embedding.shape[1] != self._projections.shape[1]:
            raise ValueError(
                f"embedding rank {self._embedding.shape[1]} does not match projection rank "
                f"{self._projections.shape[1]}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and task count, without evaluating anything."""
        return (self._projections.shape[0], self._embedding.shape[0])

    @property
    def ndim(self) -> int:
        """Always two: this stands in for a matrix."""
        return 2

    @property
    def dtype(self) -> np.dtype:
        """Element type of the matrix it stands for."""
        return self._projections.dtype

    @property
    def rank(self) -> int:
        """Width of the factorisation."""
        return self._projections.shape[1]

    @property
    def nbytes(self) -> int:
        """What the factors occupy."""
        return int(
            self._projections.nbytes
            + self._embedding.nbytes
            + self._scale.nbytes
            + self._shift.nbytes
        )

    @property
    def dense_nbytes(self) -> int:
        """What the matrix would occupy if it were materialised."""
        rows, columns = self.shape
        return int(rows * columns * self._projections.itemsize)

    def __len__(self) -> int:
        """Return the number of peptides."""
        return self.shape[0]

    def _evaluate(self, rows, columns) -> np.ndarray:
        """Compute just the requested block from the factors."""
        projections = self._projections if rows is None else self._projections[rows]
        if projections.ndim == 1:
            projections = projections[None, :]
        if columns is None:
            embedding, scale, shift = self._embedding, self._scale, self._shift
        else:
            embedding = np.atleast_2d(self._embedding[columns])
            scale = np.atleast_1d(self._scale[columns])
            shift = np.atleast_1d(self._shift[columns])
        return (projections @ embedding.T) * scale + shift

    def _select_rows(self, rows) -> FactoredPredictionMatrix:
        """Narrow to a subset of peptides, still as factors."""
        return FactoredPredictionMatrix(
            self._projections[rows], self._embedding, self._scale, self._shift
        )

    def __getitem__(self, key):
        """
        Index as the dense matrix would, evaluating only the block that is asked for.

        Selecting rows alone gives another factored matrix rather than a dense one. That is
        what lets a caller narrow to one run's peptides and still hand the result to a
        calibration, which reads a few dozen of the thousands of heads: nothing in that path
        ever builds the wide matrix. A scalar in either position drops that axis, as numpy
        does, so ``m[rows, head]`` is one dimensional and ``m[i, j]`` is a scalar.
        """
        rows, columns = key if isinstance(key, tuple) else (key, None)
        if isinstance(rows, slice) and rows == slice(None):
            rows = None
        if columns is None and rows is not None and not _is_scalar(rows):
            return self._select_rows(rows)
        row_scalar = _is_scalar(rows)
        column_scalar = columns is not None and _is_scalar(columns)
        block = self._evaluate(rows, columns)
        if row_scalar:
            block = block[0]
            return block[0] if column_scalar else block
        return block[:, 0] if column_scalar else block

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the whole matrix, for callers that really need it, at its full size."""
        dense = self._evaluate(None, None)
        return dense if dtype is None else dense.astype(dtype)

    def __getattr__(self, name: str):
        """
        Fall back to the dense matrix for anything not answered from the factors.

        Reductions such as ``.min()`` and ``.mean()`` have no cheap factored form, so code
        that calls them gets the same answer it always did, at the same cost it always had.
        Only indexing, which is the path that matters for a wide matrix, stays lazy.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __repr__(self) -> str:
        """Show the shape it stands for and what it actually costs."""
        rows, columns = self.shape
        return (
            f"{type(self).__name__}(shape=({rows}, {columns}), rank={self.rank}, "
            f"{_human(self.nbytes)} held for a {_human(self.dense_nbytes)} matrix)"
        )


def _is_scalar(index) -> bool:
    """Whether an index selects one element rather than a subset."""
    return np.isscalar(index) or (isinstance(index, np.generic) and np.ndim(index) == 0)


def _human(n: int) -> str:
    """Format a byte count in the largest unit that keeps it above one."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GiB"
