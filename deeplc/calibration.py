"""
Retention time calibration utilities.

This module provides calibration strategies to map source retention times to
an aligned target scale.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[import]
from sklearn.pipeline import Pipeline, make_pipeline  # type: ignore[import]
from sklearn.preprocessing import SplineTransformer  # type: ignore[import]

from deeplc._exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)


class Calibration(ABC):
    """Abstract base class for retention time calibration."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Indicates whether the calibration model has been fitted."""
        ...

    @abstractmethod
    def fit(self, target_rt: np.ndarray, source_rt: np.ndarray) -> None:
        """Fit the calibration from source to target."""
        ...

    @abstractmethod
    def transform(self, source_rt: np.ndarray) -> np.ndarray:
        """Transform source retention times into the calibrated target space."""
        ...


class IdentityCalibration(Calibration):
    """No calibration; returns inputs unchanged."""

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, target_rt: np.ndarray, source_rt: np.ndarray) -> None:  # noqa: ARG002
        return None

    def transform(self, source_rt: np.ndarray) -> np.ndarray:
        return source_rt


class PiecewiseLinearCalibration(Calibration):
    def __init__(
        self,
        number_of_splits: int = 50,
        extrapolate: bool = True,
        use_median: bool = True,
    ) -> None:
        """
        Piece-wise linear calibration based on per-split anchors.

        Parameters
        ----------
        number_of_splits : int
            Number of segments to split the source retention time range into.
            More segments allow more flexibility but may lead to overfitting.
        extrapolate : bool
            If True, allows extrapolation outside the fitted source retention time range.
            If False, clips input values to the fitted range.
        use_median : bool
            If True, uses the median of each segment to define anchors. If False, uses the mean.
        """
        super().__init__()
        self.number_of_splits = int(number_of_splits)
        self.extrapolate = bool(extrapolate)
        self.use_median = bool(use_median)

        self._calibrate_min: float | None = None
        self._calibrate_max: float | None = None
        self._source_breakpoints: np.ndarray | None = None
        self._slopes: np.ndarray | None = None
        self._intercepts: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self._calibrate_min is not None
            and self._calibrate_max is not None
            and self._source_breakpoints is not None
            and self._slopes is not None
            and self._intercepts is not None
        )

    @property
    def calibrate_min(self) -> float | None:
        return self._calibrate_min

    @property
    def calibrate_max(self) -> float | None:
        return self._calibrate_max

    def fit(self, target_rt: np.ndarray, source_rt: np.ndarray) -> None:
        """Fit a piece-wise linear model mapping source to target retention times."""
        target_rt, source_rt = _prepare_series(target_rt, source_rt)

        cal_min = float(source_rt[0])
        cal_max = float(source_rt[-1])
        if (not np.isfinite(cal_min)) or (not np.isfinite(cal_max)) or (cal_max <= cal_min):
            raise CalibrationError(
                "Source retention times have zero or invalid range; cannot calibrate."
            )

        boundaries = np.linspace(cal_min, cal_max, self.number_of_splits + 1, dtype=np.float32)
        starts: np.ndarray = np.searchsorted(source_rt, boundaries[:-1], side="left")  # type: ignore[var-annotated]
        ends: np.ndarray = np.searchsorted(source_rt, boundaries[1:], side="left")  # type: ignore[var-annotated]

        # Filter out empty segments
        valid_segments = ends > starts
        starts = starts[valid_segments]
        ends = ends[valid_segments]

        # Compute anchors for all segments
        aggregate_func = np.median if self.use_median else np.mean
        tgt_anchors = np.array(
            [aggregate_func(target_rt[s:e]) for s, e in zip(starts, ends, strict=True)],
            dtype=np.float32,
        )
        src_anchors = np.array(
            [aggregate_func(source_rt[s:e]) for s, e in zip(starts, ends, strict=True)],
            dtype=np.float32,
        )

        if len(src_anchors) < 2:
            raise CalibrationError(
                "Not enough anchor points to build a piecewise calibration (need >= 2)."
            )

        src_arr = np.asarray(src_anchors, dtype=np.float32)
        tgt_arr = np.asarray(tgt_anchors, dtype=np.float32)
        keep = np.concatenate(([True], src_arr[1:] > src_arr[:-1]))
        src_arr = src_arr[keep]
        tgt_arr = tgt_arr[keep]
        if src_arr.size < 2:
            raise CalibrationError(
                "After removing degenerate anchors, not enough points remain to define segments."
            )

        delta_src = src_arr[1:] - src_arr[:-1]
        delta_tgt = tgt_arr[1:] - tgt_arr[:-1]
        slopes = delta_tgt / delta_src
        intercepts = (-src_arr[:-1] * slopes) + tgt_arr[:-1]

        self._source_breakpoints = src_arr.astype(np.float32)
        self._slopes = slopes.astype(np.float32)
        self._intercepts = intercepts.astype(np.float32)
        self._calibrate_min = cal_min
        self._calibrate_max = cal_max

        LOGGER.debug(
            "Piecewise fit: anchors=%d, segments=%d, range=[%.3f, %.3f]",
            len(self._source_breakpoints),
            len(self._slopes),
            self._calibrate_min,
            self._calibrate_max,
        )

    def transform(self, source_rt: np.ndarray) -> np.ndarray:
        """Transform source retention times using the fitted piece-wise linear model."""
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")

        # Ensure type checking knows these are not None
        self._source_breakpoints = cast(np.ndarray, self._source_breakpoints)
        self._slopes = cast(np.ndarray, self._slopes)
        self._intercepts = cast(np.ndarray, self._intercepts)

        if source_rt.shape[0] == 0:
            return np.array([])

        x = source_rt.astype(np.float32, copy=False)
        x_eval = (
            np.clip(x, self._calibrate_min, self._calibrate_max) if not self.extrapolate else x
        )

        idx = np.searchsorted(self._source_breakpoints, x_eval, side="right") - 1
        idx = np.clip(idx, 0, len(self._source_breakpoints) - 2)
        y = self._slopes[idx] * x_eval + self._intercepts[idx]
        return y

    def get_calibration_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the calibration anchors as two arrays (x, y)."""
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")

        # Ensure type checking knows these are not None
        self._source_breakpoints = cast(np.ndarray, self._source_breakpoints)
        self._slopes = cast(np.ndarray, self._slopes)
        self._intercepts = cast(np.ndarray, self._intercepts)

        x = self._source_breakpoints.astype(np.float64)
        y = np.empty_like(x, dtype=np.float64)
        y[0] = float(self._slopes[0] * x[0] + self._intercepts[0])
        if len(x) > 1:
            prev_idx = np.arange(0, len(x) - 1)
            y[1:] = (self._slopes[prev_idx] * x[1:] + self._intercepts[prev_idx]).astype(
                np.float64
            )
        return x, y


class SplineTransformerCalibration(Calibration):
    def __init__(self) -> None:
        super().__init__()
        self._calibrate_min: float | None = None
        self._calibrate_max: float | None = None
        self._model_left: LinearRegression | None = None
        self._model_main: Pipeline | LinearRegression | None = None
        self._model_right: LinearRegression | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self._calibrate_min is not None
            and self._calibrate_max is not None
            and self._model_left is not None
            and self._model_main is not None
            and self._model_right is not None
        )

    def fit(
        self,
        target_rt: np.ndarray,
        source_rt: np.ndarray,
        simplified: bool = False,
    ) -> None:
        """Fit a spline-based model mapping source to target retention times."""
        target_rt, source_rt = _prepare_series(target_rt, source_rt)

        # TODO: What's the use of `simplified`? Was taken from original code.
        if simplified:
            linear_model = LinearRegression()
            linear_model.fit(source_rt.reshape(-1, 1), target_rt)
            linear_model_left = linear_model
            spline_model = linear_model
            linear_model_right = linear_model
        else:
            spline = SplineTransformer(degree=4, n_knots=int(len(source_rt) / 500) + 5)
            spline_model = make_pipeline(spline, LinearRegression())
            spline_model.fit(source_rt.reshape(-1, 1), target_rt)

            n_top = int(len(source_rt) * 0.1)
            X_left = source_rt[:n_top]
            y_left = target_rt[:n_top]
            linear_model_left = LinearRegression()
            linear_model_left.fit(X_left.reshape(-1, 1), y_left)

            X_right = source_rt[-n_top:]
            y_right = target_rt[-n_top:]
            linear_model_right = LinearRegression()
            linear_model_right.fit(X_right.reshape(-1, 1), y_right)

        self._calibrate_min = float(np.min(source_rt))
        self._calibrate_max = float(np.max(source_rt))
        self._model_left = linear_model_left
        self._model_main = spline_model
        self._model_right = linear_model_right

    def transform(self, source_rt: np.ndarray) -> np.ndarray:
        """Transform source retention times using the fitted spline model."""
        if not self.is_fitted:
            raise CalibrationError("The model has not been fitted yet. Call fit() first.")

        # Ensure type checking knows the models are not None
        model_main = cast(Pipeline | LinearRegression, self._model_main)
        model_left = cast(LinearRegression, self._model_left)
        model_right = cast(LinearRegression, self._model_right)

        if source_rt.shape[0] == 0:
            return np.array([])

        y_pred_spline = model_main.predict(source_rt.reshape(-1, 1))
        y_pred_left = model_left.predict(source_rt.reshape(-1, 1))
        y_pred_right = model_right.predict(source_rt.reshape(-1, 1))
        within_range = (source_rt >= self._calibrate_min) & (source_rt <= self._calibrate_max)
        within_range = within_range.ravel()

        cal_preds = np.copy(y_pred_spline)
        cal_preds[~within_range & (source_rt.ravel() < self._calibrate_min)] = y_pred_left[
            ~within_range & (source_rt.ravel() < self._calibrate_min)
        ]
        cal_preds[~within_range & (source_rt.ravel() > self._calibrate_max)] = y_pred_right[
            ~within_range & (source_rt.ravel() > self._calibrate_max)
        ]
        return np.array(cal_preds)


def _prepare_series(
    target_rt: np.ndarray,
    source_rt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare target/source arrays: shape, sort by source, cast to float32."""
    if len(target_rt) != len(source_rt):
        raise ValueError(
            "Target and source retention times must have the same length. Got "
            f"{len(target_rt)} and {len(source_rt)}."
        )
    if len(target_rt.shape) > 1:
        target_rt = target_rt.flatten()
    if len(source_rt.shape) > 1:
        source_rt = source_rt.flatten()

    idx = np.argsort(source_rt)
    target_rt = np.array(target_rt, dtype=np.float32)[idx]
    source_rt = np.array(source_rt, dtype=np.float32)[idx]

    return target_rt, source_rt
