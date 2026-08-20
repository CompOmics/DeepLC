"""DeepLC core functions."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from psm_utils import PSM, Peptidoform, PSMList

from deeplc import _model_ops
from deeplc._reference_selection import select_reference_psms
from deeplc.calibration import (
    Calibration,
    SplineTransformerCalibration,
)
from deeplc.data import DeepLCDataset, split_datasets

LOGGER = logging.getLogger(__name__)

DEEPLC_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = DEEPLC_DIR / "package_data" / "models" / "multitask_model.pt"

#: Fused-trunk multitask model, trained across 6,543 LC setups. Not the default:
#: switching would change every prediction, so the choice is left to the caller
#: until the calibration path is adapted to its low-rank head.
FLEXCNN_MULTITASK_MODEL = DEEPLC_DIR / "package_data" / "models" / "multitask_flexcnn_model.pt"


def predict(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    model: torch.nn.Module | PathLike | str | None = None,
    predict_kwargs: dict | None = None,
    return_matrix: bool = False,
) -> np.ndarray:
    """
    Predict retention times for a list of PSMs using a trained model.

    Parameters
    ----------
    psm_list
        List of PSMs to predict retention times for.
    model
        Trained model or path to model file. If None, the default DeepLC model is used.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.
    return_matrix
        If True, return the full prediction matrix of shape ``(n, n_heads)`` when using a
        multitask model. If False (default), return a 1D array of shape ``(n,)`` using
        head 0 when model output is 2D.

    Returns
    -------
    np.ndarray
        Retention time predictions. Shape ``(n,)`` unless ``return_matrix=True`` and model
        produces multitask output, in which case shape is ``(n, n_heads)``.

    """
    # The model is loaded before the dataset is built because the features it
    # needs depend on the model. A checkpoint that describes itself carries a
    # feature specification, and a model trained on the 67-dimensional global
    # vector cannot be fed the 55-dimensional default.
    loaded_model = _model_ops.load_model(model or DEFAULT_MODEL)
    feature_spec = getattr(loaded_model, "feature_spec", None) or {}

    result = _model_ops.predict(
        model=loaded_model,
        data=DeepLCDataset.from_psm_list(
            _parse_psms(psm_list),
            add_ccs_features=bool(feature_spec.get("add_ccs_features", False)),
            add_terminal_composition=bool(feature_spec.get("add_terminal_composition", False)),
        ),
        **(predict_kwargs or {}),
    ).numpy()
    if not return_matrix:
        return result[:, 0]
    return result


def calibrate(
    psm_list_reference: PSMList,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | None = None,
    predict_kwargs: dict | None = None,
) -> Calibration:
    """
    Return a `Calibration` instance fitted to the reference dataset.

    Parameters
    ----------
    psm_list_reference
        List of PSMs to use as reference for calibration.
    model
        Trained model or path to model file.
    calibration
        Calibration instance to use. If None, SplineTransformerCalibration is used.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    Calibration
        Fitted calibration instance.

    """
    # Get calibration
    if calibration is None:
        LOGGER.debug("No calibration provided, using SplineTransformerCalibration by default.")
        calibration = SplineTransformerCalibration()
    elif not isinstance(calibration, Calibration):
        raise ValueError(
            f"Expected calibration to be of type `Calibration`, got {type(calibration)}"
        )
    if calibration.is_fitted:
        LOGGER.warning(
            "Provided Calibration is already fitted. Refitting will overwrite existing fit."
        )

    if any(psm_list_reference["is_decoy"]):
        LOGGER.warning(
            "Reference PSM list contains decoy PSMs. "
            "These will be included in the calibration fitting."
        )

    # Predict initial retention times for the reference dataset
    LOGGER.debug("Predicting retention times for reference...")
    source_rt_cal = predict(
        psm_list=psm_list_reference,
        model=model,
        predict_kwargs=predict_kwargs,
        return_matrix=True,
    )

    # Fit calibration
    LOGGER.debug("Fitting calibration...")
    target_rt_cal = np.array(psm_list_reference["retention_time"], dtype=np.float32)

    # Select the best head for calibration if the model predicts for multiple LC setups
    if source_rt_cal.shape[1] > 1:
        calibration.selected_model_head = _best_correlating_head(source_rt_cal, target_rt_cal)
    source_rt_cal = source_rt_cal[:, calibration.selected_model_head or 0]

    calibration.fit(target=target_rt_cal, source=source_rt_cal)

    return calibration


def predict_and_calibrate(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList | list[PSM | Peptidoform | str] | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | None = None,
    predict_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Predict retention times and calibrate to a reference.

    Parameters
    ----------
    psm_list
        List of PSMs to predict retention times for.
    psm_list_reference
        List of PSMs to use as reference for calibration. If None, the best PSMs are
        automatically selected from ``psm_list`` (auto-calibration). This requires that the input
        PSM list contains observed retention times, score and decoy status to select the best PSMs
        for auto-calibration.
    model
        Trained model or path to model file.
    calibration
        Calibration instance to use. If None, SplineTransformerCalibration is used.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    np.ndarray
        Calibrated retention time predictions.

    """
    parsed_psm_list = _parse_psms(psm_list)

    if psm_list_reference is None:
        parsed_psm_list_ref = select_reference_psms(parsed_psm_list)
    else:
        parsed_psm_list_ref = _parse_psms(psm_list_reference)

    # Predict initial retention times
    LOGGER.info("Predicting retention times...")
    predicted_rt = predict(
        psm_list=parsed_psm_list,
        model=model,
        predict_kwargs=predict_kwargs,
        return_matrix=True,
    )

    if calibration is not None and not isinstance(calibration, Calibration):
        raise ValueError(
            f"Expected calibration to be of type `Calibration`, got {type(calibration)}"
        )

    # Fit calibration if not already fitted
    if calibration is None or not calibration.is_fitted:
        calibration = calibrate(
            psm_list_reference=parsed_psm_list_ref,
            model=model,
            calibration=calibration,
            predict_kwargs=predict_kwargs,
        )
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")

    if predicted_rt.shape[1] > 1:
        if calibration.selected_model_head is None:
            raise ValueError(
                "Calibration has no selected_model_head. Either use calibrate() to fit it, "
                "or set calibration.selected_model_head manually before calling "
                "predict_and_calibrate() with a multitask model."
            )
        predicted_rt = predicted_rt[:, calibration.selected_model_head]
    else:
        predicted_rt = predicted_rt[:, 0]

    # Apply calibration to predictions
    calibrated_rt = calibration.transform(predicted_rt)

    return calibrated_rt


def finetune_and_predict(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList | list[PSM | Peptidoform | str] | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    train_kwargs: dict | None = None,
    predict_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Fine-tune the model to a reference and predict new retention times.

    Parameters
    ----------
    psm_list
        List of PSMs to predict retention times for.
    psm_list_reference
        List of PSMs to use as reference for calibration. If None, the best PSMs are automatically
        selected from ``psm_list`` (auto-calibration). This requires that the input PSM list
        contains observed retention times, score and decoy status to select the best PSMs for
        auto-calibration.
    model
        Trained model or path to model file.
    train_kwargs
        Additional keyword arguments to pass to the training function.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    np.ndarray
        Calibrated retention time predictions after fine-tuning.

    """
    parsed_psm_list = _parse_psms(psm_list)

    if psm_list_reference is None:
        parsed_psm_list_ref = select_reference_psms(parsed_psm_list)
    else:
        parsed_psm_list_ref = _parse_psms(psm_list_reference)

    # Fine-tune the model
    finetuned_model = finetune(
        psm_list_reference=parsed_psm_list_ref,
        model=model,
        train_kwargs=train_kwargs,
    )

    # Predict retention times with fine-tuned model
    LOGGER.info("Predicting retention times with fine-tuned model...")
    predicted_rt = predict(
        psm_list=parsed_psm_list,
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
    )

    # Fit calibration with simple PiecewiseLinearCalibration to the fine-tuned model predictions
    LOGGER.info("Fitting calibration with fine-tuned model predictions...")
    calibration = calibrate(
        psm_list_reference=parsed_psm_list_ref,
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
    )

    # Apply calibration to predictions
    calibrated_rt = calibration.transform(predicted_rt)

    return calibrated_rt


def finetune(
    psm_list_reference: PSMList,
    psm_list_validation: PSMList | None = None,
    validation_split: float = 0.1,
    model: torch.nn.Module | PathLike | str | None = None,
    train_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Fine-tune an existing model.

    Parameters
    ----------
    psm_list_reference
        List of PSMs to use as reference for fine-tuning.
    psm_list_validation
        List of PSMs to use for validation during fine-tuning. If None, a split from psm_list is
        used.
    validation_split
        Fraction of ``psm_list_reference`` to use for validation when ``psm_list_validation``
        is None.
    model
        Trained model or path to model file.
    train_kwargs
        Additional keyword arguments to pass to the training function.

    Returns
    -------
    torch.nn.Module
        Fine-tuned model.

    """
    LOGGER.info("Fine-tuning model...")
    if any(psm_list_reference["is_decoy"]):
        # TODO: Move to reusable validation step?
        LOGGER.warning("PSM list contains decoy PSMs. These will be used for fine tuning.")
    training_data = DeepLCDataset.from_psm_list(psm_list_reference)
    validation_data = (
        DeepLCDataset.from_psm_list(psm_list_validation) if psm_list_validation else None
    )
    training_dataset, validation_dataset = split_datasets(
        training_data, validation_data=validation_data, validation_split=validation_split
    )
    train_kwargs_local = dict(train_kwargs or {})
    adapter_hidden_size = int(train_kwargs_local.pop("adapter_hidden_size", 256))
    freeze_epochs = int(train_kwargs_local.pop("freeze_epochs", 5))
    train_kwargs_local.setdefault("epochs", 50)

    loaded_model = _model_ops.load_model(
        model or DEFAULT_MODEL,
        device=train_kwargs_local.get("device"),
    )
    loaded_model.add_adapter(hidden_size=adapter_hidden_size)
    train_kwargs_local["freeze_epochs"] = freeze_epochs

    finetuned_model = _model_ops.train(
        model=loaded_model,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        **train_kwargs_local,
    )
    return finetuned_model


def train(
    psm_list_reference: PSMList,
    psm_list_validation: PSMList | None = None,
    validation_split: float = 0.1,
    train_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Train a new model from scratch.

    Parameters
    ----------
    psm_list_reference
        List of PSMs to use as reference for fine-tuning.
    psm_list_validation
        List of PSMs to use for validation. If None, a split from psm_list is used.
    validation_split
        If psm_list_validation is None, this fraction of psm_list will be used for validation.
    train_kwargs
        Additional keyword arguments to pass to the training function.

    Returns
    -------
    torch.nn.Module
        Trained model.

    """
    training_data = DeepLCDataset.from_psm_list(psm_list_reference)
    validation_data = (
        DeepLCDataset.from_psm_list(psm_list_validation) if psm_list_validation else None
    )
    training_dataset, validation_dataset = split_datasets(
        training_data, validation_data=validation_data, validation_split=validation_split
    )
    LOGGER.info("Training new model...")
    trained_model = _model_ops.train(
        model=None,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        **(train_kwargs or {}),
    )
    return trained_model


def save_model(model: torch.nn.Module, path: PathLike | str) -> None:
    """
    Save a model's state dict to a file.

    Use :func:`load_model` (via :func:`predict`) to reload the saved checkpoint.

    Parameters
    ----------
    model
        Trained model instance to save.
    path
        Destination file path.

    """
    torch.save(model.state_dict(), path)


def _parse_psms(psm_list: PSMList | list[PSM | Peptidoform | str]) -> PSMList:
    """
    Parse a list of PSMs, Peptidoforms, or strings into a PSMList.

    Note that this function can only be used for inputs that do not require additional data,
    such as retention times or decoy status. It cannot be used for reference or validation
    data sets that require observed retention times for calibration or training.

    """
    if isinstance(psm_list, PSMList):
        return psm_list
    elif isinstance(psm_list, list):
        if all(isinstance(psm, PSM) for psm in psm_list):
            return PSMList(psm_list=psm_list)
        elif all(isinstance(psm, Peptidoform) for psm in psm_list) or all(
            isinstance(psm, str) for psm in psm_list
        ):
            return PSMList(
                psm_list=[PSM(spectrum_id=i, peptidoform=pf) for i, pf in enumerate(psm_list)]
            )
        else:
            raise ValueError("List must contain either PSMs, Peptidoforms, or strings.")
    else:
        raise ValueError("Input must be a PSMList or a list of PSMs, Peptidoforms, or strings.")


def _best_correlating_head(predictions: np.ndarray, targets: np.ndarray) -> int:
    """Return the head index with highest valid Pearson correlation to targets."""
    best_idx = 0
    best_corr = float("-inf")

    for idx in range(predictions.shape[1]):
        pred_col = predictions[:, idx]
        mask = np.isfinite(pred_col) & np.isfinite(targets)
        if mask.sum() < 3:
            continue
        pred_masked = pred_col[mask]
        target_masked = targets[mask]
        if np.std(pred_masked) < 1e-8 or np.std(target_masked) < 1e-8:
            continue
        corr = np.corrcoef(pred_masked, target_masked)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_idx = idx

    return best_idx
