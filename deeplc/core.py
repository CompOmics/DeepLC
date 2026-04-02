"""DeepLC core functions."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from psm_utils import PSM, Peptidoform, PSMList

from deeplc import _model_ops
from deeplc.calibration import (
    Calibration,
    SplineTransformerCalibration,
)
from deeplc.data import DeepLCDataset, split_datasets
from deeplc.multitask import MultitaskAdapter

LOGGER = logging.getLogger(__name__)

DEEPLC_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "full_hc_PXD005573_pub_1fd8363d9af9dcad3be7553c39396960.pt"
DEFAULT_MODEL = DEEPLC_DIR / "package_data" / "models" / DEFAULT_MODEL_NAME


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


def _is_multitask_output(predictions: np.ndarray) -> bool:
    return predictions.ndim == 2 and predictions.shape[1] > 1


def predict(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    model: torch.nn.Module | PathLike | str | None = None,
    predict_kwargs: dict | None = None,
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

    Returns
    -------
    np.ndarray
        Retention time predictions.

    """
    return _model_ops.predict(
        model=model or DEFAULT_MODEL,
        data=DeepLCDataset.from_psm_list(_parse_psms(psm_list)),
        **(predict_kwargs or {}),
    ).numpy()


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
    )

    # Fit calibration
    LOGGER.debug("Fitting calibration...")
    target_rt_cal = np.array(psm_list_reference["retention_time"], dtype=np.float32)

    if _is_multitask_output(source_rt_cal):
        selected_head_idx = _best_correlating_head(source_rt_cal, target_rt_cal)
        source_rt_cal = source_rt_cal[:, selected_head_idx]
        setattr(calibration, "selected_head_idx", int(selected_head_idx))

    calibration.fit(target=target_rt_cal, source=source_rt_cal)

    return calibration


def predict_and_calibrate(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList,
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
        List of PSMs to use as reference for calibration.
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
    # Predict initial retention times
    LOGGER.info("Predicting retention times...")
    predicted_rt = predict(
        psm_list=_parse_psms(psm_list),
        model=model,
        predict_kwargs=predict_kwargs,
    )

    if calibration is not None and not isinstance(calibration, Calibration):
        raise ValueError(
            f"Expected calibration to be of type `Calibration`, got {type(calibration)}"
        )

    # Fit calibration if not already fitted
    if calibration is None or not calibration.is_fitted:
        calibration = calibrate(
            psm_list_reference=psm_list_reference,
            model=model,
            calibration=calibration,
            predict_kwargs=predict_kwargs,
        )
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")

    if _is_multitask_output(predicted_rt):
        selected_head_idx = getattr(calibration, "selected_head_idx", None)
        if selected_head_idx is None:
            ref_pred_rt = predict(
                psm_list=psm_list_reference,
                model=model,
                predict_kwargs=predict_kwargs,
            )
            if _is_multitask_output(ref_pred_rt):
                ref_targets = np.array(psm_list_reference["retention_time"], dtype=np.float32)
                selected_head_idx = _best_correlating_head(ref_pred_rt, ref_targets)
            else:
                selected_head_idx = 0
        predicted_rt = predicted_rt[:, int(selected_head_idx)]

    # Apply calibration to predictions
    calibrated_rt = calibration.transform(predicted_rt)

    return calibrated_rt


def finetune_and_predict(
    psm_list: PSMList | list[PSM | Peptidoform | str],
    psm_list_reference: PSMList,
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
        List of PSMs to use as reference for fine-tuning and calibration.
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
    # Fine-tune the model
    finetuned_model = finetune(
        psm_list_reference=psm_list_reference,
        model=model,
        train_kwargs=train_kwargs,
    )

    # Predict retention times with fine-tuned model
    LOGGER.info("Predicting retention times with fine-tuned model...")
    predicted_rt = predict(
        psm_list=_parse_psms(psm_list),
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
    )

    # Fit calibration with simple PiecewiseLinearCalibration to the fine-tuned model predictions
    LOGGER.info("Fitting calibration with fine-tuned model predictions...")
    calibration = calibrate(
        psm_list_reference=psm_list_reference,
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
    model_for_training: torch.nn.Module | PathLike | str | None = model or DEFAULT_MODEL

    loaded_model = _model_ops.load_model(
        model_for_training,
        device=train_kwargs_local.get("device"),
    )

    sample_features, _ = training_dataset[0]
    sample_features = [feature.unsqueeze(0).to(next(loaded_model.parameters()).device) for feature in sample_features]
    with torch.no_grad():
        sample_output = loaded_model(*sample_features)

    if sample_output.ndim == 2 and sample_output.shape[1] > 1:
        adapter_hidden_size = int(train_kwargs_local.pop("adapter_hidden_size", 256))
        freeze_epochs = int(train_kwargs_local.pop("freeze_epochs", 5))
        model_for_training = MultitaskAdapter(
            multitask_model=loaded_model,
            n_heads=sample_output.shape[1],
            hidden_size=adapter_hidden_size,
        )
        train_kwargs_local["freeze_epochs"] = freeze_epochs
    else:
        model_for_training = loaded_model

    finetuned_model = _model_ops.train(
        model=model_for_training,
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
