"""DeepLC core functions."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from psm_utils.psm_list import PSMList

from deeplc import _model_ops
from deeplc.calibration import (
    Calibration,
    PiecewiseLinearCalibration,
    SplineTransformerCalibration,
)
from deeplc.data import DeepLCDataset, split_datasets

LOGGER = logging.getLogger(__name__)

DEEPLC_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "full_hc_PXD005573_pub_1fd8363d9af9dcad3be7553c39396960.pt"
DEFAULT_MODEL = DEEPLC_DIR / "package_data" / "models" / DEFAULT_MODEL_NAME


def predict(
    psm_list: PSMList,
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
    LOGGER.info("Predicting retention times...")
    return _model_ops.predict(
        model=model or DEFAULT_MODEL,
        data=DeepLCDataset.from_psm_list(psm_list),
        **(predict_kwargs or {}),
    ).numpy()


def predict_and_calibrate(
    psm_list: PSMList,
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
        psm_list=psm_list,
        model=model,
        predict_kwargs=predict_kwargs,
    )

    # Fit calibration
    # Validate passed calibration instance or create default if None
    if calibration is None:
        LOGGER.debug("No calibration provided, using SplineTransformerCalibration by default.")
        calibration = SplineTransformerCalibration()
    elif not isinstance(calibration, Calibration):
        raise ValueError(
            f"Expected calibration to be of type Calibration, got {type(calibration)}"
        )

    # Fit calibration if not already fitted
    if not calibration.is_fitted:
        LOGGER.info("Fitting calibration...")
        if any(psm_list_reference["is_decoy"]):
            LOGGER.warning(
                "Reference PSM list contains decoy PSMs. "
                "These will be included in the calibration fitting."
            )
        target_rt_cal = np.array(psm_list_reference["retention_time"], dtype=np.float32)
        source_rt_cal = predict(
            psm_list=psm_list_reference,
            model=model,
            predict_kwargs=predict_kwargs,
        )
        calibration.fit(target=target_rt_cal, source=source_rt_cal)
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")

    # Apply calibration to predictions
    calibrated_rt = calibration.transform(predicted_rt)

    return calibrated_rt


def finetune_and_predict(
    psm_list: PSMList,
    psm_list_reference: PSMList,
    model: torch.nn.Module | PathLike | str | None = None,
    partial_freeze: bool = False,
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
    partial_freeze
        If True, only the final layer of the model will be finetuned.
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
        psm_list=psm_list_reference,
        model=model,
        partial_freeze=partial_freeze,
        train_kwargs=train_kwargs,
    )

    # Predict retention times with fine-tuned model
    LOGGER.info("Predicting retention times with fine-tuned model...")
    predicted_rt = predict(
        psm_list=psm_list,
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
    )

    # Fit calibration
    LOGGER.info("Fitting calibration with fine-tuned model predictions...")
    if any(psm_list_reference["is_decoy"]):  #  remove this one since already in finetune?
        LOGGER.warning(
            "Reference PSM list contains decoy PSMs. "
            "These will be included in the calibration fitting."
        )
    calibration = PiecewiseLinearCalibration()
    target_rt_cal = np.array(psm_list_reference["retention_time"], dtype=np.float32)
    source_rt_cal = predict(
        psm_list=psm_list_reference,
        model=finetuned_model,
        predict_kwargs=predict_kwargs,
    )
    calibration.fit(target=target_rt_cal, source=source_rt_cal)

    # Apply calibration to predictions
    calibrated_rt = calibration.transform(predicted_rt)

    return calibrated_rt


def finetune(
    psm_list: PSMList,
    psm_list_validation: PSMList | None = None,
    validation_split: float = 0.1,
    model: torch.nn.Module | PathLike | str | None = None,
    partial_freeze: bool = False,
    train_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Fine-tune an existing model.

    Parameters
    ----------
    psm_list
        List of PSMs to use as reference for fine-tuning.
    psm_list_validation
        List of PSMs to use for validation during fine-tuning. If None, a split from psm_list is
        used.
    model
        Trained model or path to model file.
    partial_freeze
        If True, only the final layer of the model will be finetuned.
    train_kwargs
        Additional keyword arguments to pass to the training function.

    Returns
    -------
    torch.nn.Module
        Fine-tuned model.

    """
    LOGGER.info("Fine-tuning model...")
    if any(psm_list["is_decoy"]):
        # TODO: Move to reusable validation step?
        LOGGER.warning("PSM list contains decoy PSMs. These will be used for fine tuning.")
    training_data = DeepLCDataset.from_psm_list(psm_list)
    validation_data = (
        DeepLCDataset.from_psm_list(psm_list_validation) if psm_list_validation else None
    )
    training_dataset, validation_dataset = split_datasets(
        training_data, validation_data=validation_data, validation_split=validation_split
    )
    finetuned_model = _model_ops.train(
        model=model or DEFAULT_MODEL,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        trainable_layers="33_1" if partial_freeze else None,  # TODO: Don't hardcode
        **(train_kwargs or {}),
    )
    return finetuned_model


def train(
    psm_list: PSMList,
    psm_list_validation: PSMList | None = None,
    validation_split: float = 0.1,
    train_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Train a new model from scratch.

    Parameters
    ----------
    psm_list
        List of PSMs to use for training.
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
    training_data = DeepLCDataset.from_psm_list(psm_list)
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
