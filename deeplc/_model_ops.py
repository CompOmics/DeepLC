"""Training, predicting, and evaluating with PyTorch."""

import copy
import logging
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from deeplc._data import split_datasets

logger = logging.getLogger(__name__)


def load_model(
    model: torch.nn.Module | PathLike | str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a model from a file or return a randomly initialized model if none is provided."""
    # If device is not specified, use the default device (GPU if available, else CPU)
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from file if a path is provided
    if isinstance(model, str | Path):
        loaded_model = torch.load(model, weights_only=False, map_location=selected_device)
    elif isinstance(model, torch.nn.Module):
        loaded_model = model
    elif model is None:
        # TODO: Implement randomly initialized model; requires model architecture definition
        raise NotImplementedError("Loading randomly initialized model is not implemented yet.")
    else:
        raise TypeError(f"Expected a PyTorch Module or a file path, got {type(model)} instead.")

    # Ensure the model is on the specified device
    loaded_model.to(selected_device)

    return loaded_model


def train(
    model: torch.nn.Module | PathLike | str | None,
    train_data: Dataset,
    validation_data: Dataset | None = None,
    validation_split: float = 0.1,
    device: str = "cpu",
    learning_rate: float = 0.001,
    epochs: int = 10,
    batch_size: int = 512,
    patience: int = 5,
    trainable_layers: str | None = None,
) -> torch.nn.Module:
    """
    Train or fine-tune the model.

    Parameters
    ----------
    model
        Model to train or path to model file.
    train_data
        Training dataset.
    validation_data
        Validation dataset. If None, a portion of train_data is used.
    validation_split
        Fraction of train_data to use for validation when validation_data is None.
    device
        Device to train on ('cpu' or 'cuda').
    learning_rate
        Learning rate for optimizer.
    epochs
        Maximum number of training epochs.
    batch_size
        Batch size for training and validation.
    patience
        Number of epochs with no improvement before early stopping.
    trainable_layers
        If provided, only layers containing this keyword in their name will be trainable.
        All other layers will be frozen. If None, all layers are trainable.

    Returns
    -------
    torch.nn.Module
        Trained model.

    """
    model = load_model(model, device)

    # Freeze layers if requested
    if trainable_layers is not None:
        _freeze_layers(model, trainable_layers)
        logger.debug(f"Frozen all layers except those containing '{trainable_layers}'")

    # Parse dataset and split arguments; setup loaders
    train_dataset, val_dataset = split_datasets(train_data, validation_data, validation_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = _get_optimizer(model, learning_rate)
    loss_fn = torch.nn.L1Loss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        avg_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        avg_val_loss = _validate_epoch(model, val_loader, loss_fn, device)

        logger.debug(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.debug(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_wts)
    return model


def predict(
    model: torch.nn.Module | PathLike | str | None,
    data: Dataset,
    device: str = "cpu",
    batch_size: int = 512,
    num_workers: int = 0,
) -> torch.Tensor:
    """Predict using the model for the given dataset."""
    model = load_model(model, device)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = _predict_epoch(model, data_loader, device)
    return predictions.cpu().detach()


def evaluate(
    model: torch.nn.Module | PathLike | str | None,
    data: Dataset,
    device: str = "cpu",
    batch_size: int = 512,
    num_workers: int = 0,
) -> float:
    """Evaluate the model on the given dataset."""
    model = load_model(model, device)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_fn = torch.nn.L1Loss()
    avg_loss = _validate_epoch(model, data_loader, loss_fn, device)
    return avg_loss


def _freeze_layers(model: torch.nn.Module, unfreeze_keyword: str) -> None:
    """
    Freeze all layers except those containing the unfreeze_keyword in their name.

    """
    for name, param in model.named_parameters():
        param.requires_grad = unfreeze_keyword in name


def _get_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )


def _train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str,
) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    for features, targets in track(data_loader):
        features = [feature_tensor.to(device) for feature_tensor in features]
        targets = targets.to(device).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(*features)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = float(running_loss / len(data_loader))
    return avg_loss


def _validate_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    device: str,
) -> float:
    """Validate the model for one epoch."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, targets in track(data_loader):
            features = [feature_tensor.to(device) for feature_tensor in features]
            targets = targets.to(device).view(-1, 1)
            outputs = model(*features)
            val_loss += loss_fn(outputs, targets).item()
    avg_val_loss = float(val_loss / len(data_loader))
    return avg_val_loss


def _predict_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
) -> torch.Tensor:
    """Predict using the model for one epoch."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in track(data_loader):
            features = [feature_tensor.to(device) for feature_tensor in features]
            outputs = model(*features)
            predictions.append(outputs.cpu())
    return torch.cat(predictions, dim=0)
