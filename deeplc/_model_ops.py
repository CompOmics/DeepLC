"""Training, predicting, and evaluating with PyTorch."""

import copy
import logging
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    track,
)
from torch.utils.data import DataLoader, Dataset, Subset

from deeplc._architecture import DeepLCModel
from deeplc.data import DeepLCDataset

logger = logging.getLogger(__name__)


def load_model(
    model: torch.nn.Module | PathLike | str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a model from a file or return a randomly initialized model if none is provided."""
    # If device is not specified, use the default device (GPU if available, else CPU)
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from file if a path is provided
    if isinstance(model, (str, PathLike, Path)):
        loaded_model = torch.load(model, weights_only=False, map_location=selected_device)
    elif isinstance(model, torch.nn.Module):
        loaded_model = model
        logger.debug("Using provided PyTorch model instance")
    elif model is None:
        # Initialize a new model with default architecture
        loaded_model = DeepLCModel()
        logger.debug("Initialized new DeepLCModel with default architecture")
    else:
        raise TypeError(f"Expected a PyTorch Module or a file path, got {type(model)} instead.")

    # Ensure the model is on the specified device
    loaded_model.to(selected_device)

    return loaded_model


def train(
    model: torch.nn.Module | PathLike | str | None,
    train_dataset: DeepLCDataset | Subset[DeepLCDataset],
    validation_dataset: DeepLCDataset | Subset[DeepLCDataset],
    device: str | None = None,
    num_workers: int = 0,
    num_threads: int | None = None,
    learning_rate: float = 0.001,
    epochs: int = 25,
    batch_size: int = 512,
    patience: int = 10,
    show_progress: bool = True,
) -> torch.nn.Module:
    """
    Train or fine-tune the model.

    Parameters
    ----------
    model
        Model to train or path to model file.
    train_dataset
        Training dataset.
    validation_dataset
        Validation dataset.
    device
        Device to train on ('cpu' or 'cuda').
    num_workers
        Number of worker processes for data loading.
    num_threads
        Number of threads for model operations on CPU (ignored if using GPU).
    learning_rate
        Learning rate for optimizer.
    epochs
        Maximum number of training epochs.
    batch_size
        Batch size for training and validation.
    patience
        Number of epochs with no improvement before early stopping.
    show_progress
        If True, display a Rich progress bar during training. If False, run silently.

    Returns
    -------
    torch.nn.Module
        Trained model.

    """
    torch.set_num_threads(num_threads or torch.get_num_threads())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, device)

    # Parse datasets; setup loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if len(train_loader) == 0:
        raise ValueError("Training data loader is empty. Provide at least one training sample.")
    if len(val_loader) == 0:
        raise ValueError(
            "Validation data loader is empty. Adjust validation data or validation_split."
        )

    optimizer = _get_optimizer(model, learning_rate)
    loss_fn = torch.nn.L1Loss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    with _create_progress(disable=not show_progress) as progress:
        epoch_task = progress.add_task("Epochs", total=epochs, status="")

        for _epoch in range(epochs):
            avg_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
            avg_val_loss = _validate_epoch(model, val_loader, loss_fn, device)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Update epoch bar with loss info
            status = f"loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  best={best_val_loss:.4f}"
            if epochs_no_improve >= patience:
                status += "  [yellow]early stop[/yellow]"
            progress.update(epoch_task, advance=1, status=status)

            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_model_wts)
    return model


def predict(
    model: torch.nn.Module | PathLike | str | None,
    data: Dataset,
    device: str | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    num_threads: int | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Predict using the model for the given dataset."""
    torch.set_num_threads(num_threads or torch.get_num_threads())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, device)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = _predict_epoch(model, data_loader, device, show_progress=show_progress)
    return predictions.cpu().detach()


def evaluate(
    model: torch.nn.Module | PathLike | str | None,
    data: Dataset,
    device: str | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    num_threads: int | None = None,
) -> float:
    """Evaluate the model on the given dataset."""
    torch.set_num_threads(num_threads or torch.get_num_threads())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, device)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_fn = torch.nn.L1Loss()
    avg_loss = _validate_epoch(model, data_loader, loss_fn, device)
    return avg_loss


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
    for features, targets in data_loader:
        features = [feature_tensor.to(device) for feature_tensor in features]
        targets = targets.to(device).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(*features)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return float(running_loss / len(data_loader))


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
        for features, targets in data_loader:
            features = [feature_tensor.to(device) for feature_tensor in features]
            targets = targets.to(device).view(-1, 1)
            outputs = model(*features)
            val_loss += loss_fn(outputs, targets).item()
    return float(val_loss / len(data_loader))


def _predict_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    show_progress: bool = False,
) -> torch.Tensor:
    """Predict using the model for one epoch."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in track(
            data_loader, description="Predicting...", transient=True, disable=not show_progress
        ):
            features = [feature_tensor.to(device) for feature_tensor in features]
            outputs = model(*features)
            predictions.append(outputs.cpu())
    if not predictions:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(predictions, dim=0).squeeze()


def _create_progress(disable: bool = False) -> Progress:
    """Create a Rich progress bar for training."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        TextColumn("|"),
        TextColumn("{task.fields[status]}"),
        disable=disable,
        transient=True,
    )
