"""Main command line interface to DeepLC."""

import logging
import multiprocessing
import os
import sys
from pathlib import Path

# PyInstaller console=False sets sys.stdout/stderr to None; redirect to devnull
# so NiceGUI/pywebview don't fail silently on startup.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

import click
import pandas as pd
from psm_utils.io import READERS, read_file
from rich.logging import RichHandler

import deeplc.core
from deeplc import __version__

logger = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

PSM_FILETYPES = list(READERS.keys())


def _infer_output_name(input_filename: str, output_name: str | None = None) -> Path:
    """Infer output filename from input filename if not provided."""
    if output_name:
        return Path(output_name)
    else:
        input_path = Path(input_filename)
        return input_path.with_name(input_path.stem + "_deeplc_predictions").with_suffix(".csv")


def _read_psm_file(psms: str, psm_filetype: str | None = None):
    """Read a PSM file and return a PSMList."""
    logger.info(f"Reading PSM file: {psms}")
    kwargs = {"filetype": psm_filetype} if psm_filetype else {}
    return read_file(psms, **kwargs)


def _write_predictions(psm_list, predictions, output_path: Path):
    """Write predictions to a CSV file."""
    df = pd.DataFrame(
        {
            "peptidoform": [str(psm.peptidoform) for psm in psm_list],
            "observed_rt": [psm.retention_time for psm in psm_list],
            "predicted_rt": predictions,
        }
    )
    logger.info(f"Writing predictions to {output_path}")
    df.to_csv(output_path, index=False)


@click.group()
@click.option(
    "--logging-level",
    "-l",
    type=click.Choice(LOGGING_LEVELS.keys()),
    default="INFO",
    help="Set the logging level.",
)
@click.version_option(version=__version__)
def cli(logging_level, **kwargs):
    """DeepLC: Retention time prediction for peptides carrying any modification."""
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOGGING_LEVELS[logging_level],
        handlers=[RichHandler(rich_tracebacks=True, show_level=True, show_path=False)],
    )


def _validate_finetune(ctx, param, value):
    """Validate that --finetune is only used with --reference or --auto-calibrate."""
    if value and not ctx.params.get("reference") and not ctx.params.get("auto_calibrate"):
        raise click.UsageError("--finetune requires --reference or --auto-calibrate.")
    return value


@cli.command()
@click.argument("psms", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--psm-filetype",
    "-t",
    type=click.Choice(PSM_FILETYPES),
    default=None,
    help="File type for the input PSM file. Inferred from extension if not provided.",
)
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Reference PSM file for calibration or fine-tuning.",
)
@click.option(
    "--reference-filetype",
    "-rt",
    type=click.Choice(PSM_FILETYPES),
    default=None,
    help="File type for the reference file. Inferred from extension if not provided.",
)
@click.option(
    "--auto-calibrate",
    is_flag=True,
    default=False,
    help="Automatically select the best PSMs from the input file as calibration reference.",
)
@click.option(
    "--finetune",
    is_flag=True,
    default=False,
    callback=_validate_finetune,
    expose_value=True,
    help="Fine-tune the model to the reference before predicting. Requires --reference or --auto-calibrate.",  # noqa: E501
)
@click.option("--output", "-o", type=str, default=None, help="Output file path.")
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a model file. Uses the built-in default model if not provided.",
)
def predict(
    psms,
    psm_filetype,
    reference,
    reference_filetype,
    auto_calibrate,
    finetune,
    output,
    model,
):
    """Predict retention times for a list of peptide-spectrum matches."""
    if auto_calibrate and reference:
        raise click.UsageError("--auto-calibrate and --reference are mutually exclusive.")

    psm_list = _read_psm_file(psms, psm_filetype)
    output_path = _infer_output_name(psms, output)

    if reference:
        psm_list_reference = _read_psm_file(reference, reference_filetype)
        if finetune:
            predictions = deeplc.core.finetune_and_predict(
                psm_list=psm_list,
                psm_list_reference=psm_list_reference,
                model=model,
            )
        else:
            predictions = deeplc.core.predict_and_calibrate(
                psm_list=psm_list,
                psm_list_reference=psm_list_reference,
                model=model,
            )
    elif auto_calibrate:
        if finetune:
            predictions = deeplc.core.finetune_and_predict(psm_list=psm_list, model=model)
        else:
            predictions = deeplc.core.predict_and_calibrate(psm_list=psm_list, model=model)
    else:
        predictions = deeplc.core.predict(psm_list=psm_list, model=model)

    _write_predictions(psm_list, predictions, output_path)


@cli.command()
@click.option("--native", is_flag=True, default=False, help="Run as native desktop app.")
def gui(native):
    """Launch the DeepLC graphical user interface."""
    from deeplc.gui import main as gui_main

    gui_main(native=native)


def main():
    """Entry point for the DeepLC CLI."""
    cli()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
