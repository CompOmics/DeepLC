"""NiceGUI-based web interface for DeepLC."""

# TODO: Update to new API

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import pandas as pd
from nicegui import app, ui
from psm_utils.io import FILETYPES, read_file

from deeplc import __version__
from deeplc.calibration import PiecewiseLinearCalibration, SplineTransformerCalibration
from deeplc.core import predict, predict_and_calibrate

logger = logging.getLogger(__name__)


def _get_file_format_options() -> dict[str, str]:
    """Get file format options from psm_utils FILETYPES."""
    formats = {"auto": "Auto-detect"}

    # Add all formats that have readers
    for filetype, properties in FILETYPES.items():
        if properties["reader"] is not None:
            # Create friendly display names
            display_name = filetype.replace("_", " ").title()
            formats[filetype] = display_name

    return formats


class DeepLCGUI:
    """DeepLC graphical user interface."""

    def __init__(self):
        self.prediction_file = None
        self.calibration_file = None
        self.results_df = None
        self.use_calibration = False
        self.calibration_type = "piecewise"
        self.file_format = "auto"  # Auto-detect by default
        self.cal_file_format = "auto"
        self.format_options = _get_file_format_options()

        self._build_ui()

    def _build_ui(self):
        """Build the user interface."""
        # Header
        with ui.header().classes("items-center justify-between"):
            ui.label("DeepLC").classes("text-2xl font-bold")
            ui.label(f"v{__version__}").classes("text-sm text-gray-400")

        # Main content
        with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
            # Info card
            with ui.card().classes("w-full"):
                ui.markdown(
                    """
                    ### Deep Learning Retention Time Prediction
                    
                    Upload your peptide identification file (PSM file) to predict retention times.
                    Supports formats: MaxQuant, mzTab, Percolator, and more via psm_utils.
                    """
                )

            # Input section
            with ui.card().classes("w-full"):
                ui.label("Input Files").classes("text-xl font-semibold mb-2")

                # Prediction file upload
                ui.label("Peptide identification file (required)").classes("font-medium")
                with ui.row().classes("w-full gap-2 items-start"):
                    self.pred_upload = ui.upload(
                        label="Choose file",
                        on_upload=self._handle_prediction_upload,
                        auto_upload=True,
                    ).classes("flex-grow")
                    ui.select(
                        label="Format",
                        options=self.format_options,
                        value="auto",
                        on_change=lambda e: setattr(self, "file_format", e.value),
                    ).classes("w-48")
                self.pred_file_label = ui.label("No file selected").classes(
                    "text-sm text-gray-500"
                )  # Calibration toggle
                self.calibration_switch = ui.switch(
                    "Use calibration (requires observed retention times)",
                    on_change=self._toggle_calibration,
                ).classes("mb-2")

                # Calibration file upload (hidden by default)
                with (
                    ui.column()
                    .classes("w-full")
                    .bind_visibility_from(self, "use_calibration") as self.calibration_section
                ):
                    ui.label(
                        "Calibration file (optional - uses prediction file if not provided)"
                    ).classes("font-medium")
                    with ui.row().classes("w-full gap-2 items-start"):
                        self.cal_upload = ui.upload(
                            label="Choose calibration file",
                            on_upload=self._handle_calibration_upload,
                            auto_upload=True,
                        ).classes("flex-grow")
                        ui.select(
                            label="Format",
                            options=self.format_options,
                            value="auto",
                            on_change=lambda e: setattr(self, "cal_file_format", e.value),
                        ).classes("w-48")
                    self.cal_file_label = ui.label(
                        "Using prediction file for calibration"
                    ).classes("text-sm text-gray-500")

                    # Calibration type selector
                    ui.label("Calibration method").classes("font-medium mt-2")
                    ui.radio(
                        ["piecewise", "spline"],
                        value="piecewise",
                        on_change=lambda e: setattr(self, "calibration_type", e.value),
                    ).props("inline").classes("mb-2")
                    ui.label(
                        "Piecewise: faster, good for linear relationships | "
                        "Spline: more flexible, better for complex patterns"
                    ).classes("text-xs text-gray-500")

            # Run button
            with ui.row().classes("w-full justify-center"):
                self.run_button = (
                    ui.button(
                        "Predict Retention Times",
                        on_click=self._run_prediction,
                        icon="play_arrow",
                    )
                    .classes("text-lg")
                    .props("color=primary size=lg")
                )

            # Results section
            self.results_container = ui.column().classes("w-full")

    def _toggle_calibration(self, e):
        """Toggle calibration section visibility."""
        self.use_calibration = e.value

    async def _handle_prediction_upload(self, e):
        """Handle prediction file upload."""
        try:
            # Store the file content as bytes
            content = await e.file.read()
            self.prediction_file = content
            logger.info(f"Uploaded prediction file: {e.file.name}, size: {len(content)} bytes")
            self.pred_file_label.set_text(f"✓ {e.file.name}")
            self.pred_file_label.classes(replace="text-sm text-green-600")
            ui.notify(f"File uploaded: {e.file.name}", type="positive")
        except Exception as ex:
            logger.exception("Error reading uploaded file")
            self.pred_file_label.set_text(f"Error: {str(ex)}")
            self.pred_file_label.classes(replace="text-sm text-red-600")
            ui.notify(f"Error uploading file: {str(ex)}", type="negative")

    async def _handle_calibration_upload(self, e):
        """Handle calibration file upload."""
        try:
            content = await e.file.read()
            self.calibration_file = content
            logger.info(f"Uploaded calibration file: {e.file.name}, size: {len(content)} bytes")
            self.cal_file_label.set_text(f"✓ {e.file.name}")
            self.cal_file_label.classes(replace="text-sm text-green-600")
            ui.notify(f"Calibration file uploaded: {e.file.name}", type="positive")
        except Exception as ex:
            logger.exception("Error reading calibration file")
            self.cal_file_label.set_text(f"Error: {str(ex)}")
            self.cal_file_label.classes(replace="text-sm text-red-600")
            ui.notify(f"Error uploading calibration file: {str(ex)}", type="negative")

    async def _run_prediction(self):
        """Run DeepLC prediction."""
        # Validation
        logger.info(f"Run prediction clicked. prediction_file: {self.prediction_file is not None}")
        if not self.prediction_file:
            ui.notify("Please upload a peptide identification file", type="negative")
            logger.warning("No prediction file uploaded")
            return

        # Clear previous results
        self.results_container.clear()

        # Show progress
        with self.results_container:
            progress_card = ui.card().classes("w-full")
            with progress_card:
                ui.label("Running DeepLC...").classes("text-lg font-semibold")
                spinner = ui.spinner(size="lg")

        try:
            # Parse input files
            logger.info("Reading prediction file...")
            file_format = None if self.file_format == "auto" else self.file_format

            # psm_utils requires a file path, so write to temp file
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".psm") as tmp_file:
                tmp_file.write(self.prediction_file)
                tmp_pred_path = tmp_file.name

            try:
                psm_list = read_file(tmp_pred_path, filetype=file_format)
            finally:
                Path(tmp_pred_path).unlink(missing_ok=True)

            if self.use_calibration:
                # Determine calibration source
                if self.calibration_file:
                    logger.info("Reading calibration file...")
                    cal_format = None if self.cal_file_format == "auto" else self.cal_file_format

                    # Write calibration file to temp file
                    with tempfile.NamedTemporaryFile(
                        mode="wb", delete=False, suffix=".psm"
                    ) as tmp_file:
                        tmp_file.write(self.calibration_file)
                        tmp_cal_path = tmp_file.name

                    try:
                        psm_list_cal = read_file(tmp_cal_path, filetype=cal_format)
                    finally:
                        Path(tmp_cal_path).unlink(missing_ok=True)
                else:
                    logger.info("Using prediction file for calibration...")
                    psm_list_cal = psm_list

                # Select calibration method
                if self.calibration_type == "spline":
                    calibration = SplineTransformerCalibration()
                else:
                    calibration = PiecewiseLinearCalibration(number_of_splits=50)

                # Run with calibration
                logger.info("Running prediction with calibration...")
                predictions = await self._run_in_background(
                    predict_and_calibrate,
                    psm_list=psm_list,
                    psm_list_reference=psm_list_cal,
                    calibration=calibration,
                )
            else:
                # Run without calibration
                logger.info("Running prediction without calibration...")
                predictions = await self._run_in_background(predict, psm_list=psm_list)

            # Prepare results DataFrame
            self.results_df = pd.DataFrame(
                {
                    "peptidoform": [psm.peptidoform.proforma for psm in psm_list],
                    "predicted_rt": predictions,
                }
            )

            # Add observed RT if available
            observed_rts = [
                psm.retention_time for psm in psm_list if psm.retention_time is not None
            ]
            if observed_rts and len(observed_rts) == len(psm_list):
                self.results_df["observed_rt"] = observed_rts

            # Clear progress and show results
            progress_card.delete()
            self._show_results()

        except Exception as e:
            logger.exception("Error during prediction")
            progress_card.delete()
            with self.results_container:
                with ui.card().classes("w-full bg-red-50"):
                    ui.label("Error").classes("text-xl font-semibold text-red-600")
                    ui.label(str(e)).classes("text-red-800")

    async def _run_in_background(self, func, **kwargs):
        """Run prediction function in background to avoid blocking UI."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))

    def _show_results(self):
        """Display prediction results."""
        with self.results_container:
            # Success message
            with ui.card().classes("w-full bg-green-50"):
                ui.label("✓ Prediction Complete").classes("text-xl font-semibold text-green-600")
                ui.label(f"Predicted retention times for {len(self.results_df)} peptides").classes(
                    "text-green-800"
                )

            # Results table
            with ui.card().classes("w-full"):
                ui.label("Results Preview").classes("text-xl font-semibold mb-2")

                # Create columns for the table
                columns = [
                    {"name": col, "label": col.replace("_", " ").title(), "field": col}
                    for col in self.results_df.columns
                ]

                # Convert DataFrame to list of dictionaries for the table
                rows = self.results_df.to_dict("records")

                # Display paginated table
                ui.table(
                    columns=columns,
                    rows=rows,
                    row_key="peptidoform",
                    pagination={"rowsPerPage": 10, "sortBy": None, "descending": False},
                ).classes("w-full")

            # Plot if observed RT is available
            if "observed_rt" in self.results_df.columns:
                with ui.card().classes("w-full"):
                    ui.label("Observed vs Predicted").classes("text-xl font-semibold mb-2")
                    self._plot_correlation()

            # Download button
            with ui.card().classes("w-full"):
                ui.label("Download Results").classes("text-xl font-semibold mb-2")
                csv_data = self.results_df.to_csv(index=False)
                ui.button(
                    "Download CSV",
                    on_click=lambda: ui.download(csv_data.encode(), "deeplc_predictions.csv"),
                    icon="download",
                ).props("color=primary")

    def _plot_correlation(self):
        """Plot observed vs predicted retention times using Plotly."""
        import numpy as np
        import plotly.graph_objects as go

        # Calculate metrics
        observed = self.results_df["observed_rt"].values
        predicted = self.results_df["predicted_rt"].values
        mae = np.mean(np.abs(observed - predicted))
        r2 = np.corrcoef(observed, predicted)[0, 1] ** 2

        # Create scatter plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=observed,
                y=predicted,
                mode="markers",
                name="Predictions",
                marker=dict(size=5, opacity=0.5, color="blue"),
                hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
            )
        )

        # Add diagonal line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="y=x",
                line=dict(color="red", dash="dash", width=2),
            )
        )

        fig.update_layout(
            title=f"Retention Time Prediction Performance (R² = {r2:.3f}, MAE = {mae:.2f})",
            xaxis_title="Observed RT",
            yaxis_title="Predicted RT",
            hovermode="closest",
            height=500,
            showlegend=True,
        )

        ui.plotly(fig).classes("w-full")


def start_gui(host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
    """Start the DeepLC NiceGUI interface."""
    DeepLCGUI()

    ui.run(
        host=host,
        port=port,
        title="DeepLC - Retention Time Prediction",
        reload=reload,
        favicon="🧬",
    )


if __name__ in {"__main__", "__mp_main__"}:
    start_gui()
