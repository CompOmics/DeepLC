"""NiceGUI-based web interface for DeepLC."""

import contextlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nicegui import app, run, ui
from psm_utils import PSM, PSMList
from psm_utils.io import READERS, read_file

import deeplc.core
from deeplc import __version__

logger = logging.getLogger(__name__)

PSM_FILETYPES = list(READERS.keys())
LOGO_PATH = Path(__file__).resolve().parent.parent / "img" / "deeplc_logo.svg"
PRIMARY_COLOR = "#763737"

EXAMPLE_PEPTIDES = [
    ("AAGPSLSHTSGGTQSK/2", 12.16),
    ("AAINQK[Acetyl]LIETGER/2", 34.10),
    ("AANDAGYFNDEM[Oxidation]APIEVK[Acetyl]TK/3", 37.38),
    ("AAPFSPAEK/2", 16.89),
    ("AAYFGILEK/2", 30.93),
    ("ADTQLDESSEQIDEEELTSK/2", 28.33),
    ("AGFAGDDAPR/2", 7.57),
    ("AHQVVEDGYEFFAK/2", 28.86),
    ("AIQEYNQDK/2", 8.44),
    ("ALDQFVNFSEQK/2", 32.42),
]


def create_app():
    """Create and configure the NiceGUI app."""

    # Serve logo
    if LOGO_PATH.exists():
        app.add_static_files("/static", str(LOGO_PATH.parent))

    # State
    state = {
        "psm_file": None,
        "psm_file_path": None,
        "ref_file": None,
        "ref_file_path": None,
        "use_example": False,
        "result_df": None,
    }

    @ui.page("/")
    def main_page():
        # --- Set primary color, load Font Awesome, and define muted text classes ---
        ui.colors(primary=PRIMARY_COLOR)
        ui.add_head_html(
            '<link rel="stylesheet" '
            'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">'
        )
        ui.add_css("""
            .text-muted { color: #4b5563; }
            .body--dark .text-muted { color: #d1d5db; }
            .text-muted-secondary { color: #6b7280; }
            .body--dark .text-muted-secondary { color: #9ca3af; }
            .text-muted-hint { color: #9ca3af; }
            .body--dark .text-muted-hint { color: #6b7280; }
        """)

        # --- Header bar ---
        dark = ui.dark_mode(False)
        with ui.header().classes("items-center justify-between shadow-sm"):
            with ui.row().classes("items-center gap-2"):
                ui.label("DeepLC").classes("text-2xl font-bold")
                ui.badge(f"v{__version__}").props("outline color=white")
            with ui.row().classes("items-center gap-1"):
                ui.button(
                    icon="fa-brands fa-github",
                    on_click=lambda: ui.navigate.to(
                        "https://github.com/compomics/deeplc", new_tab=True
                    ),
                ).props("flat round color=white")
                ui.button(
                    icon="dark_mode",
                    on_click=lambda: dark.toggle(),
                ).props("flat round color=white")

        with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-6"):
            # --- Branding / hero ---
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center gap-6"):
                    if LOGO_PATH.exists():
                        ui.image("/static/deeplc_logo.svg").classes("w-24")
                    with ui.column().classes("gap-1"):
                        ui.label("DeepLC").classes("text-3xl font-bold")
                        ui.label(
                            "Retention time prediction for peptides carrying any modification, "
                            "powered by deep learning."
                        ).classes("text-muted")
                        with ui.row().classes("gap-3 mt-1 items-center"):
                            ui.link(
                                "GitHub",
                                "https://github.com/compomics/deeplc",
                            ).classes("text-sm")
                            ui.link(
                                "PyPI",
                                "https://pypi.org/project/deeplc/",
                            ).classes("text-sm")
                            ui.link(
                                "Bouwmeester et al., Nat Methods 2021",
                                "https://doi.org/10.1038/s41592-021-01301-5",
                            ).classes("text-sm")

            # --- Example Data ---
            with ui.card().classes("w-full"):
                ui.label("Quick start").classes("text-xl font-semibold")
                ui.separator()
                ui.label(
                    "Try DeepLC instantly with built-in example peptides, "
                    "or upload your own files below."
                ).classes("text-sm text-muted-secondary mt-1")

                example_switch = ui.switch("Use example data").on_value_change(
                    lambda e: _toggle_example(e.value, state, upload_card)
                )

                with ui.expansion("Preview example data", icon="visibility").classes(
                    "w-full"
                ):
                    example_columns = [
                        {
                            "name": "peptidoform",
                            "label": "peptidoform",
                            "field": "peptidoform",
                            "align": "left",
                        },
                        {
                            "name": "retention_time",
                            "label": "retention_time",
                            "field": "retention_time",
                            "align": "left",
                        },
                    ]
                    example_rows = [
                        {"peptidoform": pf, "retention_time": rt}
                        for pf, rt in EXAMPLE_PEPTIDES
                    ]
                    ui.table(
                        columns=example_columns,
                        rows=example_rows,
                    ).classes("w-full").props("dense")

            # --- Input Section ---
            upload_card = ui.card().classes("w-full")
            with upload_card:
                ui.label("Input").classes("text-xl font-semibold")
                ui.separator()

                # PSM file upload
                with ui.row().classes("items-center gap-1 mt-2"):
                    ui.label("Peptide file").classes("font-medium")
                    ui.icon("help_outline", size="xs").classes(
                        "text-muted-hint cursor-help"
                    ).tooltip(
                        "Any PSM format supported by psm_utils: CSV/TSV, MaxQuant msms.txt, "
                        "Sage, Percolator, MSAmanda, mzIdentML, pepXML, and more."
                    )
                ui.label(
                    "Peptide sequences in ProForma 2.0 notation, "
                    "e.g. AAINQK[Acetyl]LIETGER/2"
                ).classes("text-xs text-muted-hint")

                ui.upload(
                    label="Upload peptide file",
                    auto_upload=True,
                    on_upload=lambda e: _handle_psm_upload(e, state),
                ).classes("w-full").props(
                    'accept=".csv,.tsv,.txt,.peprec,.mzid,.pepXML,.idXML,.parquet"'
                )

                psm_type_select = ui.select(
                    label="File type (auto-detected if not set)",
                    options=["auto"] + PSM_FILETYPES,
                    value="auto",
                ).classes("w-64")

                # Reference file upload
                with ui.row().classes("items-center gap-1 mt-4"):
                    ui.label("Reference file (optional)").classes("font-medium")
                    ui.icon("help_outline", size="xs").classes(
                        "text-muted-hint cursor-help"
                    ).tooltip(
                        "A file with known retention times used for calibration. "
                        "This improves prediction accuracy by mapping raw predictions "
                        "to your LC setup. Must include a retention_time column."
                    )
                ui.label(
                    "Same format as above, but must include observed retention times."
                ).classes("text-xs text-muted-hint")

                ui.upload(
                    label="Upload reference file",
                    auto_upload=True,
                    on_upload=lambda e: _handle_ref_upload(e, state),
                ).classes("w-full").props(
                    'accept=".csv,.tsv,.txt,.peprec,.mzid,.pepXML,.idXML,.parquet"'
                )

                ref_type_select = ui.select(
                    label="File type (auto-detected if not set)",
                    options=["auto"] + PSM_FILETYPES,
                    value="auto",
                ).classes("w-64")

            # --- Options Section ---
            with ui.card().classes("w-full"):
                ui.label("Options").classes("text-xl font-semibold")
                ui.separator()

                with ui.row().classes("items-center gap-1 mt-2"):
                    finetune_switch = ui.switch("Fine-tune model to reference")
                    ui.icon("help_outline", size="xs").classes(
                        "text-muted-hint cursor-help"
                    ).tooltip(
                        "Train the model further on your reference data before predicting. "
                        "Can improve accuracy for your specific LC setup but takes longer. "
                        "Requires a reference file."
                    )

            # --- Run Button & Progress ---
            run_button = ui.button(
                "Predict retention times",
                icon="play_arrow",
                on_click=lambda: _run_prediction(
                    state=state,
                    psm_type=psm_type_select.value,
                    ref_type=ref_type_select.value,
                    finetune=finetune_switch.value,
                    run_button=run_button,
                    progress_row=progress_row,
                    spinner=spinner,
                    status_label=status_label,
                    results_container=results_container,
                ),
            ).classes("w-full text-lg").props('color="primary" size="lg"')

            progress_row = ui.row().classes("w-full items-center gap-3 mt-2")
            progress_row.visible = False
            with progress_row:
                spinner = ui.spinner("dots", size="lg", color=PRIMARY_COLOR)
                status_label = ui.label("").classes("text-muted")

            # --- Results Section ---
            results_container = ui.column().classes("w-full gap-4")

        # --- Footer ---
        with ui.footer().classes("text-center text-sm py-3"):
            with ui.row().classes("items-center justify-center gap-2"):
                ui.label(f"DeepLC {__version__}")
                ui.label("|")
                ui.link("GitHub", "https://github.com/compomics/deeplc").classes(
                    "!text-white"
                )
                ui.label("|")
                ui.link(
                    "Bouwmeester et al., Nature Methods 2021",
                    "https://doi.org/10.1038/s41592-021-01301-5",
                ).classes("!text-white")


def _toggle_example(use_example: bool, state: dict, upload_card):
    """Toggle example data mode and disable/enable upload card."""
    state["use_example"] = use_example
    upload_card.visible = not use_example


def _build_example_psm_list() -> PSMList:
    """Build a PSMList from the built-in example peptides."""
    return PSMList(
        psm_list=[
            PSM(
                spectrum_id=str(i),
                peptidoform=peptidoform,
                retention_time=rt,
            )
            for i, (peptidoform, rt) in enumerate(EXAMPLE_PEPTIDES)
        ]
    )


def _cleanup_temp_file(state, key):
    """Remove a temporary file if it exists."""
    path = state.get(key)
    if path:
        with contextlib.suppress(OSError):
            os.unlink(path)
        state[key] = None


async def _handle_psm_upload(e, state):
    """Handle PSM file upload."""
    _cleanup_temp_file(state, "psm_file_path")
    suffix = Path(e.file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    await e.file.save(tmp.name)
    state["psm_file"] = e.file.name
    state["psm_file_path"] = tmp.name
    ui.notify(f"Uploaded: {e.file.name}", type="positive")


async def _handle_ref_upload(e, state):
    """Handle reference file upload."""
    _cleanup_temp_file(state, "ref_file_path")
    suffix = Path(e.file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    await e.file.save(tmp.name)
    state["ref_file"] = e.file.name
    state["ref_file_path"] = tmp.name
    ui.notify(f"Uploaded: {e.file.name}", type="positive")


async def _run_prediction(
    state, psm_type, ref_type, finetune, run_button, progress_row, spinner, status_label,
    results_container,
):
    """Run DeepLC prediction."""
    # --- Quick validation (before starting progress) ---
    if not state.get("use_example"):
        if not state["psm_file_path"]:
            ui.notify("Please upload a peptide file or enable example data.", type="negative")
            return
        if finetune and not state["ref_file_path"]:
            ui.notify("Fine-tuning requires a reference file.", type="negative")
            return

    # --- Disable button & show progress ---
    run_button.disable()
    progress_row.visible = True
    spinner.visible = True
    results_container.clear()

    try:
        # --- Resolve inputs ---
        status_label.text = "Reading input files..."
        if state.get("use_example"):
            psm_list = _build_example_psm_list()
            psm_list_reference = psm_list  # Use same data as reference for calibration demo
        else:
            psm_filetype = psm_type if psm_type != "auto" else None
            kwargs = {"filetype": psm_filetype} if psm_filetype else {}
            psm_list = read_file(state["psm_file_path"], **kwargs)
            psm_list_reference = None
            if state["ref_file_path"]:
                ref_filetype = ref_type if ref_type != "auto" else None
                ref_kwargs = {"filetype": ref_filetype} if ref_filetype else {}
                psm_list_reference = read_file(state["ref_file_path"], **ref_kwargs)

        n_peptides = len(psm_list)
        n_ref = len(psm_list_reference) if psm_list_reference else 0

        # --- Predicting ---
        if n_ref:
            status_label.text = (
                f"Predicting for {n_peptides} peptides "
                f"with {n_ref} reference peptides..."
            )
        else:
            status_label.text = f"Predicting for {n_peptides} peptides..."

        if psm_list_reference and finetune:
            status_label.text = (
                f"Fine-tuning model on {n_ref} reference peptides, "
                f"then predicting for {n_peptides} peptides..."
            )
            predictions = await run.cpu_bound(
                deeplc.core.finetune_and_predict,
                psm_list=psm_list,
                psm_list_reference=psm_list_reference,
            )
        elif psm_list_reference:
            predictions = await run.cpu_bound(
                deeplc.core.predict_and_calibrate,
                psm_list=psm_list,
                psm_list_reference=psm_list_reference,
            )
        else:
            predictions = await run.cpu_bound(
                deeplc.core.predict,
                psm_list=psm_list,
            )

        # --- Build result DataFrame ---
        result_df = pd.DataFrame(
            {
                "peptidoform": [str(psm.peptidoform) for psm in psm_list],
                "observed_rt": [psm.retention_time for psm in psm_list],
                "predicted_rt": predictions,
            }
        )
        state["result_df"] = result_df

        spinner.visible = False
        status_label.text = f"Done! Predicted retention times for {len(predictions)} peptides."
        _show_results(results_container, result_df)

    except Exception as e:
        logger.exception("Prediction failed")
        spinner.visible = False
        status_label.text = f"Error: {e}"
        ui.notify(f"Prediction failed: {e}", type="negative", close_button=True)

    finally:
        run_button.enable()
        _cleanup_temp_file(state, "psm_file_path")
        _cleanup_temp_file(state, "ref_file_path")


def _show_results(container, result_df: pd.DataFrame):
    """Display prediction results."""
    has_observed = result_df["observed_rt"].notna().any()

    with container:
        # --- Metrics ---
        if has_observed:
            valid = result_df.dropna(subset=["observed_rt", "predicted_rt"])
            rmse = np.sqrt(np.mean((valid["observed_rt"] - valid["predicted_rt"]) ** 2))

            with ui.card().classes("w-full"):
                ui.label("Metrics").classes("text-xl font-semibold")
                ui.separator()
                with ui.row().classes("gap-8 mt-2"):
                    with ui.column().classes("items-center"):
                        ui.label(f"{rmse:.4f}").classes("text-3xl font-bold")
                        ui.label("RMSE").classes(
                            "text-sm text-muted-secondary"
                        )
                    with ui.column().classes("items-center"):
                        ui.label(f"{len(valid)}").classes("text-3xl font-bold")
                        ui.label("Peptides with observed RT").classes(
                            "text-sm text-muted-secondary"
                        )

        # --- Visualization ---
        with ui.card().classes("w-full"):
            ui.label("Visualization").classes("text-xl font-semibold")
            ui.separator()

            if has_observed:
                # Scatter: observed vs predicted
                fig_scatter = _plot_scatter(valid)
                ui.plotly(fig_scatter).classes("w-full h-96")

                # Baseline comparison
                fig_baseline = _plot_baseline_comparison(valid)
                if fig_baseline is not None:
                    ui.separator()
                    ui.plotly(fig_baseline).classes("w-full h-96")
            else:
                fig = px.histogram(
                    result_df,
                    x="predicted_rt",
                    marginal="rug",
                    opacity=0.8,
                    histnorm="density",
                    color_discrete_sequence=[PRIMARY_COLOR],
                    labels={"predicted_rt": "Predicted retention time"},
                    title="Predicted retention time distribution",
                )
                fig.update_layout(
                    yaxis_title_text="Density",
                    bargap=0.2,
                    template="plotly_white",
                )
                ui.plotly(fig).classes("w-full h-96")

        # --- Table ---
        with ui.card().classes("w-full"):
            ui.label("Predictions").classes("text-xl font-semibold")
            ui.separator()

            columns = [
                {"name": col, "label": col, "field": col, "sortable": True, "align": "left"}
                for col in result_df.columns
            ]
            rows = result_df.head(200).to_dict("records")
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, float) and np.isnan(value):
                        row[key] = None
                    elif isinstance(value, (np.floating, np.integer)):
                        row[key] = float(value)

            ui.table(columns=columns, rows=rows, pagination={"rowsPerPage": 20}).classes(
                "w-full"
            )
            if len(result_df) > 200:
                ui.label(f"Showing first 200 of {len(result_df)} rows.").classes(
                    "text-sm text-muted-hint"
                )

        # --- Download ---
        with ui.card().classes("w-full"):
            ui.label("Download").classes("text-xl font-semibold")
            ui.separator()

            csv_data = result_df.to_csv(index=False)
            ui.button(
                "Download predictions as CSV",
                icon="download",
                on_click=lambda: ui.download(csv_data.encode(), "deeplc_predictions.csv"),
            ).props('color="primary"')


def _plot_scatter(valid: pd.DataFrame) -> go.Figure:
    """Scatter plot of observed vs predicted retention times with diagonal."""
    fig = px.scatter(
        valid,
        x="observed_rt",
        y="predicted_rt",
        hover_data=["peptidoform"],
        opacity=0.3,
        color_discrete_sequence=[PRIMARY_COLOR],
        labels={
            "observed_rt": "Observed retention time",
            "predicted_rt": "Predicted retention time",
        },
        title="Predicted vs. observed retention times",
    )
    fig.update_traces(marker=dict(size=5))

    # Diagonal reference line
    axis_min = min(valid["observed_rt"].min(), valid["predicted_rt"].min())
    axis_max = max(valid["observed_rt"].max(), valid["predicted_rt"].max())
    fig.add_scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        showlegend=False,
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def _plot_baseline_comparison(valid: pd.DataFrame) -> go.Figure | None:
    """
    Plot current RMSE performance in context of baseline DeepLC runs.

    Adapted from deeplc.plot.distribution_baseline, using RMSE instead of relative MAE.
    """
    baseline_path = (
        Path(__file__).resolve().parent
        / "package_data"
        / "baseline_performance"
        / "baseline_predictions.csv"
    )
    if not baseline_path.exists():
        return None

    baseline_df = pd.read_csv(baseline_path)

    # Compute relative RMSE for baseline runs (using transfer_learning column and diff)
    # Baseline CSV has columns: calibrate, new_model, transfer_learning, diff
    # We compute relative error as percentage of the RT range (diff column)
    baseline_df["rel_mae_best"] = (
        baseline_df[["calibrate", "new_model", "transfer_learning"]].min(axis=1)
        / baseline_df["diff"]
        * 100
    )
    baseline_df = baseline_df.dropna(subset=["rel_mae_best"])

    # Current run: relative MAE as % of observed RT range (matching baseline methodology)
    rt_range = valid["observed_rt"].max() - valid["observed_rt"].min()
    if rt_range == 0:
        return None
    mae = np.mean(np.abs(valid["observed_rt"] - valid["predicted_rt"]))
    current_rel_mae = (mae / rt_range) * 100
    percentile = round((baseline_df["rel_mae_best"] > current_rel_mae).mean() * 100, 1)

    # X-axis range with padding
    all_values = np.append(baseline_df["rel_mae_best"].to_numpy(), current_rel_mae)
    padding = (all_values.max() - all_values.min()) / 20
    x_min = max(0, all_values.min() - padding)
    x_max = all_values.max() + padding

    fig = px.histogram(
        baseline_df,
        x="rel_mae_best",
        marginal="rug",
        opacity=0.8,
        color_discrete_sequence=[PRIMARY_COLOR],
        labels={"rel_mae_best": "Relative MAE (%)"},
    )
    fig.add_vline(
        x=current_rel_mae,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Your result (better than {percentile}% of runs)",
        annotation_position="top left",
        row=1,
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_layout(
        title=f"Your performance vs. {len(baseline_df)} baseline datasets",
        xaxis_title="Relative mean absolute error (%)",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def main():
    """Run the DeepLC GUI."""
    logging.basicConfig(level=logging.INFO)
    create_app()
    ui.run(title="DeepLC", favicon="🧬", port=8080, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
