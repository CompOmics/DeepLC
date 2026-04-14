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
from deeplc._reference_selection import Q_VALUE_THRESHOLD

logger = logging.getLogger(__name__)

PSM_FILETYPES = list(READERS.keys())
LOGO_PATH = Path(__file__).resolve().parent.parent / "img" / "deeplc_logo.svg"
PRIMARY_COLOR = "#763737"
UPLOAD_ACCEPT = ".csv,.tsv,.txt,.peprec,.mzid,.pepXML,.idXML,.parquet"
MAX_SCATTER_POINTS = 10_000

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
            with ui.card().classes("w-full"), ui.row().classes("items-center gap-6"):
                if LOGO_PATH.exists():
                    ui.image("/static/deeplc_logo.svg").classes("w-24")
                with ui.column().classes("gap-1"):
                    ui.label("DeepLC").classes("text-3xl font-bold")
                    ui.label(
                        "Retention time prediction for peptides carrying any modification"
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

                with ui.expansion("Preview example data", icon="visibility").classes("w-full"):
                    ui.table(
                        columns=[
                            {"name": c, "label": c, "field": c, "align": "left"}
                            for c in ["peptidoform", "retention_time"]
                        ],
                        rows=[
                            {"peptidoform": pf, "retention_time": rt}
                            for pf, rt in EXAMPLE_PEPTIDES
                        ],
                    ).classes("w-full").props("dense")

            # --- Input Section ---
            upload_card = ui.card().classes("w-full")
            with upload_card:
                ui.label("Input").classes("text-xl font-semibold")
                ui.separator()

                with ui.column().classes("gap-1 "):
                    ui.markdown(
                        "Upload a file with peptide-spectrum matches in any format "
                        "supported by [psm_utils](https://psm-utils.readthedocs.io/en/stable/api/psm_utils.io/), "
                        "including MaxQuant msms.txt, Sage, Percolator, MSAmanda, "
                        "mzIdentML, pepXML, and more. A generic "
                        "[TSV format](https://psm-utils.readthedocs.io/en/v1.5.2/api/psm_utils.io/#module-psm_utils.io.tsv) "
                        "is also accepted, requiring `spectrum_id` and `peptidoform` columns, "
                        "and optionally `retention_time` (for calibration), `score`, `qvalue`, "
                        "and `is_decoy` (for auto-calibration)."
                    ).classes("text-sm text-muted-secondary")
                    ui.markdown(
                        "Peptide sequences should use "
                        "[ProForma 2.0](https://www.psidev.info/proforma) notation "
                        "(e.g. `AAINQK[Acetyl]LIETGER/2`). Modification labels must be "
                        "resolvable to atomic compositions, otherwise they will not be "
                        "considered for predictions."
                    ).classes("text-sm text-muted-secondary")

                ui.upload(
                    label="Upload peptide file",
                    auto_upload=True,
                    on_upload=lambda e: _handle_upload(e, state, "psm_file", "psm_file_path"),
                ).classes("w-full").props(f'accept="{UPLOAD_ACCEPT}"')

                psm_type_select = ui.select(
                    label="File type (auto-detected if not set)",
                    options=["auto"] + PSM_FILETYPES,
                    value="auto",
                ).classes("w-64")

            # --- Calibration Section ---
            with ui.card().classes("w-full"):
                ui.label("Calibration").classes("text-xl font-semibold")
                ui.separator()

                ui.markdown(
                    "Calibration maps DeepLC's raw predictions to your specific LC setup, "
                    "significantly improving accuracy. It requires a set of peptides with "
                    "known retention times as reference. You can skip calibration, let "
                    "DeepLC automatically select the best PSMs from your input file, or "
                    "provide a separate reference file."
                ).classes("text-sm text-muted-secondary")

                calibration_mode = (
                    ui.radio(
                        {
                            "none": "No calibration",
                            "auto": "Auto-calibrate from input",
                            "reference": "Provide a reference file",
                        },
                        value="none",
                    )
                    .props("inline")
                    .classes("mt-2")
                )

                # Auto-calibrate description
                auto_calibrate_info = ui.markdown(
                    "Automatically selects the best PSMs from the input file as "
                    "calibration reference, based on q-values or scores. Requires "
                    "PSMs with observed retention times."
                ).classes("text-xs text-muted-hint mt-1")
                auto_calibrate_info.bind_visibility_from(calibration_mode, "value", value="auto")

                # Reference file upload (visible only when "reference" is selected)
                ref_upload_container = ui.column().classes("w-full gap-2 mt-2")
                ref_upload_container.bind_visibility_from(
                    calibration_mode, "value", value="reference"
                )
                with ref_upload_container:
                    ui.markdown(
                        "Upload a reference file with known retention times. Same format "
                        "as the input file, but must include observed retention times in "
                        "the `retention_time` column."
                    ).classes("text-xs text-muted-hint")

                    ui.upload(
                        label="Upload reference file",
                        auto_upload=True,
                        on_upload=lambda e: _handle_upload(e, state, "ref_file", "ref_file_path"),
                    ).classes("w-full").props(f'accept="{UPLOAD_ACCEPT}"')

                    ref_type_select = ui.select(
                        label="File type (auto-detected if not set)",
                        options=["auto"] + PSM_FILETYPES,
                        value="auto",
                    ).classes("w-64")

                # Fine-tune option (visible when calibration is enabled)
                finetune_row = ui.row().classes("items-center gap-1 mt-2")
                finetune_row.bind_visibility_from(
                    calibration_mode, "value", backward=lambda v: v != "none"
                )
                with finetune_row:
                    finetune_switch = ui.switch("Fine-tune model")
                    ui.icon("help_outline", size="xs").classes(
                        "text-muted-hint cursor-help"
                    ).tooltip(
                        "Train the model further on the reference data before predicting. "
                        "Can improve accuracy for your specific LC setup but takes longer."
                    )

            # --- Run Button & Progress ---
            run_button = (
                ui.button(
                    "Predict retention times",
                    icon="play_arrow",
                    on_click=lambda: _run_prediction(
                        state=state,
                        psm_type=psm_type_select.value,
                        ref_type=ref_type_select.value,
                        calibration_mode=calibration_mode.value,
                        finetune=finetune_switch.value,
                        run_button=run_button,
                        progress_card=progress_card,
                        spinner=spinner,
                        status_label=status_label,
                        results_container=results_container,
                    ),
                )
                .classes("w-full text-lg")
                .props('color="primary" size="lg"')
            )

            progress_card = ui.card().classes("w-full")
            progress_card.visible = False
            with progress_card, ui.row().classes("w-full items-center gap-3"):
                spinner = ui.spinner("dots", size="lg", color=PRIMARY_COLOR)
                status_label = ui.label("").classes("text-muted")

            # --- Results Section ---
            results_container = ui.column().classes("w-full gap-4")

        # --- Footer ---
        with (
            ui.footer().classes("text-center text-sm py-3"),
            ui.row().classes("items-center justify-center gap-2"),
        ):
            ui.label(f"DeepLC {__version__}")
            ui.label("|")
            ui.link("GitHub", "https://github.com/compomics/deeplc").classes("!text-white")
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
    """Remove a temporary file and its parent temp directory if it exists."""
    path = state.get(key)
    if path:
        with contextlib.suppress(OSError):
            os.unlink(path)
        with contextlib.suppress(OSError):
            os.rmdir(Path(path).parent)
        state[key] = None


async def _handle_upload(e, state, file_key, path_key):
    """Handle file upload, saving to a temp directory that preserves the original filename."""
    _cleanup_temp_file(state, path_key)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / e.file.name
    await e.file.save(str(tmp_path))
    state[file_key] = e.file.name
    state[path_key] = str(tmp_path)
    ui.notify(f"Uploaded: {e.file.name}", type="positive")


async def _scroll_to(element):
    """Smooth-scroll the viewport to the given UI element."""
    await ui.run_javascript(
        f'document.getElementById("c{element.id}").scrollIntoView({{behavior: "smooth"}})'
    )


async def _read_input_file(path, filetype="auto"):
    """Read a PSM file off the main thread, with optional filetype override."""
    kwargs = {"filetype": filetype} if filetype != "auto" else {}
    return await run.io_bound(read_file, path, **kwargs)


def _subsample(df: pd.DataFrame, max_points: int = MAX_SCATTER_POINTS) -> pd.DataFrame:
    """Randomly subsample a DataFrame if it exceeds max_points."""
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42)


def _add_scatter_trace(fig, df, name, color, opacity=0.3):
    """Add a subsampled scattergl trace to a plotly figure."""
    df = _subsample(df)
    fig.add_scattergl(
        x=df["observed_rt"],
        y=df["predicted_rt"],
        mode="markers",
        name=name,
        marker=dict(size=5, color=color, opacity=opacity),
        hovertext=df["peptidoform"],
    )


async def _run_prediction(
    state,
    psm_type,
    ref_type,
    calibration_mode,
    finetune,
    run_button,
    progress_card,
    spinner,
    status_label,
    results_container,
):
    """Run DeepLC prediction."""
    auto_calibrate = calibration_mode == "auto"
    use_reference = calibration_mode == "reference"

    # --- Quick validation (before starting progress) ---
    if not state.get("use_example"):
        if not state["psm_file_path"]:
            ui.notify("Please upload a peptide file or enable example data.", type="negative")
            return
        if use_reference and not state["ref_file_path"]:
            ui.notify("Please upload a reference file.", type="negative")
            return

    # --- Disable button & show progress ---
    run_button.disable()
    progress_card.visible = True
    spinner.visible = True
    results_container.clear()
    await _scroll_to(progress_card)

    try:
        # --- Resolve inputs ---
        status_label.text = "Reading input files..."
        if state.get("use_example"):
            psm_list = _build_example_psm_list()
            psm_list_reference = psm_list  # Use same data as reference for calibration demo
        else:
            psm_list = await _read_input_file(state["psm_file_path"], psm_type)
            psm_list_reference = None
            if state["ref_file_path"]:
                psm_list_reference = await _read_input_file(state["ref_file_path"], ref_type)

        n_peptides = len(psm_list)

        # --- Auto-calibrate: select reference PSMs off the main thread ---
        if auto_calibrate and not psm_list_reference:
            from deeplc._reference_selection import select_reference_psms

            status_label.text = "Selecting reference PSMs for auto-calibration..."
            psm_list_reference = await run.io_bound(select_reference_psms, psm_list)
            ui.notify(
                f"Auto-calibration: selected {len(psm_list_reference)} reference PSMs.",
                type="positive",
            )

        n_ref = len(psm_list_reference) if psm_list_reference else 0

        # --- Predicting ---
        if psm_list_reference and finetune:
            status_label.text = (
                f"Fine-tuning model on {n_ref} reference peptides, "
                f"then predicting for {n_peptides} peptides..."
            )
            predictions = await run.io_bound(
                deeplc.core.finetune_and_predict,
                psm_list=psm_list,
                psm_list_reference=psm_list_reference,
            )
        elif psm_list_reference:
            status_label.text = (
                f"Predicting for {n_peptides} peptides with {n_ref} reference peptides..."
            )
            predictions = await run.io_bound(
                deeplc.core.predict_and_calibrate,
                psm_list=psm_list,
                psm_list_reference=psm_list_reference,
            )
        else:
            status_label.text = f"Predicting for {n_peptides} peptides..."
            predictions = await run.io_bound(
                deeplc.core.predict,
                psm_list=psm_list,
            )

        # --- Build result DataFrame ---
        result_df = pd.DataFrame(
            {
                "peptidoform": [str(psm.peptidoform) for psm in psm_list],
                "observed_rt": [psm.retention_time for psm in psm_list],
                "predicted_rt": predictions,
                "qvalue": [psm.qvalue for psm in psm_list],
                "is_decoy": [psm.is_decoy for psm in psm_list],
            }
        )
        state["result_df"] = result_df

        spinner.visible = False
        status_label.text = f"Done! Predicted retention times for {len(predictions)} peptides."
        _show_results(results_container, result_df)
        await _scroll_to(results_container)

    except RuntimeError:
        # Client was disconnected (e.g. page reload during prediction)
        logger.debug("Client disconnected during prediction.")

    except Exception as e:
        logger.exception("Prediction failed")
        with contextlib.suppress(RuntimeError):
            spinner.visible = False
            status_label.text = "Prediction failed."
            error_type = type(e).__name__
            with results_container, ui.card().classes("w-full border-l-4 border-red-500"):
                ui.label(error_type).classes("font-bold text-red-600")
                ui.label(str(e)).classes("text-sm text-muted")

    finally:
        with contextlib.suppress(RuntimeError):
            run_button.enable()


def _show_results(container, result_df: pd.DataFrame):
    """Display prediction results."""
    has_observed = bool(result_df["observed_rt"].notna().any())
    has_td = bool(result_df["is_decoy"].notna().any())
    has_qvalues = bool(result_df["qvalue"].notna().any())

    # Determine which PSMs to use for metrics
    valid = result_df.dropna(subset=["observed_rt", "predicted_rt"])
    if has_td or has_qvalues:
        # Use accepted targets only for metrics
        # If TD labels exist, treat unknown as decoy (conservative); otherwise assume target
        accepted = valid[~valid["is_decoy"].fillna(has_td)]
        if has_qvalues:
            accepted = accepted[accepted["qvalue"].fillna(1.0) <= Q_VALUE_THRESHOLD]
    else:
        accepted = valid

    with container:
        # --- Metrics ---
        if has_observed and len(accepted) > 0:
            rt_range = accepted["observed_rt"].max() - accepted["observed_rt"].min()
            mae = np.mean(np.abs(accepted["observed_rt"] - accepted["predicted_rt"]))
            rmae = (mae / rt_range * 100) if rt_range > 0 else float("nan")

            with ui.card().classes("w-full"):
                ui.label("Metrics").classes("text-xl font-semibold")
                ui.separator()
                if has_td or has_qvalues:
                    scope = "accepted target" if has_qvalues else "target"
                    qv = f" (q-value ≤ {Q_VALUE_THRESHOLD})" if has_qvalues else ""
                    ui.label(f"Calculated on {scope} PSMs only{qv}.").classes(
                        "text-xs text-muted-hint"
                    )
                with ui.row().classes("gap-8 mt-2"):
                    with ui.column().classes("items-center"):
                        ui.label(f"{mae:.4f}").classes("text-3xl font-bold")
                        ui.label("MAE").classes("text-sm text-muted-secondary")
                    with ui.column().classes("items-center"):
                        ui.label(f"{rmae:.2f}%").classes("text-3xl font-bold")
                        ui.label("Relative MAE").classes("text-sm text-muted-secondary")
                    with ui.column().classes("items-center"):
                        ui.label(f"{len(accepted)}").classes("text-3xl font-bold")
                        ui.label("Accepted PSMs" if has_qvalues else "Target PSMs").classes(
                            "text-sm text-muted-secondary"
                        )

        # --- Visualization ---
        with ui.card().classes("w-full"):
            ui.label("Visualization").classes("text-xl font-semibold")
            ui.separator()

            if has_observed:
                # Scatter: observed vs predicted
                fig_scatter = _plot_scatter(valid, has_td=has_td, has_qvalues=has_qvalues)
                ui.plotly(fig_scatter).classes("w-full h-96")

                # Baseline comparison
                fig_baseline = _plot_baseline_comparison(accepted)
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

            ui.table(columns=columns, rows=rows, pagination={"rowsPerPage": 20}).classes("w-full")
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


def _plot_scatter(
    valid: pd.DataFrame, has_td: bool = False, has_qvalues: bool = False
) -> go.Figure:
    """Scatter plot of observed vs predicted retention times with diagonal."""
    fig = go.Figure()

    if has_td or has_qvalues:
        # Classify PSMs: treat unknown is_decoy conservatively based on whether TD labels exist
        is_decoy = valid["is_decoy"].fillna(has_td)
        if has_qvalues:
            is_accepted = (~is_decoy) & (valid["qvalue"].fillna(1.0) <= Q_VALUE_THRESHOLD)
            accepted_label = "Target (accepted)"
        else:
            is_accepted = ~is_decoy
            accepted_label = "Target"
        is_rejected_target = (~is_decoy) & (~is_accepted)

        # Plot each category (order: back to front)
        for mask, name, color in [
            (is_decoy, "Decoy", "lightgrey"),
            (is_rejected_target, "Target (not accepted)", "silver"),
            (is_accepted, accepted_label, PRIMARY_COLOR),
        ]:
            if mask.any():
                _add_scatter_trace(fig, valid[mask], name, color)
    else:
        _add_scatter_trace(fig, valid, None, PRIMARY_COLOR)
        fig.update_traces(showlegend=False)

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
    fig.update_layout(
        template="plotly_white",
        title=(
            "Predicted vs. observed retention times"
            + (
                f" (subsampled to {MAX_SCATTER_POINTS:,} points per category)"
                if len(valid) > MAX_SCATTER_POINTS
                else ""
            )
        ),
        xaxis_title="Observed retention time",
        yaxis_title="Predicted retention time",
        showlegend=bool(has_td or has_qvalues),
    )
    return fig


def _plot_baseline_comparison(valid: pd.DataFrame) -> go.Figure | None:
    """
    Plot current relative MAE performance in context of baseline DeepLC runs.

    Adapted from deeplc.plot.distribution_baseline for GUI use.
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

    # Compute relative MAE for baseline runs (using transfer_learning column and diff)
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
    ui.run(title="DeepLC", favicon="/static/deeplc_logo.svg", port=8080, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
