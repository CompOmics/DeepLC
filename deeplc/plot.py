# TODO: Update for new core functions and new data structures

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter(
    df: pd.DataFrame,
    predicted_column: str = "Predicted retention time",
    observed_column: str = "Observed retention time",
    xaxis_label: str = "Observed retention time",
    yaxis_label: str = "Predicted retention time",
    plot_title: str = "Predicted vs. observed retention times",
) -> go.Figure:
    """
    Plot a scatter plot of the predicted vs. observed retention times.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the predicted and observed retention times.
    predicted_column : str, optional
        Name of the column containing the predicted retention times, by default
        ``Predicted retention time``.
    observed_column : str, optional
        Name of the column containing the observed retention times, by default
        ``Observed retention time``.
    xaxis_label : str, optional
        X-axis label, by default ``Observed retention time``.
    yaxis_label : str, optional
        Y-axis label, by default ``Predicted retention time``.
    plot_title : str, optional
        Scatter plot title, by default ``Predicted vs. observed retention times``

    """
    # Draw scatter
    fig = px.scatter(
        df,
        x=observed_column,
        y=predicted_column,
        opacity=0.3,
    )

    # Draw diagonal line
    fig.add_scatter(
        x=[min(df[observed_column]), max(df[observed_column])],
        y=[min(df[observed_column]), max(df[observed_column])],
        mode="lines",
        line=dict(color="red", width=3, dash="dash"),
    )

    # Hide legend
    fig.update_layout(
        title=plot_title,
        showlegend=False,
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_label,
    )

    return fig


def distribution_baseline(
    df: pd.DataFrame,
    predicted_column: str = "Predicted retention time",
    observed_column: str = "Observed retention time",
) -> go.Figure:
    """
    Plot a distribution plot of the relative mean absolute error of the current
    DeepLC performance compared to the baseline performance.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the predicted and observed retention times.
    predicted_column : str, optional
        Name of the column containing the predicted retention times, by default
        ``Predicted retention time``.
    observed_column : str, optional
        Name of the column containing the observed retention times, by default
        ``Observed retention time``.

    """
    # Get baseline data
    baseline_df = pd.read_csv(
        Path(__file__).absolute().parent.joinpath("baseline_performance/baseline_predictions.csv")
    )
    baseline_df["rel_mae_best"] = baseline_df[
        ["rel_mae_transfer_learning", "rel_mae_new_model", "rel_mae_calibrate"]
    ].min(axis=1)
    baseline_df.fillna(0.0, inplace=True)

    # Calculate current RMAE and percentile compared to baseline
    mae = sum(abs(df[observed_column] - df[predicted_column])) / len(df.index)
    mae_rel = (mae / max(df[observed_column])) * 100
    percentile = round((baseline_df["rel_mae_transfer_learning"] < mae_rel).mean() * 100, 1)

    # Calculate x-axis range with 5% padding
    all_values = np.append(baseline_df["rel_mae_transfer_learning"].to_numpy(), mae_rel)
    padding = (all_values.max() - all_values.min()) / 20  # 5% padding
    x_min = all_values.min() - padding
    x_max = all_values.max() + padding

    # Make labels human-readable
    hover_label_mapping = {
        "train_number": "Training dataset size",
        "rel_mae_transfer_learning": "RMAE with transfer learning",
        "rel_mae_new_model": "RMAE with new model from scratch",
        "rel_mae_calibrate": "RMAE with calibrating existing model",
        "rel_mae_best": "RMAE with best method",
    }
    label_mapping = hover_label_mapping.copy()
    label_mapping.update({"Unnamed: 0": "Dataset"})

    # Generate plot
    fig = px.histogram(
        data_frame=baseline_df,
        x="rel_mae_best",
        marginal="rug",
        hover_data=hover_label_mapping.keys(),
        hover_name="Unnamed: 0",
        labels=label_mapping,
        opacity=0.8,
    )
    fig.add_vline(
        x=mae_rel,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current performance (percentile {percentile}%)",
        annotation_position="top left",
        name="Current performance",
        row=1,
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_layout(
        title=(f"Current DeepLC performance compared to {len(baseline_df.index)} datasets"),
        xaxis_title="Relative mean absolute error (%)",
    )

    return fig
