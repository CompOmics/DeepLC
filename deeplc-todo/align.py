"""Align retention times across LC-MS runs from a Sage results TSV.

This script ports the end-to-end logic from `align.ipynb` into a runnable CLI:

* Read `results.plasma_proteomics.tsv` using psm_utils:
    `psm_list = read_file(path, filetype="sage")`
* Build a peptidoform × run RT matrix (median aggregation), ignoring charge.
* Rank runs by summed pairwise overlap (# shared peptidoforms).
* Iteratively align runs (rank order) onto a growing reference scale using a
  spline regression model (scikit-learn).
* Emit calibrated RT matrix + reference table + diagnostic table.
* Optional plots:
  - run overlap bar chart
  - per-alignment diagnostic plots
  - before/after SD across runs
* Optional filtering by occurrence (%) and SD thresholds (absolute and/or
  relative to gradient length).

Outputs are written into the directory given by `--outdir`.

Notes
-----
RT units are whatever Sage reports (often minutes). Ensure `--gradient-length`
uses the same units.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


# -------------------------
# I/O (psm_utils -> DataFrame)
# -------------------------


def load_psm_list(tsv_path: str | Path):
    """Load Sage TSV results into a psm_utils PSMList."""

    try:
        from psm_utils.io import read_file
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "psm_utils is not installed. Install dependencies from requirements.txt and retry."
        ) from e

    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {tsv_path}")

    return read_file(str(tsv_path), filetype="sage")


def psm_list_to_long_df(psm_list) -> pd.DataFrame:
    """Convert psm_utils PSMList into a long table with (run, peptidoform, rt).

    We ignore charge state by keeping only the string form of peptidoform/peptide.
    """

    rows: list[dict[str, Any]] = []
    for psm in psm_list:
        run = (
            getattr(psm, "filename", None)
            or getattr(psm, "run", None)
            or getattr(psm, "spectrum_id", None)
        )
        pep = (
            getattr(psm, "peptidoform", None)
            or getattr(psm, "peptide", None)
            or getattr(psm, "sequence", None)
        )
        rt = getattr(psm, "retention_time", None) or getattr(psm, "rt", None)

        if run is None or pep is None or rt is None:
            continue

        rows.append({"run": str(run), "peptidoform": str(pep), "rt": float(rt)})

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(
            "No usable PSM rows were extracted. Check that your Sage TSV contains RT and run identifiers "
            "and that psm_utils mapped them correctly."
        )
    return long_df


def build_rt_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to peptidoform×run matrix of retention times (median aggregation)."""

    return long_df.pivot_table(
        index="peptidoform",
        columns="run",
        values="rt",
        aggfunc="median",
        observed=True,
        dropna=False,
    )


# -------------------------
# Run overlap ranking
# -------------------------


def rank_runs_by_overlap(rt_matrix: pd.DataFrame) -> pd.DataFrame:
    """Rank runs by summed pairwise overlap (shared peptidoforms)."""

    present = rt_matrix.notna()
    X = present.astype("uint8")
    overlap = X.T.dot(X)  # run x run

    scores = overlap.sum(axis=1) - pd.Series(
        overlap.values.diagonal(), index=overlap.index
    )
    n_peptidoforms = present.sum(axis=0)
    avg_shared = scores / (len(overlap.index) - 1) if len(overlap.index) > 1 else scores

    run_ranking = (
        pd.DataFrame(
            {
                "run": overlap.index,
                "overlap_score": scores.astype(int).values,
                "avg_shared": avg_shared.values,
                "n_peptidoforms": n_peptidoforms.loc[overlap.index].astype(int).values,
            }
        )
        .sort_values(["overlap_score", "n_peptidoforms"], ascending=False)
        .reset_index(drop=True)
    )
    run_ranking.insert(0, "rank", run_ranking.index + 1)
    return run_ranking


# -------------------------
# Alignment (spline regression)
# -------------------------


def fit_calibration_model(
    ref_df: pd.DataFrame,
    run_series: pd.Series,
    *,
    n_knots: int = 12,
    alpha: float = 1e-3,
    min_overlap: int | None = None,
):
    """Fit a spline model mapping rt_run -> rt_ref on overlapping peptidoforms."""

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    merged = ref_df.merge(
        run_series.rename("rt_run"),
        left_on="peptidoform",
        right_index=True,
        how="inner",
    )
    n_overlap = len(merged)

    required = max(50, n_knots * 5) if min_overlap is None else int(min_overlap)
    if n_overlap < required:
        raise ValueError(
            f"Not enough overlap ({n_overlap}) to fit spline (required={required}, n_knots={n_knots})."
        )

    x = merged[["rt_run"]].to_numpy(dtype=float)
    y = merged["rt_ref"].to_numpy(dtype=float)

    model = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=3, include_bias=False),
        Ridge(alpha=alpha),
    )
    model.fit(x, y)
    y_hat = model.predict(x)
    mae = float(np.mean(np.abs(y - y_hat)))

    return model, merged, y_hat, mae


def calibrate_run(model, run_series: pd.Series) -> pd.Series:
    """Apply a fitted model to calibrate a run's RTs onto the reference scale."""

    run_all = run_series.dropna().astype(float)
    x_all = run_all.to_numpy(dtype=float).reshape(-1, 1)
    return pd.Series(model.predict(x_all), index=run_all.index, name="rt_cal")


def update_reference(ref_df: pd.DataFrame, rt_cal: pd.Series) -> pd.DataFrame:
    """Weighted reference update with new calibrated RT series (weight=1 for the new run)."""

    out = ref_df.set_index("peptidoform").copy()
    out = out.join(rt_cal, how="outer")
    out["weight"] = out["weight"].fillna(0.0)

    has_ref = out["rt_ref"].notna()
    has_new = out["rt_cal"].notna()
    both = has_ref & has_new
    only_new = (~has_ref) & has_new

    out.loc[both, "rt_ref"] = (
        out.loc[both, "rt_ref"] * out.loc[both, "weight"] + out.loc[both, "rt_cal"]
    ) / (out.loc[both, "weight"] + 1.0)
    out.loc[both, "weight"] = out.loc[both, "weight"] + 1.0

    out.loc[only_new, "rt_ref"] = out.loc[only_new, "rt_cal"]
    out.loc[only_new, "weight"] = 1.0

    return out.drop(columns=["rt_cal"]).reset_index()


def maybe_plot_alignment(
    x_run: np.ndarray,
    y_ref: np.ndarray,
    y_hat: np.ndarray,
    *,
    run_name: str,
    rank: int,
    max_points: int = 2000,
) -> None:
    """Diagnostic plots for one alignment (subsamples if large)."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    x_run = np.asarray(x_run, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    n = x_run.shape[0]
    if n == 0:
        return

    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        x_run_s = x_run[idx]
        y_ref_s = y_ref[idx]
        y_hat_s = y_hat[idx]
    else:
        x_run_s, y_ref_s, y_hat_s = x_run, y_ref, y_hat

    order = np.argsort(x_run_s)
    x_sorted = x_run_s[order]
    yhat_sorted = y_hat_s[order]

    resid = y_ref_s - y_hat_s

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Alignment diagnostics (rank {rank}): {run_name}", y=1.05)

    axes[0].scatter(x_run_s, y_ref_s, s=6, alpha=0.35, label="overlap points")
    axes[0].plot(x_sorted, yhat_sorted, color="black", linewidth=2, label="spline fit")
    axes[0].set_xlabel("Run RT")
    axes[0].set_ylabel("Reference RT")
    axes[0].set_title("Overlap + fitted mapping")
    axes[0].legend(loc="best", frameon=False)

    axes[1].scatter(x_run_s, resid, s=6, alpha=0.35)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Run RT")
    axes[1].set_ylabel("Ref - Pred")
    axes[1].set_title("Residuals vs RT")

    axes[2].hist(resid, bins=40, color="#4C72B0", alpha=0.9)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual distribution")

    plt.tight_layout()
    plt.show()


@dataclass
class AlignmentResult:
    run_ranking: pd.DataFrame
    ref: pd.DataFrame
    diag_df: pd.DataFrame
    calibrated_matrix: pd.DataFrame


@dataclass
class AlignmentInputs:
    """Convenience container for intermediate inputs/outputs.

    This is mainly useful when calling the library-style API.
    """

    long_df: pd.DataFrame
    rt_matrix: pd.DataFrame


@dataclass
class AlignmentOutputs:
    """Full outputs for programmatic use (includes intermediates)."""

    inputs: AlignmentInputs
    result: AlignmentResult


def align_from_psm_list(
    psm_list,
    *,
    n_knots: int = 12,
    alpha: float = 1e-3,
    min_overlap: int | None = None,
    plot_diagnostics: bool = False,
    plot_max_runs: int | None = 5,
    make_sd_plot: bool = False,
) -> AlignmentOutputs:
    """Run the full alignment pipeline starting from an in-memory PSMList.

    Parameters
    ----------
    psm_list:
        A psm_utils PSMList (or any iterable of PSM-like objects) with run id,
        peptidoform/peptide, and retention time fields.
    n_knots, alpha, min_overlap:
        Spline calibration model hyperparameters.
    plot_diagnostics, plot_max_runs:
        If enabled, show per-run alignment diagnostic plots (requires matplotlib).
    make_sd_plot:
        If enabled, show the before/after SD plot after alignment.

    Returns
    -------
    AlignmentOutputs
        Includes the long table and raw RT matrix (inputs) and the alignment
        outputs (run ranking, reference, diagnostics, calibrated matrix).
    """

    long_df = psm_list_to_long_df(psm_list)
    rt_matrix = build_rt_matrix(long_df)
    run_ranking = rank_runs_by_overlap(rt_matrix)

    if min_overlap == 0:
        min_overlap = None

    result = align_runs(
        rt_matrix,
        run_ranking,
        n_knots=int(n_knots),
        alpha=float(alpha),
        min_overlap=min_overlap,
        plot_diagnostics=bool(plot_diagnostics),
        plot_max_runs=plot_max_runs,
    )

    if make_sd_plot:
        plot_sd_before_after(rt_matrix, result.calibrated_matrix)

    return AlignmentOutputs(
        inputs=AlignmentInputs(long_df=long_df, rt_matrix=rt_matrix),
        result=result,
    )


def align_runs(
    rt_matrix: pd.DataFrame,
    run_ranking: pd.DataFrame,
    *,
    n_knots: int = 12,
    alpha: float = 1e-3,
    min_overlap: int | None = None,
    plot_diagnostics: bool = False,
    plot_max_runs: int | None = 5,
) -> AlignmentResult:
    """Iteratively align runs (rank order) onto a growing weighted reference."""

    ranked_runs = run_ranking["run"].tolist()
    if len(ranked_runs) < 2:
        raise ValueError("Need at least 2 runs to start alignment.")

    def get_run_series(run_name: str) -> pd.Series:
        return rt_matrix[run_name].dropna().astype(float)

    run1 = ranked_runs[0]
    s1 = get_run_series(run1)
    ref = pd.DataFrame({"peptidoform": s1.index, "rt_ref": s1.values, "weight": 1.0})

    calibrated_rts: dict[str, pd.Series] = {str(run1): s1.rename("rt_cal")}

    diags: list[dict[str, Any]] = []
    for i, run_name in enumerate(ranked_runs[1:], start=2):
        run_series = get_run_series(run_name)
        model, merged, y_hat, mae = fit_calibration_model(
            ref,
            run_series,
            n_knots=n_knots,
            alpha=alpha,
            min_overlap=min_overlap,
        )
        y_hat = np.asarray(y_hat)

        if plot_diagnostics and (plot_max_runs is None or i <= plot_max_runs):
            maybe_plot_alignment(
                merged["rt_run"].to_numpy(),
                merged["rt_ref"].to_numpy(),
                y_hat,
                run_name=str(run_name),
                rank=i,
            )

        rt_cal = calibrate_run(model, run_series)
        calibrated_rts[str(run_name)] = rt_cal
        ref = update_reference(ref, rt_cal)

        diags.append(
            {
                "rank": i,
                "run": str(run_name),
                "n_overlap": int(len(merged)),
                "mae_overlap": float(mae),
                "n_run": int(run_series.shape[0]),
                "ref_size": int(ref.shape[0]),
            }
        )

    diag_df = pd.DataFrame(diags)
    calibrated_matrix = pd.concat(calibrated_rts, axis=1)
    calibrated_matrix.columns = calibrated_matrix.columns.astype(str)

    return AlignmentResult(
        run_ranking=run_ranking,
        ref=ref,
        diag_df=diag_df,
        calibrated_matrix=calibrated_matrix,
    )


# -------------------------
# SD plots and filtering
# -------------------------


def per_peptidoform_std(rt_df: pd.DataFrame, min_runs: int = 2) -> pd.Series:
    n = rt_df.notna().sum(axis=1)
    std = rt_df.std(axis=1, skipna=True)
    std[n < min_runs] = np.nan
    return std


def plot_sd_before_after(
    rt_matrix: pd.DataFrame, calibrated_matrix: pd.DataFrame
) -> None:
    """Plot per-peptidoform SD across runs before vs after calibration."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping SD plots")
        return

    raw_sd = per_peptidoform_std(rt_matrix)
    cal_sd = per_peptidoform_std(calibrated_matrix)
    cmp = pd.DataFrame({"raw_sd": raw_sd, "cal_sd": cal_sd}).dropna()

    print(f"Peptidoforms with SD in both (>=2 runs each): {cmp.shape[0]:,}")
    print("Raw SD median:", float(cmp["raw_sd"].median()))
    print("Cal SD median:", float(cmp["cal_sd"].median()))
    print(
        "Median fold-change (cal/raw):", float((cmp["cal_sd"] / cmp["raw_sd"]).median())
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Per-peptidoform RT variability across runs (SD)", y=1.05)

    axes[0].hist(cmp["raw_sd"], bins=60, alpha=0.75, label="raw", color="#DD8452")
    axes[0].hist(
        cmp["cal_sd"], bins=60, alpha=0.75, label="calibrated", color="#4C72B0"
    )
    axes[0].set_xlabel("SD (RT units)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[0].legend(frameon=False)

    axes[1].set_title("ECDF")
    for col, color in [("raw_sd", "#DD8452"), ("cal_sd", "#4C72B0")]:
        x = np.sort(cmp[col].to_numpy())
        y = np.linspace(0, 1, x.size, endpoint=True)
        axes[1].plot(x, y, label=col.replace("_", " "), color=color)
    axes[1].set_xlabel("SD (RT units)")
    axes[1].set_ylabel("Fraction ≤ x")
    axes[1].legend(frameon=False)

    n = cmp.shape[0]
    cmp_plot = cmp.sample(20000, random_state=0) if n > 20000 else cmp
    axes[2].scatter(cmp_plot["raw_sd"], cmp_plot["cal_sd"], s=6, alpha=0.25)
    lim = float(np.nanmax([cmp_plot["raw_sd"].max(), cmp_plot["cal_sd"].max()]))
    axes[2].plot([0, lim], [0, lim], color="black", linewidth=1)
    axes[2].set_xlabel("Raw SD")
    axes[2].set_ylabel("Calibrated SD")
    axes[2].set_title("Paired: raw vs calibrated")

    plt.tight_layout()
    plt.show()


def filter_by_occurrence_and_sd(
    rt_df: pd.DataFrame,
    min_occurrence_pct: float = 50.0,
    max_sd: float | None = None,
    *,
    gradient_length: float | None = None,
    max_sd_rel: float | None = None,
    min_runs: int = 2,
    return_stats: bool = False,
    keep: bool = True,
    sd_ddof: int = 1,
    verbose: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a peptidoform×run RT matrix by occurrence (%) and SD thresholds.

    Supports two SD thresholds:
    - Absolute: sd <= max_sd
    - Relative-to-gradient: (sd / gradient_length) <= max_sd_rel

    If both SD gates are provided, both must pass.
    """

    if rt_df.shape[1] == 0:
        raise ValueError("rt_df has no run columns.")
    if not (0.0 <= float(min_occurrence_pct) <= 100.0):
        raise ValueError("min_occurrence_pct must be between 0 and 100.")
    if max_sd_rel is not None:
        if gradient_length is None:
            raise ValueError("gradient_length must be provided when max_sd_rel is set.")
        if float(gradient_length) <= 0:
            raise ValueError("gradient_length must be > 0.")
        if float(max_sd_rel) < 0:
            raise ValueError("max_sd_rel must be >= 0.")
    if max_sd is not None and float(max_sd) < 0:
        raise ValueError("max_sd must be >= 0.")
    if int(min_runs) < 0:
        raise ValueError("min_runs must be >= 0.")

    n_runs = int(rt_df.shape[1])
    n_obs = rt_df.notna().sum(axis=1).astype(int)
    occurrence_pct = (n_obs / n_runs) * 100.0

    sd = rt_df.std(axis=1, skipna=True, ddof=sd_ddof)
    sd = sd.where(n_obs >= int(min_runs), np.nan)

    sd_rel = None
    if gradient_length is not None:
        sd_rel = sd / float(gradient_length)

    pass_occ = occurrence_pct >= float(min_occurrence_pct)

    pass_sd_abs = pd.Series(True, index=rt_df.index)
    if max_sd is not None:
        pass_sd_abs = sd.notna() & (sd <= float(max_sd))

    pass_sd_rel = pd.Series(True, index=rt_df.index)
    if max_sd_rel is not None:
        if sd_rel is None:
            raise ValueError("gradient_length must be provided when max_sd_rel is set.")
        pass_sd_rel = sd_rel.notna() & (sd_rel <= float(max_sd_rel))

    pass_sd = pass_sd_abs & pass_sd_rel

    pass_all = pass_occ & pass_sd
    if not keep:
        pass_all = ~pass_all

    filtered = rt_df.loc[pass_all].copy()

    stats_dict: dict[str, Any] = {
        "occurrence_pct": occurrence_pct,
        "n_obs": n_obs,
        "sd": sd,
    }
    if sd_rel is not None:
        stats_dict["sd_rel"] = sd_rel
    stats = pd.DataFrame(stats_dict, index=rt_df.index)

    if verbose:
        n_total = int(rt_df.shape[0])
        n_pass_occ = int(pass_occ.sum())
        n_pass_all = int((pass_occ & pass_sd).sum())
        print(f"Total peptidoforms: {n_total:,}")
        print(
            f"Pass occurrence >= {min_occurrence_pct:g}%: {n_pass_occ:,} ({n_pass_occ / max(n_total,1):.1%})"
        )
        if max_sd is not None:
            n_pass_abs = int(pass_sd_abs.sum())
            print(
                f"Pass SD <= {max_sd:g} (min_runs={min_runs}): {n_pass_abs:,} ({n_pass_abs / max(n_total,1):.1%})"
            )
        if max_sd_rel is not None:
            n_pass_rel = int(pass_sd_rel.sum())
            gl = float(gradient_length) if gradient_length is not None else float("nan")
            print(
                f"Pass SD/gradient <= {max_sd_rel:g} (gradient_length={gl:g}, min_runs={min_runs}): "
                f"{n_pass_rel:,} ({n_pass_rel / max(n_total,1):.1%})"
            )
        print(f"Pass all gates: {n_pass_all:,} ({n_pass_all / max(n_total,1):.1%})")

    return (filtered, stats) if return_stats else filtered


# -------------------------
# CLI glue
# -------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "results.plasma_proteomics.tsv"),
        help="Path to Sage results TSV (default: results.plasma_proteomics.tsv next to this script).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="Output directory (default: ./out).",
    )

    # Alignment model params
    p.add_argument(
        "--n-knots", type=int, default=12, help="Spline knots for calibration."
    )
    p.add_argument(
        "--alpha", type=float, default=1e-3, help="Ridge alpha for spline model."
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=0,
        help="Minimum overlap required to fit model (0 = auto heuristic).",
    )

    # Plotting
    p.add_argument(
        "--plots", action="store_true", help="Enable plots (requires matplotlib)."
    )
    p.add_argument(
        "--plot-max-runs",
        type=int,
        default=5,
        help="Max number of per-alignment diagnostic plots to show (ignored if --plots is off).",
    )

    # Filtering
    p.add_argument(
        "--filter",
        action="store_true",
        help="Also produce a filtered calibrated matrix based on occurrence and SD gates.",
    )
    p.add_argument(
        "--min-occurrence-pct",
        type=float,
        default=0.0,
        help="Filter: minimum occurrence percentage across runs.",
    )
    p.add_argument(
        "--max-sd",
        type=float,
        default=-1.0,
        help="Filter: maximum absolute SD across runs (RT units). Use <0 to disable.",
    )
    p.add_argument(
        "--gradient-length",
        type=float,
        default=-1.0,
        help="For relative SD filtering: gradient length in RT units. Use <0 to auto-estimate from calibrated RT range.",
    )
    p.add_argument(
        "--max-sd-rel",
        type=float,
        default=-1.0,
        help="Filter: maximum SD/gradient_length (fraction). Use <0 to disable.",
    )
    p.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Filter: min runs observed to evaluate SD.",
    )

    p.add_argument(
        "--no-write-long",
        action="store_true",
        help="Skip writing long_df parquet (saves disk).",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    min_overlap = None if int(args.min_overlap) <= 0 else int(args.min_overlap)

    print(f"Reading PSMs from: {in_path}")
    psm_list = load_psm_list(in_path)
    print(f"Loaded PSMs: {len(psm_list):,}")

    long_df = psm_list_to_long_df(psm_list)
    rt_matrix = build_rt_matrix(long_df)
    print(f"RT matrix shape (peptidoforms x runs): {rt_matrix.shape}")

    run_ranking = rank_runs_by_overlap(rt_matrix)
    print("\nTop runs by overlap:")
    print(run_ranking.head(10).to_string(index=False))

    if args.plots:
        try:
            import matplotlib.pyplot as plt

            top = run_ranking.head(15).sort_values("overlap_score", ascending=True)
            plt.figure(figsize=(10, 6))
            plt.barh(top["run"].astype(str), top["overlap_score"].astype(float))
            plt.title("Top runs by summed pairwise overlap")
            plt.xlabel("Overlap score (sum of shared peptidoforms vs all other runs)")
            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            print("matplotlib not installed; skipping overlap plot")

    result = align_runs(
        rt_matrix,
        run_ranking,
        n_knots=int(args.n_knots),
        alpha=float(args.alpha),
        min_overlap=min_overlap,
        plot_diagnostics=bool(args.plots),
        plot_max_runs=None if args.plot_max_runs is None else int(args.plot_max_runs),
    )

    print(f"\nCalibrated matrix shape: {result.calibrated_matrix.shape}")

    # Persist outputs
    run_ranking_path = outdir / "run_ranking.tsv"
    diag_path = outdir / "alignment_diagnostics.tsv"
    ref_path = outdir / "reference.parquet"
    cal_path = outdir / "calibrated_rts.parquet"
    raw_path = outdir / "raw_rts.parquet"

    result.run_ranking.to_csv(run_ranking_path, sep="\t", index=False)
    result.diag_df.to_csv(diag_path, sep="\t", index=False)
    result.ref.to_parquet(ref_path)
    result.calibrated_matrix.to_parquet(cal_path)
    rt_matrix.to_parquet(raw_path)

    if not args.no_write_long:
        long_df.to_parquet(outdir / "psms_long.parquet", index=False)

    print("\nWrote:")
    print(f"- {run_ranking_path}")
    print(f"- {diag_path}")
    print(f"- {ref_path}")
    print(f"- {cal_path}")
    print(f"- {raw_path}")
    if not args.no_write_long:
        print(f"- {outdir / 'psms_long.parquet'}")

    if args.plots:
        plot_sd_before_after(rt_matrix, result.calibrated_matrix)

    if args.filter:
        max_sd = None if float(args.max_sd) < 0 else float(args.max_sd)
        max_sd_rel = None if float(args.max_sd_rel) < 0 else float(args.max_sd_rel)

        if max_sd_rel is None:
            gradient_length: float | None = None
        else:
            if float(args.gradient_length) < 0:
                # Auto-estimate from calibrated RT range if user didn't provide one
                s = result.calibrated_matrix.stack(dropna=True)
                if len(s):
                    arr = s.to_numpy(dtype=float)
                    gradient_length = float(np.nanmax(arr) - np.nanmin(arr))
                else:
                    gradient_length = 0.0
            else:
                gradient_length = float(args.gradient_length)

        filtered, stats = filter_by_occurrence_and_sd(
            result.calibrated_matrix,
            min_occurrence_pct=float(args.min_occurrence_pct),
            max_sd=max_sd,
            gradient_length=gradient_length,
            max_sd_rel=max_sd_rel,
            min_runs=int(args.min_runs),
            return_stats=True,
            verbose=True,
        )

        filtered = cast(pd.DataFrame, filtered)
        stats = cast(pd.DataFrame, stats)

        filtered_path = outdir / "calibrated_rts.filtered.parquet"
        stats_path = outdir / "calibrated_rts.filter_stats.tsv"
        filtered.to_parquet(filtered_path)
        stats.to_csv(stats_path, sep="\t")

        print("\nFiltered outputs:")
        print(f"- {filtered_path}")
        print(f"- {stats_path}")

    return 0


if __name__ == "__main__":
    psm_list = load_psm_list("results.plasma_proteomics.tsv")

    out = align_from_psm_list(
        psm_list,
        n_knots=12,
        alpha=1e-3,
        min_overlap=None,
        plot_diagnostics=False,
        make_sd_plot=False,
    )

    ref = out.result.ref

    filtered, stats = filter_by_occurrence_and_sd(
        out.result.calibrated_matrix,
        min_occurrence_pct=10,
        gradient_length=ref["rt_ref"].max() - ref["rt_ref"].min(),
        max_sd_rel=0.03,
        min_runs=2,
        return_stats=True,
        verbose=True,
    )

    print(filtered.median(axis=1))
