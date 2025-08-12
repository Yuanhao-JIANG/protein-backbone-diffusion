# benchmark.py
import csv, math, re
from pathlib import Path
from typing import List, Optional, Iterable, Tuple, Dict, Sequence, Any
import numpy as np
import torch
from tqdm import tqdm
from sampler import batch_sampler
from tmtools import tm_align
import pandas as pd
import matplotlib.pyplot as plt


_TM_RE = re.compile(r"TM-score\s*=\s*([0-9.]+)")

def list_npy(npy_dir: str | Path, limit: Optional[int] = None) -> list[Path]:
    files = sorted(Path(npy_dir).glob("*.npy"))
    return files[:limit] if limit else files

def read_ca_npy(p: str | Path) -> np.ndarray:
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected [L,3], got {arr.shape} in {p}")
    return arr.astype(float)

def write_ca_pdb(coords: np.ndarray, out_path: str | Path):
    with open(out_path, "w") as f:
        for i, (x,y,z) in enumerate(coords, 1):
            f.write(f"ATOM  {i:5d}  CA  GLY A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
        f.write("END\n")

def native_scale_from_coords(nat: np.ndarray) -> float:
    # normalization used in training
    mu = nat.mean(axis=0, keepdims=True)
    return np.linalg.norm(nat - mu, axis=1).mean()

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    C = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1,1,d]) @ Wt
    return Pc @ R + Q.mean(0, keepdims=True)

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pa = kabsch_align(P, Q)
    diff = Pa - Q
    return float(np.sqrt((diff*diff).sum() / P.shape[0]))

def run_tmalign(gen: np.ndarray, nat: np.ndarray, normalize: str = "max"):
    """
    Compute TM score (and RMSD) via tmtools.tm_align.

    Args:
        gen: [L1, 3] generated Cα coords
        nat: [L2, 3] native Cα coords
        normalize: which symmetric TM to report:
            - "max": max(TM_norm_chain1, TM_norm_chain2) (common default)
            - "avg": 0.5*(TM_norm_chain1 + TM_norm_chain2)
            - "chain1": TM normalized by len(gen)
            - "chain2": TM normalized by len(nat) (original TM-align default)

    Returns:
        tm_score (float), rmsd (float), result (tmtools result object)
    """
    # Poly-A sequences; residue types are irrelevant for TM-score
    seq1 = "A" * len(gen)
    seq2 = "A" * len(nat)

    res = tm_align(gen.astype(np.float64), nat.astype(np.float64), seq1, seq2)

    if normalize == "max":
        tm = float(max(res.tm_norm_chain1, res.tm_norm_chain2))
    elif normalize == "avg":
        tm = float(0.5 * (res.tm_norm_chain1 + res.tm_norm_chain2))
    elif normalize == "chain1":
        tm = float(res.tm_norm_chain1)
    elif normalize == "chain2":
        tm = float(res.tm_norm_chain2)
    else:
        raise ValueError(f"Unknown normalize='{normalize}'")

    return tm, float(res.rmsd), res

def chunk_by_total_nodes(lengths: List[int], max_nodes: int) -> List[List[int]]:
    batch, batches, total = [], [], 0
    for L in lengths:
        if total + L > max_nodes and batch:
            batches.append(batch); batch = []; total = 0
        batch.append(L); total += L
    if batch: batches.append(batch)
    return batches

def run_benchmark(
    model: Any,
    sde: Any,
    npy_dir: str | Path,
    out_csv: str | Path,
    *,
    device: str | torch.device = "cuda",
    limit: Optional[int] = None,
    max_nodes_per_batch: int = 8000,   # tune to GPU memory
    save_gen_pdb_dir: Optional[str | Path] = None,
):
    """
    Runs a batched benchmark of a generative protein model against native structures.

    This function:
      1. Loads native CA-coordinate .npy files from a directory.
      2. Groups them into batches constrained by `max_nodes_per_batch` for GPU memory efficiency.
      3. Generates corresponding structures using the provided `model` and `sde`.
      4. Aligns generated structures to natives and computes RMSD and TM-score.
      5. Optionally saves generated structures as PDB files.
      6. Logs results to a CSV file.

    Parameters
    ----------
    model : Any
        The trained score-based model to benchmark.
    sde : Any
        The SDE (stochastic differential equation) object used for sampling.
    npy_dir : str or Path
        Path to the directory containing native CA-coordinate .npy files.
    out_csv : str or Path
        Path to the CSV file where benchmark results will be written/appended.
    device : str or torch.device, optional
        Device for model sampling (default: "cuda").
    limit : int, optional
        If set, only the first `limit` .npy files are used.
    max_nodes_per_batch : int, optional
        Maximum total number of residues per batch to fit into GPU memory.
    save_gen_pdb_dir : str or Path, optional
        If set, the directory where generated PDBs will be saved.

    Notes
    -----
    - CSV columns: [name, L, rmsd, tm, error]
    - TM score is computed via `run_tmalign`, which uses `tmtools.tm_align`; RMSD is Kabsch-aligned.
    """
    files = list_npy(npy_dir, limit)
    if not files:
        raise ValueError(f"No .npy files in {npy_dir}")

    # read all natives first
    natives = []
    meta = []
    for p in files:
        arr = read_ca_npy(p)
        natives.append(arr)
        meta.append((p.stem, arr.shape[0]))

    # sort by length for better batching
    idx = sorted(range(len(meta)), key=lambda i: meta[i][1])
    natives, files, meta = [natives[i] for i in idx], [files[i] for i in idx], [meta[i]  for i in idx]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fcsv = open(out_csv, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(["name", "L", "rmsd", "tm (max)", "error"])

    pdb_dir = None
    if save_gen_pdb_dir:
        pdb_dir = Path(save_gen_pdb_dir); pdb_dir.mkdir(parents=True, exist_ok=True)

    # batching
    lengths = [L for _, L in meta]
    batches = chunk_by_total_nodes(lengths, max_nodes_per_batch)

    pointer = 0
    for lens in tqdm(batches, desc="Benchmark (batched)"):
        # sample a batch of structures with given lengths
        try:
            gens = batch_sampler(model=model, sde=sde, lengths=lens, snr=0.16, n_corr_steps=1, device=device)  # List[np.ndarray], each [L_i,3]
            assert len(gens) == len(lens)
        except Exception as e:
            # write errors for this batch
            for j in range(len(lens)):
                name, L = meta[pointer + j]
                writer.writerow([name, L, math.nan, math.nan, "", f"sample_error:{e}"])
            pointer += len(lens)
            continue

        # evaluate
        for j, gen in enumerate(gens):
            name, L = meta[pointer + j]
            nat = natives[pointer + j]
            rmsd = tm = math.nan
            err = ""
            try:
                if L >= 3:
                    gen_scaled = gen * native_scale_from_coords(nat)
                    rmsd = kabsch_rmsd(gen_scaled, nat)
                    tm, rmsd_tmalign, _ = run_tmalign(gen_scaled, nat, normalize="max")
                    if pdb_dir:
                        gen_pdb_path = str(pdb_dir / f"{name}_gen.pdb")
                        write_ca_pdb(gen_scaled, gen_pdb_path)
                else:
                    err = "TooFewResidues"
            except Exception as e:
                err = str(e)
            writer.writerow([name, L, rmsd, tm, err])

        pointer += len(lens)

    fcsv.close()


def _load_and_clean(path: str | Path,
                    model_name: str,
                    sep: str = ",") -> pd.DataFrame:
    """
    Load a semicolon-separated benchmark CSV and normalize columns.

    Expected columns (case-insensitive, flexible):
      - name
      - L
      - rmsd
      - tm (or 'tm (max)' will be normalized to 'tm')
      - error (optional). If missing, success=True for all rows.

    Rows are annotated with:
      - 'success': boolean success flag inferred from the 'error' column
      - 'Model': the provided model_name
    """
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize TM column
    if "tm (max)" in df.columns and "tm" not in df.columns:
        df = df.rename(columns={"tm (max)": "tm"})

    # Coerce numeric
    for col in ("l", "rmsd", "tm"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build success flag
    if "error" in df.columns:
        err = df["error"]
        success = (err.fillna(0) == 0)
    else:
        success = pd.Series(True, index=df.index)

    df["success"] = success
    df["Model"] = model_name
    return df

def _ci95_mean(x: pd.Series) -> Tuple[float, float]:
    """Approximate 95% CI for the mean (normal approx; fine for large n)."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * s / np.sqrt(n) if n > 0 else np.nan
    return m - half, m + half

def load_all(files: Dict[str, str | Path], sep: str = ",") -> pd.DataFrame:
    return pd.concat([_load_and_clean(path, model, sep=sep) for model, path in files.items()], ignore_index=True)

def summarise_benchmarks(
        df_all: pd.DataFrame,
        *,
        save_csv: Optional[str | Path] = None,
        rmsd_thresholds: Iterable[float] = (2.0, 4.0, 6.0),
        tm_thresholds: Iterable[float] = (0.5, 0.3),
) -> pd.DataFrame:
    """
    Create a summary table for multiple benchmark CSVs.

    Parameters
    ----------
    df_all: pandas.DataFrame
        Data frame.
    save_csv : str | Path | None
        If provided, save the summary table to this path.
    rmsd_thresholds : iterable of float
        RMSD thresholds to report as percentages (lower is better).
    tm_thresholds : iterable of float
        TM-score thresholds to report as percentages (higher is better).

    Returns
    -------
    pd.DataFrame
        One row per model with sample counts, error rate, central tendency,
        dispersion, 95% CIs, and threshold-based percentages.
    """
    rows = []
    for model, g in df_all.groupby("Model", sort=False):
        n_total = len(g)
        n_success = int(g["success"].sum())
        error_rate = (n_total - n_success) / n_total if n_total else np.nan

        g_ok = g[g["success"]].copy()
        # Core stats
        rmsd_mean = g_ok["rmsd"].mean()
        rmsd_median = g_ok["rmsd"].median()
        rmsd_std = g_ok["rmsd"].std(ddof=1)
        tm_mean = g_ok["tm"].mean()
        tm_median = g_ok["tm"].median()
        tm_std = g_ok["tm"].std(ddof=1)

        rmsd_lo, rmsd_hi = _ci95_mean(g_ok["rmsd"])
        tm_lo, tm_hi = _ci95_mean(g_ok["tm"])

        # Thresholds
        thr_cols = {}
        if len(g_ok):
            for t in rmsd_thresholds:
                thr_cols[f"% RMSD < {t}A"] = (g_ok["rmsd"] < t).mean()
            for t in tm_thresholds:
                thr_cols[f"% TM >= {t}"] = (g_ok["tm"] >= t).mean()
        else:
            for t in rmsd_thresholds:
                thr_cols[f"% RMSD < {t}A"] = np.nan
            for t in tm_thresholds:
                thr_cols[f"% TM >= {t}"] = np.nan

        rows.append({
            "Model": model,
            "N (total)": n_total,
            "N (success)": n_success,
            "Error rate": error_rate,

            "RMSD mean (lower is better)": rmsd_mean,
            "RMSD median (lower is better)": rmsd_median,
            "RMSD std": rmsd_std,
            "RMSD 95% CI low": rmsd_lo,
            "RMSD 95% CI high": rmsd_hi,

            "TM mean (larger is better)": tm_mean,
            "TM median (larger is better)": tm_median,
            "TM std": tm_std,
            "TM 95% CI low": tm_lo,
            "TM 95% CI high": tm_hi,
            **thr_cols
        })

    summary = pd.DataFrame(rows)

    # Order columns: fixed part first, then dynamic thresholds (stable alphabetical)
    fixed_order = [
        "Model", "N (total)", "N (success)", "Error rate",
        "RMSD mean (lower is better)", "RMSD median (lower is better)", "RMSD std", "RMSD 95% CI low", "RMSD 95% CI high",
        "TM mean (larger is better)", "TM median (larger is better)", "TM std", "TM 95% CI low", "TM 95% CI high",
    ]
    dynamic_cols = sorted([c for c in summary.columns if c not in fixed_order], key=lambda x: (x.startswith("% TM") is False, x))
    ordered = fixed_order + dynamic_cols
    summary = summary[ordered]

    # Save (optional)
    if save_csv is not None:
        out = Path(save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)

    return summary

# ---------- Shared helpers ----------
def _ref_lines_for_metric(metric: str) -> Sequence[float]:
    """Default reference lines for RMSD/TM based on common interpretation."""
    if metric.lower() == "rmsd":
        return [2.0, 4.0, 6.0]
    if metric.lower() in ("tm", "tm (max)"):
        return [0.3, 0.5]
    return []

def _metric_col(df: pd.DataFrame, metric: str) -> str:
    """Resolve flexible TM column naming."""
    m = metric.strip().lower()
    if m in df.columns:
        return m
    if m == "tm" and "tm (max)" in df.columns:
        return "tm (max)"
    raise KeyError(f"Metric '{metric}' not found in columns {list(df.columns)}")

def _success_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.get("success", True)].copy()

# ---------- 1) Distribution plots ----------
def plot_metric_distribution(
    df_all: pd.DataFrame,
    metric: str = "rmsd",
    *,
    bins: int = 80,
    xlim: Optional[tuple[float, float]] = None,
    add_ref_lines: bool = True,
    density: bool = True,
    alpha: float = 0.35,
    figsize: tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
):
    """
    Overlaid histograms of a metric (RMSD or TM) by model.

    Parameters
    ----------
    df_all: DataFrame with columns ['Model', metric, 'success']
    metric: 'rmsd' or 'tm' (flexible: 'tm (max)' also handled)
    bins: number of histogram bins
    xlim: optional x-axis limits
    add_ref_lines: add standard threshold lines (RMSD: 2/4/6 A, TM: 0.3/0.5)
    density: normalize histograms to probability density
    alpha: transparency for overlays
    figsize: figure size
    save_path: optional path to save PNG
    """
    metric_col = _metric_col(df_all, metric)
    df = _success_only(df_all)
    plt.figure(figsize=figsize)
    models = list(df["Model"].unique())

    # Determine a common x range if not provided
    data_all = df[metric_col].dropna().values
    if xlim is None and len(data_all):
        lo, hi = np.nanpercentile(data_all, [0.5, 99.5])
        pad = (hi - lo) * 0.05
        xlim = (max(lo - pad, np.nanmin(data_all)), hi + pad)

    for model in models:
        x = df.loc[df["Model"] == model, metric_col].dropna().values
        if len(x) == 0:
            continue
        plt.hist(x, bins=bins, density=density, alpha=alpha, label=str(model))

    if add_ref_lines:
        for v in _ref_lines_for_metric(metric_col):
            plt.axvline(v, linestyle="--", linewidth=1)

    plt.xlabel(metric_col.upper())
    plt.ylabel("Density" if density else "Count")
    plt.title(f"{metric_col.upper()} distribution by model")
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ---------- 2) Box / Violin plots ----------
def plot_metric_box_violin(
    df_all: pd.DataFrame,
    metric: str = "rmsd",
    *,
    kind: str = "box",  # 'box' or 'violin'
    showfliers: bool = False,
    add_ref_lines: bool = True,
    figsize: tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
):
    """
    Boxplot or violin plot of a metric grouped by model.

    Parameters
    ----------
    df_all: DataFrame with columns ['Model', metric, 'success']
    metric: 'rmsd' or 'tm'
    kind: 'box' or 'violin'
    showfliers: whether to show outliers in boxplot
    add_ref_lines: add standard RMSD/TM reference lines
    figsize: figure size
    save_path: optional path to save PNG
    """
    metric_col = _metric_col(df_all, metric)
    df = _success_only(df_all)
    models = list(df["Model"].unique())
    data_by_model = [df.loc[df["Model"] == m, metric_col].dropna().values for m in models]

    plt.figure(figsize=figsize)
    if kind == "box":
        bp = plt.boxplot(
            data_by_model,
            labels=models,
            showfliers=showfliers,
            patch_artist=False,
        )
    elif kind == "violin":
        vp = plt.violinplot(
            data_by_model,
            showmeans=False,
            showextrema=True,
            showmedians=True,
        )
        # X tick labels under violins
        plt.xticks(range(1, len(models) + 1), models, rotation=0)
    else:
        raise ValueError("kind must be 'box' or 'violin'")

    if add_ref_lines:
        for v in _ref_lines_for_metric(metric_col):
            plt.axhline(v, linestyle="--", linewidth=1)

    plt.ylabel(metric_col.upper())
    plt.title(f"{metric_col.upper()} by model")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ---------- 3) Scatter vs length (L) ----------
def plot_metric_vs_length(
    df_all: pd.DataFrame,
    metric: str = "rmsd",
    *,
    alpha: float = 0.35,
    s: float = 9.0,
    jitter: float = 0.0,
    add_linear_trend: bool = True,
    add_ref_lines: bool = True,
    figsize: tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
):
    """
    Scatter of metric vs. sequence length L for each model, with optional linear trend.

    Parameters
    ----------
    df_all: DataFrame with columns ['Model', 'L', metric, 'success']
    metric: 'rmsd' or 'tm'
    alpha: point transparency
    s: marker size
    jitter: optional horizontal jitter (in residues) to reduce overplotting
    add_linear_trend: fit and plot y = a*L + b per model (ordinary least squares)
    add_ref_lines: add reference lines
    figsize: figure size
    save_path: optional path to save PNG
    """
    metric_col = _metric_col(df_all, metric)
    df = _success_only(df_all).dropna(subset=["l", metric_col])

    plt.figure(figsize=figsize)
    models = list(df["Model"].unique())

    for model in models:
        sub = df[df["Model"] == model]
        if jitter > 0:
            x = sub["l"].values + np.random.uniform(-jitter, jitter, size=len(sub))
        else:
            x = sub["l"].values
        y = sub[metric_col].values

        plt.scatter(x, y, s=s, alpha=alpha, label=str(model))

        if add_linear_trend and len(sub) >= 2:
            # Simple OLS fit
            A = np.vstack([sub["l"].values, np.ones(len(sub))]).T
            a, b = np.linalg.lstsq(A, sub[metric_col].values, rcond=None)[0]
            xx = np.linspace(sub["l"].min(), sub["l"].max(), 200)
            yy = a * xx + b
            plt.plot(xx, yy, linewidth=1)

    # Reference lines (horizontal) help interpret quality bands
    if add_ref_lines:
        for v in _ref_lines_for_metric(metric_col):
            plt.axhline(v, linestyle="--", linewidth=1)

    plt.xlabel("L (residues)")
    plt.ylabel(metric_col.upper())
    plt.title(f"{metric_col.upper()} vs length (L)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
