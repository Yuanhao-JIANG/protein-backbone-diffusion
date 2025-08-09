# benchmark.py
import csv, math, re
from pathlib import Path
from typing import List, Optional, Any
import numpy as np
import torch
from tqdm import tqdm
from sampler import batch_sampler
from tmtools import tm_align


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
    Compute a TM-score (and RMSD) via tmtools.tm_align.

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
        If set, directory where generated PDBs will be saved.

    Notes
    -----
    - CSV columns: [name, L, rmsd, tm, error]
    - TM-score is computed via `run_tmalign`, which uses `tmtools.tm_align`; RMSD is Kabsch-aligned.
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
            gens = batch_sampler(model=model, sde=sde, lengths=lens, n_corr_steps=2, device=device)  # List[np.ndarray], each [L_i,3]
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
