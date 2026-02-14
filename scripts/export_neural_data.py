"""
Export Neural Data for Neural Evolution Explorer
=================================================
Reads BFEStats from .pkl (neural_data_lib) and writes the same directory/JSON
structure as generate_pseudo_data.py so the frontend can display real data.

Output layout (same as generate_pseudo_data.py):
  output_dir/
    experiments.json          # list of experiment meta
    {exp_id}/
      meta.json
      evol_traj.json          # per-gen deepsim/biggan (+ ref_deepsim/ref_biggan when ref data present)
      summary_stats.json
      gen_{NN}/
        deepsim_psth.json     # thread 0 = FC6 / DeepSim
        biggan_psth.json      # thread 1 = BigGAN
        # Optional: deepsim_img.png, biggan_img.png (not generated here)

Usage:
  # From repo root, using default pkl location (matroot):
  python scripts/export_neural_data.py --output ./neural-evolution-explorer/public/data

  # With custom pkl directory:
  python scripts/export_neural_data.py --output ./public/data --pkl-dir /path/to/Mat_Statistics

  # Limit to first N experiments:
  python scripts/export_neural_data.py --output ./public/data --experiments 5

  # Skip individual trial responses in PSTH JSON (saves storage):
  python scripts/export_neural_data.py --output ./public/data --no-trial-rates
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root: scripts/ is on path so neural_data_lib can import neural_data_utils
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from neural_data_lib import (
    load_neural_data,
    extract_all_evol_trajectory_dyna,
    extract_evol_psth_array,
    extract_natref_activation_array,
)


def get_evol_meta_for_export(BFEStats, Expi):
    """Extract evolution display meta (image size, position, thread space names) for one experiment."""
    S = BFEStats[Expi - 1]
    evol = S.get("evol")
    if evol is None:
        return {}
    space_names = evol.get("space_names")
    if space_names is None or len(space_names) < 2:
        return {}
    if isinstance(space_names[0], list):
        space_names = [n[0] for n in space_names]
    space_names = [str(n) for n in space_names]
    imgsize = evol.get("imgsize")
    imgpos = evol.get("imgpos")
    out = {
        "thread0_space": space_names[0],
        "thread1_space": space_names[1],
    }
    if imgsize is not None:
        arr = np.asarray(imgsize).flatten()
        out["image_size_deg"] = [round(float(x), 1) for x in arr]
    if imgpos is not None:
        arr = np.asarray(imgpos).flatten()
        out["image_pos_deg"] = [round(float(x), 1) for x in arr]
    return out

# PSTH time axis: assume 4 ms per bin (match generate_pseudo_data.py)
PSTH_BIN_MS = 1


def load_bfestats(pkl_dir: str | None):
    """Load BFEStats list. If pkl_dir is set, load from that directory; else use load_neural_data()."""
    if pkl_dir is not None:
        pkl_dir = Path(pkl_dir)
        with open(pkl_dir / "Both_BigGAN_FC6_Evol_Stats_expsep.pkl", "rb") as f:
            BFEStats = pickle.load(f)
        return BFEStats
    _, BFEStats = load_neural_data()
    return BFEStats


def exp_id_from_expi(Expi: int, animal: str) -> str:
    """Unique, filesystem-safe experiment id for folder and experiments.json."""
    return f"Exp{Expi:03d}-{animal}"


def build_evol_traj(
    resp_bunch: np.ndarray,
    ref_resp_arr0: list[np.ndarray] | None = None,
    ref_resp_arr1: list[np.ndarray] | None = None,
) -> list[dict]:
    """resp_bunch: (n_blocks, 6) = resp_m_0, resp_m_1, resp_sem_0, resp_sem_1, bsl_m_0, bsl_m_1.
    ref_resp_arr0/1: per-block arrays of ref image response (one value per image); mean±sem added as ref_deepsim, ref_biggan when present."""
    out = []
    n_blocks = resp_bunch.shape[0]
    for g in range(n_blocks):
        m0, m1, s0, s1 = resp_bunch[g, 0], resp_bunch[g, 1], resp_bunch[g, 2], resp_bunch[g, 3]
        entry = {
            "gen": g + 1,
            "deepsim": round(float(m0), 2),
            "deepsim_low": round(float(m0 - s0), 2),
            "deepsim_high": round(float(m0 + s0), 2),
            "biggan": round(float(m1), 2),
            "biggan_low": round(float(m1 - s1), 2),
            "biggan_high": round(float(m1 + s1), 2),
        }
        if ref_resp_arr0 is not None and g < len(ref_resp_arr0) and ref_resp_arr1 is not None and g < len(ref_resp_arr1):
            r0 = np.asarray(ref_resp_arr0[g]).flatten()
            r1 = np.asarray(ref_resp_arr1[g]).flatten()
            n0, n1 = r0.size, r1.size
            if n0 > 0:
                mean0 = float(np.mean(r0))
                sem0 = float(np.std(r0, ddof=1) / np.sqrt(n0)) if n0 > 1 else 0.0
                entry["ref_deepsim"] = round(mean0, 2)
                entry["ref_deepsim_low"] = round(mean0 - sem0, 2)
                entry["ref_deepsim_high"] = round(mean0 + sem0, 2)
            if n1 > 0:
                mean1 = float(np.mean(r1))
                sem1 = float(np.std(r1, ddof=1) / np.sqrt(n1)) if n1 > 1 else 0.0
                entry["ref_biggan"] = round(mean1, 2)
                entry["ref_biggan_low"] = round(mean1 - sem1, 2)
                entry["ref_biggan_high"] = round(mean1 + sem1, 2)
        out.append(entry)
    return out


def build_psth_json(mean_arr: np.ndarray, sem_arr: np.ndarray, evoked_rate: float,
                    trial_rates: np.ndarray | None) -> dict:
    """Build one method's psth.json (time_ms, mean_rate, sem, trial_rates, evoked_rate, n_trials)."""
    n_time = mean_arr.shape[0]
    time_ms = [round(float(t), 1) for t in np.arange(n_time) * PSTH_BIN_MS]
    n_trials = trial_rates.shape[0] if trial_rates is not None else 0
    return {
        "time_ms": time_ms,
        "mean_rate": [round(float(r), 2) for r in mean_arr],
        "sem": [round(float(s), 2) for s in sem_arr],
        "trial_rates": [
            [round(float(r), 2) for r in trial_rates[i]]
            for i in range(n_trials)
        ] if trial_rates is not None else [],
        "evoked_rate": round(float(evoked_rate), 1),
        "n_trials": n_trials,
    }


def export_experiment(
    Expi: int,
    meta_row: dict,
    resp_bunch: np.ndarray,
    psth_col0: list[np.ndarray],
    psth_col1: list[np.ndarray],
    output_dir: Path,
    exp_id: str,
    evol_meta: dict | None = None,
    include_trial_rates: bool = True,
    ref_resp_arr0: list[np.ndarray] | None = None,
    ref_resp_arr1: list[np.ndarray] | None = None,
) -> dict:
    """
    Write one experiment's folder: meta.json, evol_traj.json, summary_stats.json, gen_XX/deepsim_psth.json, biggan_psth.json.
    psth_col0 / psth_col1: list of (time x trials) arrays per block.
    ref_resp_arr0/1: optional per-block ref image response arrays (thread 0 = DeepSim, thread 1 = BigGAN).
    Returns meta dict for experiments.json.
    """
    exp_dir = output_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    animal = str(meta_row.get("Animal", ""))
    expdate = meta_row.get("expdate")
    date_str = expdate.strftime("%Y-%m-%d") if hasattr(expdate, "strftime") else str(expdate)
    area = str(meta_row.get("visual_area", ""))
    prefchan = int(meta_row.get("prefchan", 0))
    prefunit = int(meta_row.get("prefunit", 0))
    unit_str = f"Ch{prefchan}U{prefunit}"
    n_gen = resp_bunch.shape[0]

    # Evolution trajectory (with optional ref response per gen)
    evol_traj = build_evol_traj(resp_bunch, ref_resp_arr0=ref_resp_arr0, ref_resp_arr1=ref_resp_arr1)
    with open(exp_dir / "evol_traj.json", "w") as f:
        json.dump(evol_traj, f, indent=2)

    # Per-generation PSTH
    for g in range(n_gen):
        gen_num = g + 1
        gen_dir = exp_dir / f"gen_{gen_num:02d}"
        gen_dir.mkdir(exist_ok=True)

        # Evoked rate (mean in response window) for this block
        ev_ds = float(resp_bunch[g, 0])
        ev_bg = float(resp_bunch[g, 1])

        # PSTH mean/sem from precomputed trajectory; trial-level from raw psth if available
        if g < len(psth_col0) and g < len(psth_col1):
            p0 = psth_col0[g]   # (n_time, n_trials)
            p1 = psth_col1[g]
            mean0 = np.mean(p0, axis=1)
            sem0 = np.std(p0, axis=1, ddof=1) / np.sqrt(p0.shape[1]) if p0.shape[1] > 1 else np.zeros(p0.shape[0])
            mean1 = np.mean(p1, axis=1)
            sem1 = np.std(p1, axis=1, ddof=1) / np.sqrt(p1.shape[1]) if p1.shape[1] > 1 else np.zeros(p1.shape[0])
            trial_rates0 = p0.T  # (n_trials, n_time)
            trial_rates1 = p1.T
        else:
            mean0 = np.zeros(51)
            sem0 = np.zeros(51)
            mean1 = np.zeros(51)
            sem1 = np.zeros(51)
            trial_rates0 = None
            trial_rates1 = None

        for method, mean_arr, sem_arr, ev_rate, trials in [
            ("deepsim", mean0, sem0, ev_ds, trial_rates0),
            ("biggan", mean1, sem1, ev_bg, trial_rates1),
        ]:
            trials_out = trials if include_trial_rates else None
            psth_data = build_psth_json(mean_arr, sem_arr, ev_rate, trials_out)
            with open(gen_dir / f"{method}_psth.json", "w") as f:
                json.dump(psth_data, f)

    # Meta
    meta = {
        "id": exp_id,
        "animal": animal,
        "unit": unit_str,
        "area": area,
        "date": date_str,
        "num_generations": n_gen,
        "deepsim_max_rate": round(float(np.max(resp_bunch[:, 0])), 1),
        "biggan_max_rate": round(float(np.max(resp_bunch[:, 1])), 1),
        "tags": [],
    }
    if evol_meta:
        meta.update(evol_meta)
    with open(exp_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary stats
    summary = {
        "id": exp_id,
        "num_generations": n_gen,
        "deepsim_final_rate": round(float(resp_bunch[-1, 0]), 1),
        "biggan_final_rate": round(float(resp_bunch[-1, 1]), 1),
        "deepsim_rate_increase": round(float(resp_bunch[-1, 0] - resp_bunch[0, 0]), 1),
        "biggan_rate_increase": round(float(resp_bunch[-1, 1] - resp_bunch[0, 1]), 1),
    }
    with open(exp_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Export BFEStats neural data to Neural Evolution Explorer format"
    )
    parser.add_argument(
        "--output", "-o",
        default="./neural-evolution-explorer/public/data",
        help="Output directory (default: ./neural-evolution-explorer/public/data)",
    )
    parser.add_argument(
        "--pkl-dir",
        default=None,
        help="Directory containing Both_BigGAN_FC6_Evol_Stats_expsep.pkl (default: use load_neural_data() matroot)",
    )
    parser.add_argument(
        "--experiments", "-n",
        type=int,
        default=None,
        help="Max number of experiments to export (default: all)",
    )
    parser.add_argument(
        "--no-trial-rates",
        action="store_true",
        help="Do not export individual trial responses in PSTH JSON (saves storage; mean/sem still exported)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BFEStats...")
    BFEStats = load_bfestats(args.pkl_dir)
    print(f"  Loaded {len(BFEStats)} experiments.")

    print("Extracting evolution trajectories and meta...")
    resp_col, meta_df = extract_all_evol_trajectory_dyna(BFEStats)

    exp_indices = sorted(resp_col.keys())
    if args.experiments is not None:
        exp_indices = exp_indices[: args.experiments]

    all_meta = []
    for Expi in exp_indices:
        if Expi not in meta_df.index:
            continue
        meta_row = meta_df.loc[Expi]
        resp_bunch = resp_col[Expi]
        animal = str(meta_row["Animal"])
        exp_id = exp_id_from_expi(Expi, animal)

        # Per-block PSTH arrays: list of (time x trials)
        psth_col0, _, _, _ = extract_evol_psth_array(BFEStats[Expi - 1], 0)
        psth_col1, _, _, _ = extract_evol_psth_array(BFEStats[Expi - 1], 1)
        # Drop last block if truncated in resp_col (same logic as in extract_*)
        if len(psth_col0[-1]) < 10 or len(psth_col1[-1]) < 10:
            psth_col0 = psth_col0[:-1]
            psth_col1 = psth_col1[:-1]
        if len(psth_col0) != resp_bunch.shape[0]:
            # Keep same number of blocks as resp_bunch
            n_blocks = resp_bunch.shape[0]
            psth_col0 = psth_col0[:n_blocks]
            psth_col1 = psth_col1[:n_blocks]

        evol_meta = get_evol_meta_for_export(BFEStats, Expi)
        ref_resp_arr0, ref_resp_arr1 = None, None
        S = BFEStats[Expi - 1]
        if S.get("ref") is not None and isinstance(S.get("ref"), dict) and "psth" in S.get("ref", {}):
            try:
                refresp_arr0, _, _, _, _, _ = extract_natref_activation_array(S, 0)
                refresp_arr1, _, _, _, _, _ = extract_natref_activation_array(S, 1)
                n_blocks = resp_bunch.shape[0]
                ref_resp_arr0 = refresp_arr0[:n_blocks]
                ref_resp_arr1 = refresp_arr1[:n_blocks]
            except (KeyError, TypeError, IndexError):
                ref_resp_arr0, ref_resp_arr1 = None, None
        meta = export_experiment(
            Expi,
            meta_row.to_dict(),
            resp_bunch,
            psth_col0,
            psth_col1,
            output_dir,
            exp_id,
            evol_meta=evol_meta,
            include_trial_rates=not args.no_trial_rates,
            ref_resp_arr0=ref_resp_arr0,
            ref_resp_arr1=ref_resp_arr1,
        )
        all_meta.append(meta)
        print(f"  Exported {exp_id} ({meta_row['visual_area']}, {meta_row['blockN']} blocks)")

    with open(output_dir / "experiments.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nDone. Exported {len(all_meta)} experiments to {output_dir.resolve()}")
    print("  experiments.json, plus per-experiment meta.json, evol_traj.json, summary_stats.json, gen_XX/*.json")
    print("  Note: Image files (deepsim_img.png, biggan_img.png) are not generated; add them separately if needed.")


if __name__ == "__main__":
    main()
