"""
Neural Evolution Experiment Data Preprocessor
=============================================
Reads .pkl / .mat experiment data and exports structured JSON + images
for the web frontend.

USAGE:
    python preprocess_data.py --input ./raw_data --output ./public/data

OUTPUT STRUCTURE:
    public/data/
    ├── experiments.json                    ← master index
    ├── Beto-003/
    │   ├── meta.json                       ← experiment metadata
    │   ├── evol_traj.json                  ← CMAES trajectory
    │   ├── gen_01/
    │   │   ├── deepsim_img.png
    │   │   ├── biggan_img.png
    │   │   ├── deepsim_psth.json
    │   │   └── biggan_psth.json
    │   ├── gen_02/
    │   │   └── ...
    │   └── summary_stats.json              ← quick-load stats
    └── Caos-012/
        └── ...

CUSTOMIZATION:
    You'll need to adapt the `load_experiment_pkl()` and/or
    `load_experiment_mat()` functions to match YOUR specific
    pkl/mat structure. The rest of the pipeline is generic.
"""

import os
import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Uncomment if you use .mat files:
# import scipy.io as sio

# Uncomment if images are stored as arrays and you want to save PNGs:
# from PIL import Image


# =============================================================================
# DATA CLASSES — defines what the frontend expects
# =============================================================================

@dataclass
class ExperimentMeta:
    """Metadata for one experiment/unit."""
    id: str               # unique key, e.g. "Beto-003"
    animal: str           # e.g. "Beto"
    unit: str             # e.g. "Unit 003"
    area: str             # e.g. "IT", "V4"
    date: str             # e.g. "2024-10-18"
    num_generations: int
    deepsim_max_rate: float
    biggan_max_rate: float
    tags: list            # optional tags for filtering


@dataclass
class PSTHData:
    """PSTH data for one generation + one method."""
    time_ms: list         # [0, 4, 8, ..., 200]  — time axis
    mean_rate: list       # mean PSTH across trials
    sem: list             # standard error of mean
    trial_rates: list     # list of lists, one per trial
    evoked_rate: float    # scalar summary firing rate
    n_trials: int


@dataclass
class GenerationData:
    """All data for one generation."""
    gen: int
    deepsim_psth: PSTHData
    biggan_psth: PSTHData
    deepsim_best_rate: float
    biggan_best_rate: float
    # Image paths are saved separately as PNGs


@dataclass
class EvolTrajPoint:
    """One point on the evolution trajectory."""
    gen: int
    deepsim: float        # mean response
    deepsim_low: float    # lower CI bound
    deepsim_high: float   # upper CI bound
    biggan: float
    biggan_low: float
    biggan_high: float


# =============================================================================
# LOADING FUNCTIONS — ⚠️  ADAPT THESE TO YOUR DATA STRUCTURE ⚠️
# =============================================================================

def load_experiment_pkl(pkl_path: str) -> dict:
    """
    Load a single experiment from a .pkl file.
    
    ⚠️  YOU MUST ADAPT THIS to your specific pkl structure!
    
    Expected return format:
    {
        "animal": str,
        "unit": str,
        "area": str,
        "date": str,
        "generations": [
            {
                "gen": int,
                "deepsim": {
                    "image": np.ndarray (H, W, 3) or None,
                    "psth_time": np.ndarray,          # time points in ms
                    "psth_mean": np.ndarray,           # mean firing rate
                    "psth_sem": np.ndarray,            # SEM
                    "psth_trials": np.ndarray (n_trials, n_timepoints),
                    "best_rate": float,
                },
                "biggan": { ... same structure ... }
            },
            ...
        ]
    }
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    
    # =========================================================
    # EXAMPLE: Adapt this section to YOUR pkl structure
    # =========================================================
    # Below is a TEMPLATE showing common patterns. Replace with
    # your actual field names and data layout.
    #
    # Common patterns in neuroscience pkl files:
    #
    #   raw["meta"]["animal"]
    #   raw["meta"]["unit_id"]
    #   raw["meta"]["area"]
    #   raw["meta"]["date"]
    #   raw["generations"][i]["deepsim"]["best_image"]    # np array
    #   raw["generations"][i]["deepsim"]["psth"]           # (n_trials, n_time)
    #   raw["generations"][i]["deepsim"]["time_axis"]      # ms
    #   raw["generations"][i]["deepsim"]["evoked_rate"]    # scalar
    #   raw["generations"][i]["biggan"][...]               # same
    #
    # If your data uses different keys, just map them here:
    
    # --- OPTION A: If your pkl is a dict with clear structure ---
    if isinstance(raw, dict):
        # Try to extract metadata
        meta = raw.get("meta", raw.get("info", {}))
        animal = meta.get("animal", meta.get("subject", "Unknown"))
        unit = meta.get("unit", meta.get("unit_id", meta.get("channel", "Unknown")))
        area = meta.get("area", meta.get("brain_area", meta.get("region", "Unknown")))
        date = str(meta.get("date", meta.get("exp_date", "Unknown")))
        
        # Try to extract generations
        gens_raw = raw.get("generations", raw.get("blocks", raw.get("gen_data", [])))
        
        generations = []
        for i, gen_raw in enumerate(gens_raw):
            gen_num = gen_raw.get("gen", gen_raw.get("generation", i + 1))
            
            def extract_method(d, method_key):
                """Extract one method's data from a generation dict."""
                m = d.get(method_key, {})
                
                # Try common key patterns for PSTH
                psth_trials = m.get("psth_trials", m.get("psth", m.get("raster", None)))
                time_axis = m.get("time_axis", m.get("psth_time", m.get("t", None)))
                image = m.get("image", m.get("best_image", m.get("img", None)))
                best_rate = m.get("best_rate", m.get("evoked_rate", m.get("firing_rate", 0.0)))
                
                # Compute mean/sem if we have trial data
                if psth_trials is not None:
                    psth_trials = np.array(psth_trials)
                    if psth_trials.ndim == 1:
                        psth_trials = psth_trials.reshape(1, -1)
                    psth_mean = np.mean(psth_trials, axis=0)
                    psth_sem = np.std(psth_trials, axis=0) / np.sqrt(psth_trials.shape[0])
                else:
                    psth_mean = m.get("psth_mean", m.get("mean_rate", np.zeros(50)))
                    psth_sem = m.get("psth_sem", m.get("sem", np.zeros(50)))
                    psth_mean = np.array(psth_mean)
                    psth_sem = np.array(psth_sem)
                    psth_trials = np.array([psth_mean])  # fallback
                
                # Generate time axis if missing
                if time_axis is None:
                    time_axis = np.linspace(0, 200, len(psth_mean))
                
                return {
                    "image": np.array(image) if image is not None else None,
                    "psth_time": np.array(time_axis),
                    "psth_mean": psth_mean,
                    "psth_sem": psth_sem,
                    "psth_trials": psth_trials,
                    "best_rate": float(best_rate),
                }
            
            generations.append({
                "gen": gen_num,
                "deepsim": extract_method(gen_raw, "deepsim"),
                "biggan": extract_method(gen_raw, "biggan"),
            })
        
        return {
            "animal": animal,
            "unit": unit,
            "area": area,
            "date": date,
            "generations": generations,
        }
    
    # --- OPTION B: If your pkl is a custom class/object ---
    # Adapt attribute names accordingly:
    # return {
    #     "animal": raw.animal,
    #     "unit": raw.unit_id,
    #     ...
    # }
    
    raise ValueError(f"Unrecognized pkl structure: {type(raw)}. "
                     f"Top-level keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}")


def load_experiment_mat(mat_path: str) -> dict:
    """
    Load a single experiment from a .mat file.
    
    ⚠️  YOU MUST ADAPT THIS to your specific .mat structure!
    Returns same format as load_experiment_pkl().
    """
    import scipy.io as sio
    raw = sio.loadmat(mat_path, squeeze_me=True)
    
    # ADAPT THIS to your .mat structure
    # .mat files often have nested structs that scipy loads as
    # numpy structured arrays or dict-like objects.
    #
    # Example for common Matlab struct patterns:
    #
    #   raw["ExpData"]["animal"]
    #   raw["ExpData"]["generations"][0]["deepsim"]["psth"]
    
    raise NotImplementedError(
        "Adapt load_experiment_mat() to your .mat structure.\n"
        f"Available keys: {[k for k in raw.keys() if not k.startswith('__')]}"
    )


# =============================================================================
# EXPORT FUNCTIONS — these produce the web-ready output
# =============================================================================

def save_image(image_array: Optional[np.ndarray], output_path: str):
    """Save a numpy image array as PNG."""
    if image_array is None:
        return False
    
    from PIL import Image
    
    img = np.array(image_array)
    
    # Handle different formats
    if img.dtype == np.float64 or img.dtype == np.float32:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    if img.ndim == 2:  # grayscale
        Image.fromarray(img, mode='L').save(output_path)
    elif img.ndim == 3 and img.shape[2] == 3:  # RGB
        Image.fromarray(img, mode='RGB').save(output_path)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA
        Image.fromarray(img, mode='RGBA').save(output_path)
    else:
        print(f"  ⚠ Skipping image with unexpected shape: {img.shape}")
        return False
    
    return True


def export_psth_json(psth_time, psth_mean, psth_sem, psth_trials, evoked_rate) -> dict:
    """Convert PSTH arrays to JSON-serializable dict."""
    return {
        "time_ms": [round(float(t), 1) for t in psth_time],
        "mean_rate": [round(float(r), 2) for r in psth_mean],
        "sem": [round(float(s), 2) for s in psth_sem],
        "trial_rates": [
            [round(float(r), 2) for r in trial]
            for trial in psth_trials
        ],
        "evoked_rate": round(float(evoked_rate), 2),
        "n_trials": int(psth_trials.shape[0]) if hasattr(psth_trials, 'shape') else len(psth_trials),
    }


def process_experiment(exp_data: dict, exp_id: str, output_dir: str):
    """
    Process one experiment and write all files to output_dir/exp_id/.
    """
    exp_dir = Path(output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    generations = exp_data["generations"]
    num_gens = len(generations)
    
    # --- Save metadata ---
    meta = ExperimentMeta(
        id=exp_id,
        animal=exp_data["animal"],
        unit=exp_data["unit"],
        area=exp_data["area"],
        date=exp_data["date"],
        num_generations=num_gens,
        deepsim_max_rate=max(g["deepsim"]["best_rate"] for g in generations),
        biggan_max_rate=max(g["biggan"]["best_rate"] for g in generations),
        tags=[],
    )
    with open(exp_dir / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)
    
    # --- Process each generation ---
    evol_traj = []
    
    for gen_data in generations:
        gen_num = gen_data["gen"]
        gen_dir = exp_dir / f"gen_{gen_num:02d}"
        gen_dir.mkdir(exist_ok=True)
        
        print(f"  Gen {gen_num:2d}/{num_gens}", end="")
        
        for method in ["deepsim", "biggan"]:
            m = gen_data[method]
            
            # Save image
            img_path = gen_dir / f"{method}_img.png"
            saved = save_image(m.get("image"), str(img_path))
            if saved:
                print(f" [{method}:img✓]", end="")
            
            # Save PSTH
            psth_json = export_psth_json(
                m["psth_time"], m["psth_mean"], m["psth_sem"],
                m["psth_trials"], m["best_rate"]
            )
            with open(gen_dir / f"{method}_psth.json", "w") as f:
                json.dump(psth_json, f)
            print(f" [psth✓]", end="")
        
        # Build evolution trajectory point
        # Use mean ± SEM of the trial evoked rates if available
        ds = gen_data["deepsim"]
        bg = gen_data["biggan"]
        ds_rate = ds["best_rate"]
        bg_rate = bg["best_rate"]
        
        # Estimate CI from trial variability
        ds_sem = float(np.std(ds["psth_mean"])) * 0.1 if ds["psth_mean"] is not None else 10
        bg_sem = float(np.std(bg["psth_mean"])) * 0.1 if bg["psth_mean"] is not None else 10
        
        evol_traj.append({
            "gen": gen_num,
            "deepsim": round(ds_rate, 2),
            "deepsim_low": round(ds_rate - ds_sem, 2),
            "deepsim_high": round(ds_rate + ds_sem, 2),
            "biggan": round(bg_rate, 2),
            "biggan_low": round(bg_rate - bg_sem, 2),
            "biggan_high": round(bg_rate + bg_sem, 2),
        })
        
        print()  # newline
    
    # --- Save evolution trajectory ---
    with open(exp_dir / "evol_traj.json", "w") as f:
        json.dump(evol_traj, f, indent=2)
    
    # --- Save summary stats for quick loading ---
    summary = {
        "id": exp_id,
        "num_generations": num_gens,
        "deepsim_final_rate": generations[-1]["deepsim"]["best_rate"],
        "biggan_final_rate": generations[-1]["biggan"]["best_rate"],
        "deepsim_rate_increase": (
            generations[-1]["deepsim"]["best_rate"] - generations[0]["deepsim"]["best_rate"]
        ),
        "biggan_rate_increase": (
            generations[-1]["biggan"]["best_rate"] - generations[0]["biggan"]["best_rate"]
        ),
    }
    with open(exp_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return meta


def discover_experiments(input_dir: str) -> list:
    """
    Find all experiment files in input_dir.
    Returns list of (filepath, experiment_id) tuples.
    
    ⚠️  ADAPT the file discovery pattern to your naming convention!
    """
    input_path = Path(input_dir)
    experiments = []
    
    # Pattern 1: Each experiment is a separate .pkl file
    for pkl_file in sorted(input_path.glob("*.pkl")):
        # Derive experiment ID from filename
        # e.g., "Beto_18102024_unit003.pkl" → "Beto-003"
        exp_id = pkl_file.stem  # use filename without extension as ID
        experiments.append((str(pkl_file), exp_id, "pkl"))
    
    # Pattern 2: Each experiment is a separate .mat file
    for mat_file in sorted(input_path.glob("*.mat")):
        exp_id = mat_file.stem
        experiments.append((str(mat_file), exp_id, "mat"))
    
    # Pattern 3: One big pkl file with all experiments
    mega_pkl = input_path / "all_experiments.pkl"
    if mega_pkl.exists():
        with open(mega_pkl, "rb") as f:
            all_data = pickle.load(f)
        if isinstance(all_data, dict):
            for exp_id, exp_data in all_data.items():
                experiments.append((str(mega_pkl), exp_id, "mega_pkl"))
    
    return experiments


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess neural evolution experiment data for web frontend"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Directory containing raw .pkl/.mat files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory (e.g., ./public/data)")
    parser.add_argument("--format", choices=["pkl", "mat", "auto"], default="auto",
                        help="Input file format (default: auto-detect)")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Neural Evolution Data Preprocessor")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Discover experiments
    experiments_found = discover_experiments(args.input)
    print(f"Found {len(experiments_found)} experiment(s)\n")
    
    if not experiments_found:
        print("⚠ No experiment files found!")
        print("  Expected .pkl or .mat files in the input directory.")
        print("  Check the discover_experiments() function to match your file naming.")
        return
    
    # Process each experiment
    all_meta = []
    
    for filepath, exp_id, fmt in experiments_found:
        print(f"Processing: {exp_id} ({filepath})")
        
        try:
            if fmt == "pkl":
                exp_data = load_experiment_pkl(filepath)
            elif fmt == "mat":
                exp_data = load_experiment_mat(filepath)
            elif fmt == "mega_pkl":
                with open(filepath, "rb") as f:
                    all_data = pickle.load(f)
                exp_data = all_data[exp_id]
                # You may need to wrap this in your expected format
            else:
                print(f"  ⚠ Unknown format: {fmt}, skipping")
                continue
            
            meta = process_experiment(exp_data, exp_id, str(output_dir))
            all_meta.append(asdict(meta))
            print(f"  ✓ Done\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Write master experiment index
    with open(output_dir / "experiments.json", "w") as f:
        json.dump(all_meta, f, indent=2)
    
    print(f"{'='*60}")
    print(f"✓ Exported {len(all_meta)} experiments to {output_dir}")
    print(f"  Master index: {output_dir / 'experiments.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
