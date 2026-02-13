"""
Generate Pseudo Data for Neural Evolution Explorer
===================================================
Creates realistic mock data in public/data/ so you can test
the frontend immediately without any real .pkl files.

Usage:
    python generate_pseudo_data.py
    python generate_pseudo_data.py --output ./public/data --experiments 8
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠  Pillow not installed — will skip image generation.")
    print("   Install with: pip install Pillow\n")


# =============================================================================
# CONFIG
# =============================================================================

EXPERIMENTS = [
    {"animal": "Beto",  "unit": "003", "area": "IT",  "date": "2024-10-18"},
    {"animal": "Beto",  "unit": "007", "area": "IT",  "date": "2024-10-18"},
    {"animal": "Beto",  "unit": "014", "area": "V4",  "date": "2024-10-22"},
    {"animal": "Caos",  "unit": "012", "area": "V4",  "date": "2024-11-22"},
    {"animal": "Caos",  "unit": "019", "area": "IT",  "date": "2024-11-22"},
    {"animal": "Alfa",  "unit": "002", "area": "IT",  "date": "2025-01-05"},
    {"animal": "Alfa",  "unit": "008", "area": "V4",  "date": "2025-01-05"},
    {"animal": "Alfa",  "unit": "021", "area": "IT",  "date": "2025-01-12"},
]

NUM_GENERATIONS = 20
TIME_BINS = np.arange(0, 204, 4)  # 0 to 200 ms in 4ms bins
NUM_TRIALS = 15


# =============================================================================
# PSEUDO NEURAL DATA GENERATORS
# =============================================================================

def generate_psth_mean(base_rate, peak_rate, onset_ms=50, peak_ms=90, width_ms=30):
    """Generate a realistic PSTH mean curve with onset transient."""
    rates = []
    for t in TIME_BINS:
        # Baseline + onset response (gaussian bump)
        if t < onset_ms:
            r = base_rate + np.random.randn() * 3
        else:
            onset = (peak_rate - base_rate) * np.exp(-((t - peak_ms)**2) / (2 * width_ms**2))
            sustained = (peak_rate - base_rate) * 0.4 * (1 - np.exp(-(t - onset_ms) / 40))
            r = base_rate + onset + sustained + np.random.randn() * 5
        rates.append(max(0, r))
    return np.array(rates)


def generate_trial_psths(mean_psth, n_trials, variability=0.3):
    """Generate individual trial PSTHs around the mean."""
    trials = []
    for _ in range(n_trials):
        # Each trial is mean + correlated noise
        noise = np.random.randn(len(mean_psth)) * mean_psth * variability
        # Smooth the noise slightly for realism
        kernel = np.array([0.15, 0.3, 0.55, 0.3, 0.15])
        kernel /= kernel.sum()
        noise = np.convolve(noise, kernel, mode='same')
        trial = np.maximum(0, mean_psth + noise)
        trials.append(trial)
    return np.array(trials)


def generate_evol_trajectory(n_gens, start_rate, end_rate, noise=8):
    """Generate CMAES evolution trajectory with realistic convergence curve."""
    # Saturating exponential + noise
    t = np.linspace(0, 1, n_gens)
    trajectory = start_rate + (end_rate - start_rate) * (1 - np.exp(-3 * t))
    trajectory += np.random.randn(n_gens) * noise
    # Smoothly increasing variance
    sems = np.linspace(noise * 0.8, noise * 1.5, n_gens) + np.random.rand(n_gens) * 3
    return trajectory, sems


# =============================================================================
# IMAGE GENERATORS (procedural — no real neural images needed)
# =============================================================================

def generate_deepsim_image(gen, total_gens, seed):
    """Generate an abstract pattern image (simulates DeepSim output)."""
    if not HAS_PIL:
        return None

    np.random.seed(seed + gen * 137)
    size = 256
    img = Image.new('RGB', (size, size))
    pixels = np.zeros((size, size, 3), dtype=np.uint8)

    progress = gen / max(total_gens - 1, 1)

    # Layer 1: Swirling gradient base
    for y in range(size):
        for x in range(size):
            nx, ny = x / size, y / size
            v1 = np.sin(nx * 6 + gen * 0.4) * np.cos(ny * 5 - gen * 0.3)
            v2 = np.sin((nx + ny) * 8 + gen * 0.2) * 0.5
            v3 = np.cos(nx * 3 - ny * 4 + gen * 0.5) * 0.3
            v = (v1 + v2 + v3 + 2) / 4

            # Colors shift with generation (more structured = higher gen)
            r = int(np.clip(v * 160 + progress * 80 + 20, 0, 255))
            g = int(np.clip(v * 120 + (1 - progress) * 60 + 40, 0, 255))
            b = int(np.clip((1 - v) * 180 + progress * 40 + 30, 0, 255))
            pixels[y, x] = [r, g, b]

    img = Image.fromarray(pixels)

    # Add some blur for realism (neural network outputs are smooth)
    blur_amount = max(1, int(3 - progress * 2))
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount))

    return img


def generate_biggan_image(gen, total_gens, seed):
    """Generate an object-like image (simulates BigGAN output)."""
    if not HAS_PIL:
        return None

    np.random.seed(seed + gen * 251)
    size = 256
    progress = gen / max(total_gens - 1, 1)

    # Background: natural green/brown gradient
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            ny = y / size
            pixels[y, x] = [
                int(40 + ny * 30 + np.random.randint(0, 15)),
                int(80 + ny * 50 + np.random.randint(0, 20)),
                int(30 + ny * 20 + np.random.randint(0, 10)),
            ]

    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)

    # Central "object" blob that becomes more defined with generation
    cx, cy = size // 2 + int(np.random.randn() * 10), size // 2 + int(np.random.randn() * 10)
    obj_size = int(40 + progress * 50)

    # Draw layered ellipses for a rough object shape
    colors = [
        (int(140 + progress * 60), int(100 + progress * 40), int(60 + progress * 30)),
        (int(160 + progress * 50), int(120 + progress * 30), int(80 + progress * 20)),
        (int(180 + progress * 40), int(140 + progress * 20), int(100 + progress * 10)),
    ]
    for i, color in enumerate(colors):
        s = obj_size - i * 12
        if s > 0:
            draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color)

    # Detail spots
    n_spots = int(3 + progress * 8)
    for _ in range(n_spots):
        sx = cx + int(np.random.randn() * obj_size * 0.6)
        sy = cy + int(np.random.randn() * obj_size * 0.6)
        sr = int(3 + np.random.rand() * 8)
        spot_color = (
            int(np.clip(colors[1][0] + np.random.randint(-30, 30), 0, 255)),
            int(np.clip(colors[1][1] + np.random.randint(-30, 30), 0, 255)),
            int(np.clip(colors[1][2] + np.random.randint(-20, 20), 0, 255)),
        )
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=spot_color)

    img = img.filter(ImageFilter.GaussianBlur(radius=max(1, int(2 - progress))))

    return img


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_experiment(exp_config, output_dir, exp_index):
    """Generate all data for one experiment."""
    exp_id = f"{exp_config['animal']}-{exp_config['unit']}"
    exp_dir = Path(output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    seed = hash(exp_id) % 10000

    # Randomize neural characteristics per experiment
    np.random.seed(seed)
    ds_base = 40 + np.random.rand() * 40       # DeepSim baseline rate
    ds_peak = 150 + np.random.rand() * 150      # DeepSim max evoked rate
    bg_base = 50 + np.random.rand() * 50        # BigGAN baseline rate
    bg_peak = 180 + np.random.rand() * 160      # BigGAN max evoked rate

    ds_onset = 45 + np.random.rand() * 15       # onset latency
    bg_onset = 40 + np.random.rand() * 20
    ds_peak_t = 80 + np.random.rand() * 30      # peak latency
    bg_peak_t = 75 + np.random.rand() * 35

    # Generate evolution trajectories
    ds_traj, ds_sems = generate_evol_trajectory(NUM_GENERATIONS, ds_base + 30, ds_peak, noise=12)
    bg_traj, bg_sems = generate_evol_trajectory(NUM_GENERATIONS, bg_base + 40, bg_peak, noise=10)

    evol_traj = []
    all_gen_stats = []

    print(f"\n  {exp_id} ({exp_config['area']}, {exp_config['date']})")

    for g in range(NUM_GENERATIONS):
        gen_num = g + 1
        gen_dir = exp_dir / f"gen_{gen_num:02d}"
        gen_dir.mkdir(exist_ok=True)

        progress = g / max(NUM_GENERATIONS - 1, 1)

        # Current evoked rates
        ds_rate = float(ds_traj[g])
        bg_rate = float(bg_traj[g])

        # Generate PSTHs
        ds_mean = generate_psth_mean(ds_base, ds_rate, ds_onset, ds_peak_t, width_ms=25 + progress * 10)
        bg_mean = generate_psth_mean(bg_base, bg_rate, bg_onset, bg_peak_t, width_ms=22 + progress * 12)

        ds_trials = generate_trial_psths(ds_mean, NUM_TRIALS, variability=0.25 + progress * 0.1)
        bg_trials = generate_trial_psths(bg_mean, NUM_TRIALS, variability=0.22 + progress * 0.08)

        ds_sem = np.std(ds_trials, axis=0) / np.sqrt(NUM_TRIALS)
        bg_sem = np.std(bg_trials, axis=0) / np.sqrt(NUM_TRIALS)

        # Save PSTH JSONs
        for method, mean, sem, trials, rate in [
            ("deepsim", ds_mean, ds_sem, ds_trials, ds_rate),
            ("biggan", bg_mean, bg_sem, bg_trials, bg_rate),
        ]:
            psth_data = {
                "time_ms": [round(float(t), 1) for t in TIME_BINS],
                "mean_rate": [round(float(r), 2) for r in mean],
                "sem": [round(float(s), 2) for s in sem],
                "trial_rates": [
                    [round(float(r), 2) for r in trial]
                    for trial in trials
                ],
                "evoked_rate": round(rate, 1),
                "n_trials": NUM_TRIALS,
            }
            with open(gen_dir / f"{method}_psth.json", "w") as f:
                json.dump(psth_data, f)

        # Generate and save images
        ds_img = generate_deepsim_image(gen_num, NUM_GENERATIONS, seed)
        bg_img = generate_biggan_image(gen_num, NUM_GENERATIONS, seed)

        if ds_img:
            ds_img.save(gen_dir / "deepsim_img.png")
        if bg_img:
            bg_img.save(gen_dir / "biggan_img.png")

        # Evolution trajectory point
        evol_traj.append({
            "gen": gen_num,
            "deepsim": round(ds_rate, 2),
            "deepsim_low": round(ds_rate - float(ds_sems[g]), 2),
            "deepsim_high": round(ds_rate + float(ds_sems[g]), 2),
            "biggan": round(bg_rate, 2),
            "biggan_low": round(bg_rate - float(bg_sems[g]), 2),
            "biggan_high": round(bg_rate + float(bg_sems[g]), 2),
        })

        # Progress indicator
        bar = "█" * (gen_num * 20 // NUM_GENERATIONS) + "░" * (20 - gen_num * 20 // NUM_GENERATIONS)
        print(f"\r    [{bar}] Gen {gen_num:2d}/{NUM_GENERATIONS}  "
              f"DS={ds_rate:6.1f} Hz  BG={bg_rate:6.1f} Hz", end="", flush=True)

    print()  # newline after progress bar

    # Save evolution trajectory
    with open(exp_dir / "evol_traj.json", "w") as f:
        json.dump(evol_traj, f, indent=2)

    # Save metadata
    meta = {
        "id": exp_id,
        "animal": exp_config["animal"],
        "unit": f"Unit {exp_config['unit']}",
        "area": exp_config["area"],
        "date": exp_config["date"],
        "num_generations": NUM_GENERATIONS,
        "deepsim_max_rate": round(float(np.max(ds_traj)), 1),
        "biggan_max_rate": round(float(np.max(bg_traj)), 1),
        "tags": [],
    }
    with open(exp_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Summary stats
    summary = {
        "id": exp_id,
        "num_generations": NUM_GENERATIONS,
        "deepsim_final_rate": round(float(ds_traj[-1]), 1),
        "biggan_final_rate": round(float(bg_traj[-1]), 1),
        "deepsim_rate_increase": round(float(ds_traj[-1] - ds_traj[0]), 1),
        "biggan_rate_increase": round(float(bg_traj[-1] - bg_traj[0]), 1),
    }
    with open(exp_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo data for Neural Evolution Explorer")
    parser.add_argument("--output", "-o", default="./public/data",
                        help="Output directory (default: ./public/data)")
    parser.add_argument("--experiments", "-n", type=int, default=None,
                        help=f"Number of experiments to generate (default: all {len(EXPERIMENTS)})")
    parser.add_argument("--generations", "-g", type=int, default=20,
                        help="Generations per experiment (default: 20)")
    args = parser.parse_args()

    global NUM_GENERATIONS
    NUM_GENERATIONS = args.generations

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_configs = EXPERIMENTS[:args.experiments] if args.experiments else EXPERIMENTS

    print("=" * 60)
    print("Neural Evolution Explorer — Pseudo Data Generator")
    print("=" * 60)
    print(f"Output:       {output_dir.resolve()}")
    print(f"Experiments:  {len(exp_configs)}")
    print(f"Generations:  {NUM_GENERATIONS} per experiment")
    print(f"Trials:       {NUM_TRIALS} per generation")
    print(f"Images:       {'Yes (Pillow installed)' if HAS_PIL else 'No (install Pillow)'}")

    all_meta = []
    for i, exp_config in enumerate(exp_configs):
        meta = generate_experiment(exp_config, str(output_dir), i)
        all_meta.append(meta)

    # Write master index
    with open(output_dir / "experiments.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    # Summary
    total_files = len(exp_configs) * NUM_GENERATIONS * 4  # 2 psth + 2 img per gen
    total_files += len(exp_configs) * 3  # meta + evol_traj + summary per exp
    total_files += 1  # experiments.json

    print(f"\n{'=' * 60}")
    print(f"✓ Generated {len(all_meta)} experiments")
    print(f"  {total_files} total files in {output_dir.resolve()}")
    print(f"\n  Next steps:")
    print(f"    cd your-vite-project")
    print(f"    npm run dev")
    print(f"    # Open http://localhost:5173")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
