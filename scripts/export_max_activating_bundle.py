"""
Export Max-Activating Image Bundle
==================================
Uses neural_data_lib to load BFEStats, then for each experiment (Expi), each
thread, and each session (evolution block), finds the image that elicited the
maximum firing rate, loads it as a PIL Image, and saves everything to a single
.pkl bundle.

Output bundle structure (pickle):
  {
    "by_experiment": {
      Expi: {
        "thread_0": [
          {"image": PIL.Image | None, "firing_rate": float, "session": int, "image_path": str},
          ...
        ],
        "thread_1": [ ... ],
      },
      ...
    },
    "meta": {
      Expi: {"Animal", "ephysFN", "blockN", ...},  # from neural_data_utils.get_meta_dict
      ...
    },
  }

Usage:
  # Default: load from neural_data_lib matroot, write to current dir
  python scripts/export_max_activating_bundle.py --output max_activating_bundle.pkl

  # Custom pkl dir and stim drive (for image paths)
  python scripts/export_max_activating_bundle.py --output bundle.pkl --pkl-dir /path/to/Mat_Statistics --stim-drive S:

  # Limit to first N experiments
  python scripts/export_max_activating_bundle.py --output bundle.pkl --experiments 5

Performance (why first run is slow, second run faster):
  - load_bfestats: one large .pkl read; 2nd run often cached by OS.
  - load_img_resp_pairs: path resolution may trigger glob() on stim dir (light I/O).
  - Per image: os.path.exists() (stat) + Image.open() (full file read + decode).
  The many small file reads (exists + image bytes) are the main bottleneck and
  benefit most from OS page/dentry cache on a repeat run. Use --time to see split.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import trange

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from neural_data_lib import load_neural_data, load_img_resp_pairs, extract_evol_activation_array
from neural_data_utils import parse_meta, area_mapping

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_bfestats(pkl_dir: str | None):
    """Load BFEStats list. If pkl_dir is set, load from that directory; else use load_neural_data().
    IO: reads one (or two) large .pkl from disk; much faster on 2nd run due to OS page cache."""
    if pkl_dir is not None:
        pkl_dir = Path(pkl_dir)
        with open(pkl_dir / "Both_BigGAN_FC6_Evol_Stats_expsep.pkl", "rb") as f:
            BFEStats = pickle.load(f)
        return BFEStats
    _, BFEStats = load_neural_data()
    return BFEStats


def collect_max_activating_per_session(BFEStats, stimdrive: str = "S:", max_experiments: int | None = None):
    """
    For each Expi, each thread, each session (block): find max-activating image,
    load as PIL, record firing rate. Returns structure suitable for the pkl bundle.
    """
    by_experiment = {}
    meta = {}
    n_exp = len(BFEStats)
    if max_experiments is not None:
        n_exp = min(n_exp, max_experiments)

    for Expi in trange(1, n_exp + 1):
        S = BFEStats[Expi - 1]
        if S.get("evol") is None:
            continue

        # Build meta without get_meta_dict to avoid neuro_data_analysis import
        try:
            Animal, expdate = parse_meta(S)
        except Exception:
            Animal, expdate = None, None
        ephysFN = S["meta"].get("ephysFN")
        prefchan = int(S["evol"]["pref_chan"][0])
        prefunit = int(S["evol"]["unit_in_pref_chan"][0])
        visual_area = area_mapping(prefchan, Animal, expdate) if Animal else None
        spacenames = S["evol"]["space_names"]
        space1 = spacenames[0] if isinstance(spacenames[0], str) else spacenames[0][0]
        space2 = spacenames[1] if isinstance(spacenames[1], str) else spacenames[1][0]
        resp_arr0, bsl_arr0, gen_arr0, _, _, _ = extract_evol_activation_array(S, 0)
        resp_arr1, bsl_arr1, gen_arr1, _, _, _ = extract_evol_activation_array(S, 1)
        if not resp_arr0 or not resp_arr1:
            blockN = 0
        elif len(resp_arr0[-1]) < 10 or len(resp_arr1[-1]) < 10:
            blockN = len(resp_arr0) - 1
        else:
            blockN = len(resp_arr0)
        meta[Expi] = {
            "Animal": Animal, "expdate": expdate, "ephysFN": ephysFN,
            "prefchan": prefchan, "prefunit": prefunit, "visual_area": visual_area,
            "space1": space1, "space2": space2, "blockN": blockN,
        }
        by_experiment[Expi] = {"thread_0": [], "thread_1": []}

        for thread in (0, 1):
            try:
                imgfps_arr, resp_arr, bsl_arr, gen_arr = load_img_resp_pairs(
                    BFEStats, Expi, "Evol", thread=thread, stimdrive=stimdrive, output_fmt="arr"
                )
            except Exception as e:
                print(f"Expi {Expi} thread {thread}: load_img_resp_pairs failed: {e}")
                continue

            # Same last-block trim as in neural_data_lib
            if len(resp_arr) > 0 and len(resp_arr[-1]) < 10:
                resp_arr = resp_arr[:-1]
                imgfps_arr = imgfps_arr[:-1]
                gen_arr = gen_arr[:-1]

            for session_i, (fps, resp) in enumerate(zip(imgfps_arr, resp_arr)):
                if len(resp) == 0:
                    continue
                max_idx = int(np.argmax(resp))
                firing_rate = float(resp[max_idx])
                image_path = fps[max_idx]
                gen_num = int(gen_arr[session_i][0]) if hasattr(gen_arr[session_i], "__len__") else session_i + 1

                image_pil = None
                # IO: exists() = one stat() per path; Image.open + decode = full file read (main bottleneck).
                # Both are much faster on 2nd run when paths/files are in OS cache (e.g. NFS dentry/page cache).
                if HAS_PIL and image_path and os.path.exists(image_path):
                    try:
                        image_pil = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        print(f"Expi {Expi} thread {thread} session {session_i}: could not open {image_path}: {e}")

                by_experiment[Expi][f"thread_{thread}"].append({
                    "image": image_pil,
                    "firing_rate": firing_rate,
                    "session": session_i,
                    "gen": gen_num,
                    "image_path": image_path,
                })

    return {"by_experiment": by_experiment, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description="Export max-activating image + firing rate bundle (pkl)")
    parser.add_argument("--output", "-o", type=str, default="max_activating_bundle.pkl", help="Output .pkl path")
    parser.add_argument("--pkl-dir", type=str, default=None, help="Directory containing *_expsep.pkl (default: use neural_data_lib matroot)")
    parser.add_argument("--stim-drive", type=str, default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets", help="Stimulus drive letter/path substitution (e.g. N: -> S:)")
    parser.add_argument("--experiments", type=int, default=None, help="Max number of experiments to process (default: all)")
    parser.add_argument("--time", action="store_true", help="Print rough timing for load / collect / save (shows IO bottleneck).")
    args = parser.parse_args()

    if not HAS_PIL:
        print("Warning: PIL/Pillow not installed. 'image' fields in the bundle will be None; firing_rate and image_path will still be set.")

    import time as _time
    t0 = _time.perf_counter()
    print("Loading BFEStats...")
    BFEStats = load_bfestats(args.pkl_dir)
    t_load = _time.perf_counter() - t0
    print(f"Loaded {len(BFEStats)} experiments.")
    if args.time:
        print(f"  [timing] load BFEStats: {t_load:.1f}s (IO: disk read; faster on 2nd run if cached)")

    t1 = _time.perf_counter()
    print("Collecting max-activating image and firing rate per session per thread...")
    bundle = collect_max_activating_per_session(
        BFEStats, stimdrive=args.stim_drive, max_experiments=args.experiments
    )
    t_collect = _time.perf_counter() - t1
    if args.time:
        print(f"  [timing] collect (paths + exists + Image.open): {t_collect:.1f}s (IO: many file reads; dominant on first run)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t2 = _time.perf_counter()
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    t_save = _time.perf_counter() - t2
    print(f"Saved bundle to {out_path} ({out_path.stat().st_size / 1024:.1f} KB).")
    if args.time:
        print(f"  [timing] save pkl: {t_save:.1f}s (IO: disk write)")
        print(f"  [timing] total: {t_load + t_collect + t_save:.1f}s")
    print("Keys:", list(bundle.keys()))
    if bundle["by_experiment"]:
        ex = next(iter(bundle["by_experiment"]))
        n0 = len(bundle["by_experiment"][ex]["thread_0"])
        n1 = len(bundle["by_experiment"][ex]["thread_1"])
        print(f"Example Expi {ex}: thread_0 has {n0} sessions, thread_1 has {n1} sessions.")


if __name__ == "__main__":
    main()
