"""
Export Max-Activating Images to Neural Evolution Explorer
=========================================================
Reads the max-activating bundle (.pkl) and writes one image per generation per
method into the same directory layout used by export_neural_data.py, so the
frontend can show real max-activating stimuli.

Expects bundle structure (from export_max_activating_bundle.py):
  {
    "by_experiment": {
      Expi: {
        "thread_0": [ {"image": PIL.Image, "firing_rate": float, "session": int, "gen": int, "image_path": str}, ... ],
        "thread_1": [ ... ],
      },
      ...
    },
    "meta": { Expi: {"Animal", "ephysFN", "blockN", ...}, ... },
  }

Output layout (matches export_neural_data.py + frontend expectations):
  output_dir/
    {exp_id}/
      gen_01/
        deepsim_img.webp  # max-activating image for thread_0 (default: webp for compression)
        biggan_img.webp   # max-activating image for thread_1 (use --format png for PNG)
      gen_02/
        ...
      ...

Usage:
  # Default: bundle on OneDrive, output to public/data
  python scripts/export_max_activating_images.py

  # Custom paths
  python scripts/export_max_activating_images.py --bundle /path/to/bundle.pkl --output ./neural-evolution-explorer/public/data

  # Limit to first N experiments
  python scripts/export_max_activating_images.py --experiments 5

  # Output format: webp (default, better compression) or png
  python scripts/export_max_activating_images.py --format webp
  python scripts/export_max_activating_images.py --format png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Allow running from repo root
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

DEFAULT_BUNDLE = (
    "/Users/binxuwang/OneDrive - Harvard University/Mat_Statistics/"
    "Both_BigGAN_FC6_max_activating_bundle.pkl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "neural-evolution-explorer"
    / "public"
    / "data"
)


def exp_id_from_expi(Expi: int, animal: str) -> str:
    """Unique, filesystem-safe experiment id (same as export_neural_data.py)."""
    return f"Exp{Expi:03d}-{animal}"


def export_images_from_bundle(
    bundle_path: Path,
    output_dir: Path,
    max_experiments: int | None = None,
    output_format: str = "webp",
) -> None:
    """
    Load the bundle and write max-activating images per generation into
    output_dir/{exp_id}/gen_{NN}/deepsim_img.{ext} and biggan_img.{ext}.
    output_format: "webp" (default, better compression) or "png".
    """
    ext = "webp" if output_format.lower() == "webp" else "png"
    save_kw = {}
    if ext == "webp":
        save_kw = {"format": "WEBP", "quality": 85}
    filenames = (f"deepsim_img.{ext}", f"biggan_img.{ext}")
    thread_files = [("thread_0", filenames[0]), ("thread_1", filenames[1])]
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    by_experiment = bundle["by_experiment"]
    meta = bundle.get("meta", {})

    exp_indices = sorted(by_experiment.keys())
    if max_experiments is not None:
        exp_indices = exp_indices[:max_experiments]

    for Expi in exp_indices:
        data = by_experiment[Expi]
        m = meta.get(Expi, {})
        animal = str(m.get("Animal", "Unknown"))
        exp_id = exp_id_from_expi(Expi, animal)
        exp_dir = output_dir / exp_id

        for thread_key, filename in thread_files:
            entries = data.get(thread_key, [])
            for entry in entries:
                img = entry.get("image")
                if img is None:
                    continue
                gen = entry.get("gen", entry.get("session", 0) + 1)
                gen_dir = exp_dir / f"gen_{gen:02d}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                out_path = gen_dir / filename
                img.save(out_path, **save_kw)
        print(f"  Exported {exp_id} ({len(data.get('thread_0', []))} deepsim, {len(data.get('thread_1', []))} biggan generations)")

    print(f"\nDone. Wrote max-activating images for {len(exp_indices)} experiments to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Export max-activating images from bundle to Neural Evolution Explorer public/data layout"
    )
    parser.add_argument(
        "--bundle", "-b",
        type=str,
        default=DEFAULT_BUNDLE,
        help=f"Path to max-activating bundle .pkl (default: {DEFAULT_BUNDLE})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--experiments", "-n",
        type=int,
        default=None,
        help="Export only the first N experiments (default: all)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=("webp", "png"),
        default="webp",
        help="Output image format: webp (default, better compression) or png",
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.is_file():
        print(f"Error: bundle not found: {bundle_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bundle from {bundle_path}...")
    export_images_from_bundle(
        bundle_path,
        output_dir,
        max_experiments=args.experiments,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
