"""
Pkl Inspector — Run this on your .pkl file to see its structure.
This helps you figure out how to adapt load_experiment_pkl().

Usage:
    python inspect_pkl.py your_file.pkl
    python inspect_pkl.py your_file.pkl --depth 4
"""

import pickle
import sys
import numpy as np
from pathlib import Path


def inspect(obj, depth=0, max_depth=3, prefix=""):
    indent = "  " * depth
    
    if depth > max_depth:
        print(f"{indent}...")
        return
    
    if isinstance(obj, dict):
        print(f"{indent}{prefix}dict ({len(obj)} keys)")
        for k, v in obj.items():
            key_str = f'["{k}"]' if isinstance(k, str) else f'[{k}]'
            if isinstance(v, (dict, list, tuple)):
                inspect(v, depth + 1, max_depth, prefix=f"{key_str} → ")
            elif isinstance(v, np.ndarray):
                print(f"{indent}  {key_str} → ndarray shape={v.shape} dtype={v.dtype} "
                      f"range=[{v.min():.3g}, {v.max():.3g}]")
            elif isinstance(v, (int, float)):
                print(f"{indent}  {key_str} → {type(v).__name__}: {v}")
            elif isinstance(v, str):
                val = v[:60] + "..." if len(v) > 60 else v
                print(f"{indent}  {key_str} → str: \"{val}\"")
            else:
                print(f"{indent}  {key_str} → {type(v).__name__}")
    
    elif isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        print(f"{indent}{prefix}{type_name} (len={len(obj)})")
        if len(obj) > 0:
            print(f"{indent}  [0]:")
            inspect(obj[0], depth + 2, max_depth)
            if len(obj) > 1:
                print(f"{indent}  ... ({len(obj) - 1} more items)")
    
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{prefix}ndarray shape={obj.shape} dtype={obj.dtype} "
              f"range=[{obj.min():.3g}, {obj.max():.3g}]")
    
    else:
        # Custom class/object
        print(f"{indent}{prefix}{type(obj).__name__}")
        if hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                if k.startswith('_'):
                    continue
                attr_str = f".{k}"
                if isinstance(v, (dict, list, tuple)):
                    inspect(v, depth + 1, max_depth, prefix=f"{attr_str} → ")
                elif isinstance(v, np.ndarray):
                    print(f"{indent}  {attr_str} → ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"{indent}  {attr_str} → {type(v).__name__}: {repr(v)[:80]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pkl.py <file.pkl> [--depth N]")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    max_depth = 3
    if "--depth" in sys.argv:
        idx = sys.argv.index("--depth")
        max_depth = int(sys.argv[idx + 1])
    
    print(f"{'='*60}")
    print(f"Inspecting: {pkl_path}")
    print(f"File size: {Path(pkl_path).stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    inspect(data, max_depth=max_depth)
    
    print(f"\n{'='*60}")
    print("Use the structure above to adapt load_experiment_pkl()")
    print(f"in preprocess_data.py to map your fields.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
