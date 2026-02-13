# Neural Evolution Explorer — Setup Guide

## Overview

This project has two parts:

```
┌─────────────────────┐       ┌─────────────────────┐
│  1. DATA PIPELINE   │       │  2. WEB FRONTEND     │
│  (Python, run once) │──────▶│  (React, deployed)   │
│                     │       │                      │
│  Reads .pkl / .mat  │ JSON  │  Interactive viewer   │
│  Exports JSON + PNG │ + PNG │  with playback, PSTH  │
└─────────────────────┘       └─────────────────────┘
```

---

## Quick Start (5 steps)

### Step 1: Create the project

```bash
npm create vite@latest neural-evolution-explorer -- --template react
cd neural-evolution-explorer
npm install recharts
```

### Step 2: Replace `src/App.jsx`

Copy the provided `src/App.jsx` into your project's `src/App.jsx`.
Delete `src/App.css` and `src/index.css` (they're not needed).

In `src/main.jsx`, make sure it just renders `<App />`:
```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

### Step 3: Set up your Python pipeline

```bash
pip install numpy Pillow
# pip install scipy     ← only if using .mat files
```

Put `preprocess_data.py` somewhere convenient.

### Step 4: Adapt the loader to YOUR data

Open `preprocess_data.py` and edit the `load_experiment_pkl()` function.
The key is mapping YOUR pkl structure to this standard format:

```python
# Your function must return this structure:
{
    "animal": "Beto",            # subject name
    "unit": "Unit 003",          # unit/channel identifier
    "area": "IT",                # brain area
    "date": "2024-10-18",        # experiment date
    "generations": [
        {
            "gen": 1,
            "deepsim": {
                "image": np.ndarray or None,   # (H, W, 3) uint8 or float
                "psth_time": np.ndarray,       # [0, 4, 8, ..., 200] in ms
                "psth_mean": np.ndarray,       # mean firing rate per bin
                "psth_sem": np.ndarray,        # SEM per bin
                "psth_trials": np.ndarray,     # (n_trials, n_bins)
                "best_rate": float,            # evoked firing rate
            },
            "biggan": { ... same ... }
        },
        ...  # one dict per generation
    ]
}
```

**Tip: Inspect your pkl first!**
```python
import pickle
with open("your_file.pkl", "rb") as f:
    data = pickle.load(f)

# Check what's inside:
print(type(data))
if isinstance(data, dict):
    print(data.keys())
    # Drill down:
    for k in data.keys():
        v = data[k]
        print(f"  {k}: {type(v)}", end="")
        if hasattr(v, 'shape'):
            print(f" shape={v.shape}", end="")
        print()
```

### Step 5: Run the pipeline & start the app

```bash
# Export your data
python preprocess_data.py --input ./raw_data --output ./public/data

# Verify the output
ls public/data/
# → experiments.json  Beto-003/  Caos-012/  ...

ls public/data/Beto-003/gen_01/
# → deepsim_img.png  biggan_img.png  deepsim_psth.json  biggan_psth.json

# Start the dev server
npm run dev
# → Open http://localhost:5173
```

---

## File Format Reference

### `public/data/experiments.json`
Master index loaded on page load:
```json
[
  {
    "id": "Beto-003",
    "animal": "Beto",
    "unit": "Unit 003",
    "area": "IT",
    "date": "2024-10-18",
    "num_generations": 20,
    "deepsim_max_rate": 312.5,
    "biggan_max_rate": 321.7,
    "tags": []
  },
  ...
]
```

### `public/data/{exp_id}/evol_traj.json`
Evolution trajectory (one entry per generation):
```json
[
  {
    "gen": 1,
    "deepsim": 120.5,
    "deepsim_low": 108.2,
    "deepsim_high": 132.8,
    "biggan": 145.3,
    "biggan_low": 133.1,
    "biggan_high": 157.5
  },
  ...
]
```

### `public/data/{exp_id}/gen_XX/{method}_psth.json`
PSTH data per generation per method:
```json
{
  "time_ms": [0, 4, 8, 12, ..., 200],
  "mean_rate": [10.5, 12.3, ...],
  "sem": [2.1, 1.8, ...],
  "trial_rates": [
    [8.2, 11.0, ...],
    [12.8, 13.5, ...],
    ...
  ],
  "evoked_rate": 214.4,
  "n_trials": 12
}
```

### Images
- `public/data/{exp_id}/gen_XX/deepsim_img.png`
- `public/data/{exp_id}/gen_XX/biggan_img.png`
- Any resolution works; they'll be scaled to fit the UI

---

## Deploying to the Public Web

### Option A: Vercel (recommended, free)

```bash
npm install -g vercel
vercel
# Follow prompts → get a public URL in seconds
```

### Option B: Netlify

```bash
npm run build
# Drag & drop the `dist/` folder to https://app.netlify.com/drop
```

### Option C: GitHub Pages

```bash
# In vite.config.js, set base:
export default defineConfig({
  base: '/your-repo-name/',
  plugins: [react()]
})

npm run build
# Push dist/ to gh-pages branch
```

---

## Common Customizations

### Different method names (not DeepSim/BigGAN)?
In `preprocess_data.py`, change the method keys in your loader.
In `App.jsx`, search-replace "deepsim" and "biggan" with your names,
and update the display labels and colors.

### More than 2 methods?
Add a third column to the grid in App.jsx:
```jsx
gridTemplateColumns: "1fr 1fr 1fr"  // was "1fr 1fr"
```

### Only have pre-rendered MP4 videos (no raw data)?
You can embed videos instead of the interactive charts:
```jsx
<video src={`/data/${expId}/animation.mp4`}
  controls autoPlay loop muted
  style={{ width: "100%", borderRadius: 8 }} />
```

### Large datasets (hundreds of experiments)?
Add search/filter to the sidebar:
```jsx
const [search, setSearch] = useState("");
const filtered = experiments.filter(e =>
  e.id.toLowerCase().includes(search.toLowerCase()) ||
  e.area.toLowerCase().includes(search.toLowerCase())
);
```

### Want a backend instead of static files?
If you have too much data for static hosting (> 500MB), use FastAPI:

```python
# server.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()

@app.get("/api/experiments")
def list_experiments():
    return json.load(open("data/experiments.json"))

@app.get("/api/experiment/{exp_id}/gen/{gen}")
def get_generation(exp_id: str, gen: int):
    # Load from pkl on demand instead of pre-exported JSON
    ...

app.mount("/", StaticFiles(directory="dist", html=True))
```

Then change `DATA_BASE` in App.jsx to `"/api"`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Blank page, "Failed to load data" | Run `preprocess_data.py` first, check `public/data/experiments.json` exists |
| Images show "No image" | Check PNGs were exported; verify image arrays aren't None in your pkl |
| Charts empty | Check PSTH json files have data; verify `time_ms` and `mean_rate` arrays |
| Playback stutters | Preloader handles this; if still slow, reduce trial count in PSTH export |
| `KeyError` in preprocessing | Your pkl structure doesn't match the template — inspect & adapt the loader |
