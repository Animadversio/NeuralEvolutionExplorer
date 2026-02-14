Exporting neural data for browsing 
===================
Source data bundles are these 
```
Both_BigGAN_FC6_max_activating_bundle.pkl (1.9G)
Both_BigGAN_FC6_Evol_Stats_expsep.pkl (0.9G)
```
We will export them into json and png files for browser. 

Export neural data, spikes, PSTH etc.
```bash
python scripts/export_neural_data.py --output ./neural-evolution-explorer/public/data -n 30  --no-trial-rates
```
Export max activating images. 
```bash
python scripts/export_max_activating_images.py --output ./neural-evolution-explorer/public/data -n 10 --format webp
```

To save static file spaces one can add flags
* No flags 
    * around `4 G`
* Add `--format webp` 
    * data size `627M`
* Both `--format webp` and `--no-trial-rates` (recommended, if limited on space and loading)
    * data size `265M`
