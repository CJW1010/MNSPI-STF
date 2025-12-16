# Data folder

This folder documents the datasets used in the study and provides a **minimal reproducible subset** for reviewers/readers.

## 1. Data sources
- **GF-1 WFV** imagery (fine spatial resolution): processed by the authors (radiometric calibration, atmospheric correction, cropping, and co-registration).
- **Sentinel-2 L2A** surface reflectance products: resampled and co-registered to match the GF-1 spatial grid.

**Important:** The repository may not redistribute raw/original satellite products if they are subject to licensing or distribution restrictions.  
To support reproducibility, we provide a **demo subset** (small ROI) and all scripts to reproduce the analysis pipeline.

## 2. Demo subset (for reproducibility)
Place the reproducible dataset in:

`data/demo_subset/`

Recommended structure (example):

## 3. File naming conventions (recommended)
- `GF1_target_cloudy.tif`: GF-1 image at target date with cloud contamination (input to MNSPI).
- `GF1_aux.tif`: auxiliary-date GF-1 image used for temporal support.
- `GF1_reference_clear.tif`: cloud-free reference used for quantitative evaluation.
- `cloud_mask.tif`: binary mask (1=cloud/cloud-shadow; 0=clear), simulated or manual/refined.

## 4. Coordinate system / resolution
- GF-1 WFV: 16 m (four bands: Blue, Green, Red, NIR).
- Sentinel-2 L2A: resampled/co-registered to match the GF-1 grid for fusion experiments.

## 5. How to reproduce the paper with the demo subset
After placing the demo subset, run the scripts in:
- `scripts/01_cloud_simulation/` (optional if masks already provided)
- `scripts/02_metrics_apa/` (to compute RMSE/AD/EDGE/LBP and generate APA plots)
- `scripts/03_classification/` (to reproduce land-cover classification experiments)
