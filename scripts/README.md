# Scripts

This folder contains all scripts required to reproduce the experiments and figures in the manuscript.

## Recommended workflow
1. **Data preparation** (if you have access to original data)
   - preprocessing, band stacking, co-registration
2. **Cloud scenario construction**
   - simulated cloud masks (Perlin noise + morphological closing/opening)
3. **Cloud removal (MNSPI)**
   - performed using third-party implementation (not redistributed)
4. **Spatiotemporal fusion**
   - STARFM / ESTARFM / FSDAF using third-party implementations (not redistributed)
5. **Evaluation**
   - compute metrics (RMSE/AD/EDGE/LBP), generate APA plots, and run classification

## Folder structure (to be populated)
- `00_setup/`
  - dependency checks and wrapper scripts for third-party codes
- `01_cloud_simulation/`
  - scripts to generate simulated cloud masks and scenarios
- `02_metrics_apa/`
  - scripts to compute metrics and generate APA/radar plots
- `03_classification/`
  - scripts to train/evaluate land-cover classification and post-processing (e.g., median filtering)

## Notes on third-party algorithms
MNSPI/STARFM/ESTARFM/FSDAF implementations are not included.  
See `third_party/README.md` for installation instructions and how to link external implementations to this workflow.
