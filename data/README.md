# Data

This study uses GF-1 WFV imagery and Sentinel-2 L2A surface reflectance products.

## What is provided in this repository
Due to potential licensing and data-use restrictions on full satellite scenes, this repository may not redistribute the complete original imagery used in the manuscript. Instead, we provide:

- **Data organization templates and naming conventions** (folder structure used in the experiments)
- **Experiment metadata** describing:
  - study area and study period
  - acquisition dates used in each scenario (as reported in the manuscript tables)
  - scenario definitions (simulated thin/thick cloud; single/multiple patches; real-cloud case)
- **Guidance for reproducing data preparation steps**, including band selection, resampling, stacking, cropping, and co-registration settings consistent with the manuscript

## What is NOT redistributed
- Full original GF-1 WFV scenes and Sentinel-2 L2A scenes used in the manuscript (when redistribution is not permitted).
- Any third-party algorithm implementations (see `third_party/README.md`).

## How to obtain the data
- **Sentinel-2 L2A**: available from public Copernicus/ESA distribution portals. Users can query the study area and acquisition dates reported in the manuscript.
- **GF-1 WFV**: should be obtained from the official data provider/platform accessible to the user or institution, using the study area and acquisition dates reported in the manuscript.

## Study period and region
- Region: Gaoyou Lake and surrounding plains, Yancheng, Jiangsu Province, China.
- Period: April–June 2020 (see manuscript Tables for exact acquisition dates).

## Reproducibility note
End-to-end reproduction of the full pipeline requires:
1) access to the satellite imagery listed above,
2) third-party implementations of MNSPI/STARFM/ESTARFM/FSDAF (not redistributed),
3) applying the preprocessing and experimental settings reported in the manuscript (Section 2).

If data redistribution is restricted, we recommend that users reproduce the workflow by:
- downloading Sentinel-2 L2A for the same area and dates,
- obtaining GF-1 WFV via institutional access,
- and following the same scenario settings and evaluation protocol documented in the manuscript.
