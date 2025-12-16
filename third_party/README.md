# Third-party algorithms (not redistributed)

This repository provides a reproducible workflow for evaluating a **cloud removal → spatiotemporal fusion** pipeline.  
**It does NOT redistribute** third-party implementations of the following algorithms:

- **MNSPI** (cloud removal / gap filling)
- **STARFM**
- **ESTARFM**
- **FSDAF**

These algorithms are cited in the manuscript and must be obtained from their **original authors/sources** and used in compliance with the corresponding licenses/terms.

## How to use third-party codes with this repository
1. Download/install each third-party implementation from its original source.
2. Configure the local paths in `configs/third_party_paths.yaml` (to be created by the user).
3. Use the wrapper scripts in `scripts/00_setup/` (to be provided) to call the third-party code and standardize inputs/outputs for this project.

## Notes
- If any third-party implementation cannot be installed or redistributed due to licensing restrictions, the workflow in this repository can still be reproduced for the parts that are fully open (e.g., simulated cloud mask generation, metric computation, APA plots, and classification evaluation) using the provided demo subset.
- Please cite the original papers when using these third-party algorithms.
