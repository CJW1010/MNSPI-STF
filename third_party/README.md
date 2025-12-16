# Third-party algorithms (NOT redistributed)

This repository supports the reproducibility of a **cloud removal → spatiotemporal fusion** workflow.
For licensing/availability reasons, it **does NOT redistribute** third-party implementations of core algorithms.

Users must obtain implementations from the **original authors/sources** and use them in compliance with the corresponding licenses/terms.

## Algorithms and citations (DOIs)

- **MNSPI**  
  Zhu, X., Gao, F., Liu, D., Chen, J., 2011.  
  IEEE Geosci. Remote Sens. Lett. 9(3), 521–525.  
  https://doi.org/10.1109/LGRS.2011.2173290

- **STARFM**  
  Gao, F., Masek, J., Schwaller, M., Hall, F., 2006.  
  IEEE Trans. Geosci. Remote Sens. 44(8), 2207–2218.  
  https://doi.org/10.1109/TGRS.2006.872081

- **ESTARFM**  
  Zhu, X., Chen, J., Gao, F., Chen, X., Masek, J.G., 2010.  
  Remote Sens. Environ. 114(11), 2610–2623.  
  https://doi.org/10.1016/j.rse.2010.05.032

- **FSDAF**  
  Zhu, X., Helmer, E.H., Gao, F., Liu, D., Chen, J., Lefsky, M.A., 2016.  
  Remote Sens. Environ. 172, 165–177.  
  https://doi.org/10.1016/j.rse.2015.11.016

## Evaluation framework reference (APA)

The APA-style “all-round performance assessment” concept used in the manuscript follows:  
Zhu, X., Zhan, W., Zhou, J., Chen, X., Liang, Z., Xu, S., Chen, J., 2022.  
Remote Sensing of Environment, 274, 113002.  
https://doi.org/10.1016/j.rse.2022.113002

## Local configuration (paths only)

To reproduce the full pipeline locally, create:

`configs/third_party_paths.yaml`

Example (local machine paths):
```yaml
mnspi_root: "PATH_TO_MNSPI_CODE"
starfm_root: "PATH_TO_STARFM_CODE"
estarfm_root: "PATH_TO_ESTARFM_CODE"
fsdaf_root: "PATH_TO_FSDAF_CODE"
