# Case study: Ford Dry Lake, California (Nov 2025 – Nov 2026)

One year of InSAR4SM surface soil moisture from **NISAR (L-band)** and **Sentinel-1 (C-band)** at the Ford Dry Lake SCAN station (Mojave Desert), validated against in-situ data (ISMN, 5 cm) and compared with ERA5-Land.

| Pooled over 5 tracks, 172 estimates | RMSE | bias | R |
|---|---|---|---|
| **InSAR4SM** (2 × 2 km mean) | **1.7 %** | −0.9 % | **0.90** |
| ERA5-Land (9 km) | 4.0 % | −3.4 % | 0.84 |

- **[FordDryLake_case_study.pdf](FordDryLake_case_study.pdf)**: 6-page report.
- **[FordDryLake_case_study.ipynb](FordDryLake_case_study.ipynb)**: the same analysis as a notebook. It runs in seconds from the files in `results/`, with no SAR download needed.

## Files

| File | Content |
|---|---|
| `results/` | Compact outputs of the 12-month run: track summaries, InSAR4SM / ISMN / ERA5-Land at every acquisition, per-cell estimates, daily reference series |
| `case_study.py` | Statistics and figures (shared by the notebook and the report) |
| `prepare_results.py` | Builds `results/` from a processing run |
| `build_report.py` | Builds the PDF (needs `pip install reportlab`) |
| `figures/` | The figures as PNG |

## Reproduce from scratch

```bash
conda activate insar4sm   # NASA Earthdata in ~/.netrc, Copernicus CDS key in ~/.cdsapirc
python tests/smoke/run_ford_dry_lake.py --project-dir RUN_DIR --ismn-csv ISMN_5cm.csv \
    --time-start 20251122T000000 --time-end 20261111T230000 --ds --min-acquisitions 10   # ~2.5 h
python case_studies/FordDryLake/prepare_results.py --run-dir RUN_DIR --ismn-csv ISMN_5cm.csv
python case_studies/FordDryLake/build_report.py
```

The ISMN data (station *FordDryLake*, network SCAN, 5 cm) is available from https://ismn.earth after free registration.

## References
- Karamvasis, K., & Karathanassi, V. (2023). Soil moisture estimation from Sentinel-1 interferometric observations over arid regions. *Computers & Geosciences*, 178, 105410. https://doi.org/10.1016/j.cageo.2023.105410
- Katsamas, G., Karamvasis, K., & Karathanassi, V. (2026). Toward Improved Soil Moisture Retrieval Using NISAR InSAR Closure Phase Measurements. *EARSeL Symposium, Athens*. [PDF](https://drive.google.com/file/d/1nKk2JU313nhIxNLPmR2ZAa3G-Cnvtlm3/view?usp=drive_link)

## Acknowledgements
<a href="https://github.com/moistx-labs"><img src="../../figures/moistx_logo.png" width="160" alt="moistX"></a>

The development of InSAR4SM is supported by [moistX](https://github.com/moistx-labs). The NISAR and Sentinel-1 geocoded-SLC workflow was contributed by Giorgos Katsamas ([@GiorgosKatsamas](https://github.com/GiorgosKatsamas)).

**Data credits:**
- **In situ:** International Soil Moisture Network (Dorigo et al., 2021), network SCAN (USDA NRCS).
- **Meteorology:** ERA5-Land (Muñoz-Sabater et al., 2021), Copernicus Climate Change Service.
- **SAR:** NISAR L2 GSLC (NASA/ISRO, ASF DAAC); OPERA L2 CSLC-S1 (NASA JPL, ASF DAAC), containing modified Copernicus Sentinel data 2025–2026.
