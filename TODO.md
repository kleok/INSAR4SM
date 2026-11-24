# TODO: `geocoded` branch (NISAR + Sentinel-1, no ISCE2)

Open technical questions and checks for the geocoded-SLC workflow. Tick items off as they are verified.

## Open questions

### 1. NISAR GSLC phase flattening
- [ ] **Check whether NISAR L2 GSLC phase is flattened** (topographic/geometric phase removed).
  `GEOCODED_stack.calc_insar_stack` ([insar4sm/geocoded_stack.py](insar4sm/geocoded_stack.py)) sets `perp_baseline_stack` to zeros for every sensor, assuming flattened SLCs.
  - Sentinel-1 COMPASS: explicit, COMPASS runs with `flatten: True` ([insar4sm/S1/s1_cslc.py](insar4sm/S1/s1_cslc.py)).
  - Sentinel-1 OPERA CSLC-S1: ✅ verified on `OPERA_L2_CSLC-S1_T100-213521-IW3_20251201T133628Z_..._v1.1`, where `topographic_flattening_applied = True` and `ellipsoidal_flattening_applied = True`. [insar4sm/S1/opera_cslc.py](insar4sm/S1/opera_cslc.py) checks these flags on every burst and refuses unflattened ones.
  - NISAR: **not verified.** To check:
    - the NISAR GSLC product specification
    - the granule metadata, e.g. `/science/LSAR/GSLC/metadata/processingInformation/parameters/` for a flattening flag
    - the phase of an interferogram between two GSLCs over the Ford Dry Lake test AOI (residual topographic fringes ⇒ not flattened)
  - If it is not flattened: flatten with the DEM and the orbits, or model the perpendicular baseline term per acquisition.

### 2. L-band model parameters
- [ ] `sm_dry_state`, `beta` (`InSAR_phase_model`), `dry_dates_ratio`, `opt_parms` and the dry-date logic were tuned for Sentinel-1 C-band. Validate them for NISAR L-band (HH, ~5 cm sensing depth) against ISMN 5 cm in-situ data.

### 3. NISAR product name
- [ ] `GSLC_SHORT_NAME = 'NISAR_L2_GSLC_PROVISIONAL_V1'` ([insar4sm/NISAR/nisar_gslc.py](insar4sm/NISAR/nisar_gslc.py)) will change when the final (non-provisional) products are released. Make it configurable or search both.

### 3b. OPERA CSLC-S1
- [ ] Coverage is North America only. Outside it, `s1_source = 'COMPASS'` (needs `INSAR4SM_env_compass.yml`).
- [ ] Bursts of the same track may come in different UTM zones (e.g. the T100 burst at Ford Dry Lake, −115.1°, is in EPSG:32612). `read_opera_window` raises an error if the bursts of a track differ; reproject instead if that happens for an AOI.
- [ ] The Ford Dry Lake AOI is at the western edge of burst T100-213521-IW3: ~13 % of the AOI has no T100 data.
- [ ] The `azimuth_carrier_phase` layer is not removed. It is the same on every date of a burst on a fixed grid, so it should cancel in the interferograms; verify on the coherence.

### 4. Incidence angle approximation
- [ ] `local_incidence_angle` takes the satellite position at the zero-Doppler time of the grid centre only. That's fine for AOIs of a few km; check the error for larger AOIs, or compute it per row.

### 5. Results change vs the published ISCE2 workflow
The following fixes from PR #14 change soil moisture results compared with the paper:
- [ ] Dry-date inequality constraints in `inversion.py`: previously only the last dry date was constrained (lambda late binding).
- [ ] Precipitation: ERA5-Land `tp` is now de-accumulated to hourly values and summed over `days_back` **days** (previously 24 rows of accumulated hourly values).
- [ ] Dry-date precipitation test is now `<=` the quantile (previously `<`, which selected no dates when most dates have no rain).
- [ ] Compare Ford Dry Lake results with the paper results in `Data/INSAR4SM_r1_orb*_125m_FordDryLake_sm_point_ts.obj` (Sentinel-1 orbits 100, 166, 173).

## Housekeeping
- [ ] Merge conflict with `dev`: `insar4sm/inversion.py` line with `np.bool_` (dev) vs `bool` (this branch). Keep `bool`.
- [ ] `.readthedocs.yaml` points to `docs/conf.py`, which doesn't exist. Remove it or add docs.
- [ ] Add CI (GitHub Actions) running `pytest tests/`.
- [ ] Version/release tag on `main` before merging this branch, so ISCE2 users can stay on the last ISCE2 version.

## Test case
- Site: **Ford Dry Lake** SCAN station (33.65470 N, −115.09763 E), BWh, shrubland, sand 79 % / clay 11 %.
- AOI: 2 × 2 km around the station, in [tests/data/FordDryLake_aoi.geojson](tests/data/FordDryLake_aoi.geojson).
- Smoke test (~2 months): [tests/smoke/run_ford_dry_lake.py](tests/smoke/run_ford_dry_lake.py), compared with ISMN 5 cm soil moisture.
- Offline unit tests: `pytest tests/`.

### Smoke test result (2026-11-23, NISAR + OPERA S1, 2025-12-01 → 2026-01-31): ✅ runs end to end
Soil moisture in %, compared with ISMN 5 cm. Station cell = InSAR4SM grid cell containing the station.

| Sensor | Track | Dates | RMSE station cell | R station cell | RMSE AOI mean | RMSE ERA5-Land | R ERA5-Land |
|---|---|---|---|---|---|---|---|
| NISAR | 005A | 5 | 13.0 | 0.28 | 6.2 | 5.2 | 0.84 |
| NISAR | 085D | 5 | 4.1 | **0.94** | 7.0 | 6.8 | 0.98 |
| S1 | 100D | 5 | **1.3** | 0.66 | 5.5 | 4.8 | 0.70 |
| S1 | 166A | 5 | 5.0 | −0.10 | 3.3 | 6.5 | 0.87 |
| S1 | 173D | 10 | 4.1 | −0.12 | 3.8 | 5.6 | 0.83 |

This is a smoke test, not a validation (2 months, 5–10 dates, DS test off, 1 dry date). Findings to follow up:
- [x] ~~**NaN dates in short stacks**~~: short stacks only. The 12-month stacks have 83–100 % valid cells (see below).
  Original observation: in the 5-date stacks, one or two dates are NaN in most cells (e.g. last date of S1 100D in 231/256 cells, 2nd date of NISAR 085D in 213/256). The 10-date stack (173D) has almost none. Check which step returns NaN (sorting / coherence / inversion) for short stacks.
- [ ] Short stacks always use the "Coherence loss due to temporal decorrelation cannot be estimated" fallback: with 5 dates, `int(5 × 0.3) = 1` dry date, so there is no dry-date pair. `coh_funcs.py` now also catches the `ValueError` this raises.
- [ ] NISAR 005A overestimates (13–33 % vs 8–10 % in situ). NISAR 085D catches the 2026-01-03 rain event (in situ 16 %), but overshoots it (24–31 %). → L-band calibration (§2).
- [ ] S1 173D underestimates in December (3 % vs 8–9 %).
- [x] Rerun over 12 months with `DS_flag = True` (see below).

### Validation run (2026-11-24, 12 months 2025-11-22 → 2026-11-11, DS test on, NISAR + OPERA S1): ✅
Command: `run_ford_dry_lake.py --time-start 20251122T000000 --time-end 20261111T230000 --ds --min-acquisitions 10` (~150 min, mostly downloads).
Soil moisture in %, compared with ISMN 5 cm. Valid cells = fraction of grid cells with an estimate on every date.

| Sensor | Track | Dates | Valid cells | RMSE station cell | R station cell | RMSE AOI mean | R AOI mean | RMSE ERA5-Land | R ERA5-Land |
|---|---|---|---|---|---|---|---|---|---|
| NISAR | 005A | 26 | 1.00 | 3.0 | 0.88 | 2.6 | 0.86 | 3.9 | 0.92 |
| NISAR | 085D | 25 | 1.00 | 3.5 | 0.34 | **1.1** | **0.94** | 4.5 | 0.75 |
| NISAR | 106A | 23 | 0.00 | – | – | – | – | 3.4 | 0.81 |
| S1 | 100D | 29 | 0.83 | **1.4** | **0.93** | **1.1** | **0.97** | 3.5 | 0.84 |
| S1 | 166A | 31 | 0.99 | 2.8 | 0.92 | 1.9 | 0.91 | 4.4 | 0.78 |
| S1 | 173D | 61 | 0.99 | 1.9 | 0.82 | 1.5 | 0.94 | 3.7 | 0.84 |

- InSAR4SM beats ERA5-Land in RMSE on every track: AOI mean 1.1–2.6 % vs 3.5–4.5 %. Its AOI-mean R is comparable or better: 0.86–0.97 vs 0.75–0.92.
- Sentinel-1 (C-band) and NISAR (L-band) both work with the current (C-band-tuned) parameters. The L-band calibration (§2) is still worth doing (NISAR 085D station cell R 0.34).
- [ ] **NISAR 106A**: frame 018 covers only 2 % of the AOI on every date (the AOI is at the frame edge), so no estimate. Skip tracks whose stack covers less than a minimum fraction of the AOI (e.g. 50 %) before the inversion, instead of inverting an empty stack.
- [ ] NISAR 157D was skipped (9 acquisitions < 10).
- [ ] Compare with the paper results in `Data/*FordDryLake*.obj` (S1 orbits 100/166/173).
- [x] Plots: time series of InSAR4SM per track vs ISMN and ERA5-Land, plus a map of the AOI estimates → done in the case study (below).

### Case study (2026-11-24): ✅ [case_studies/FordDryLake/](case_studies/FordDryLake/)
Built from the 12-month run: a 6-page PDF report (`FordDryLake_case_study.pdf`) and a notebook (`FordDryLake_case_study.ipynb`) that runs from the small files in `results/` (~260 kB). Pooled over the 5 tracks (172 estimates): InSAR4SM RMSE **1.7 %**, bias −0.9 %, R **0.90**, vs ERA5-Land RMSE 4.0 %, bias −3.4 %, R 0.84. That's an estimate on 137 days, with a median gap of 2 days.
- [ ] Update the repository link in the report and notebook (currently "branch *geocoded*") once `geocoded` is merged into `main`.
- [ ] `build_report.py` needs `reportlab`, which is not in either environment. Add it to `INSAR4SM_env.yml`, or keep it as an optional `pip install reportlab`.
- [ ] `results/` contains ISMN in-situ values (at the acquisition times and as daily means). Confirm this is fine under the ISMN data policy before publishing (SCAN data is public; ISMN asks for citation).
- [ ] When `prepare_results.py` or `case_study.py` change, rebuild with:
  `python case_studies/FordDryLake/prepare_results.py ...`, `jupyter nbconvert --execute --inplace ...ipynb` and `python case_studies/FordDryLake/build_report.py`.
