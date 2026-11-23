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
