# Summary: Paek et al. 2026, ApJ, 1001, 198

**Title:** A Hybrid Framework for Kilonova Anomaly Detection Using Single-epoch SEDs from the 7-Dimensional Telescope  
**Authors:** Gregory S. H. Paek, Myungshin Im, Seo-Won Chang, Hyeonho Choi, Ji Hoon Kim  
**Published:** ApJ 1001, 198 (33pp), 2026 April 20  
**DOI:** https://doi.org/10.3847/1538-4357/ae5229  

---

## Context and Motivation

The upcoming LSST/Rubin Observatory will generate ~10^7 alerts per night. Within this torrent, identifying rare kilonovae (KNe) — the EM counterparts to binary neutron star (BNS) mergers — is critical for GW-EM multi-messenger astronomy. AT 2017gfo remains the only spectroscopically confirmed KN, making supervised KN classification with labeled training data nearly impossible.

Key challenges:
- Large GW localization areas and faint, rapidly fading KNe
- Most LIGO/Virgo/KAGRA KN candidates were actually SNe, AGN, CVs, asteroids, or other contaminants
- Spectroscopy is too slow for faint/rapidly evolving transients
- Broadband photometric classification requires multiepoch light curves (latency problem)
- Impostors like LBVs and Type IIb SNe can mimic KN photometric behavior for days

The solution proposed: use **single-epoch, medium-band SEDs** from the 7-Dimensional Telescope (7DT), which simultaneously captures many bands in one snapshot, providing ~low-resolution spectrophotometry without multiepoch requirements.

---

## The 7-Dimensional Telescope (7DT)

- Array of **20 × 0.5-m telescopes** at El Sauce Observatory, Chile
- Each telescope: SONY IMX455 CMOS (9576 × 6388 pixels, 0.51"/pix), FoV = 1.3 × 0.9 deg²
- **Mediumband filter system**: 20 filters (m400–m875, 25 nm spacing) currently operational; ultimate goal is **40 filters** (m400–m887, 12.5 nm spacing)
- Simultaneously observes in all bands by distributing filters across telescopes → true single-epoch SED within minutes
- Spectroscopic mode: R ~ 30–70, ~1.2 deg² FoV
- Primary science goal: rapid KNe identification in GW-EM follow-up (7-Dimensional Sky Survey, 7DS)

**7DT-Simulator** (`G. S. Paek & D. Tak 2025`): custom software that generates realistic synthetic photometry accounting for filter throughput, CMOS QE, atmospheric transmission, sky background, readout noise, dark current → magnitude errors with realistic SNR; default 3 × 100s exposures, 2" seeing.

---

## Data

### Non-KN Training Spectra (8 classes)
Collected from OSC and WISeREP, quality-cut to wavelength coverage 387.5–900 nm:

| Class | Label | N (selected) | N (augmented) |
|-------|-------|-------------|---------------|
| Type Ia SN | Ia | 2838 | 11,352 |
| Type II SN | II | 1856 | 11,136 |
| Type Ibc SN | Ibc | 357 | 10,353 |
| Superluminous SN | SLSN | 82 | 10,004 |
| AGN | AGN | 160 | 10,080 |
| TDE | TDE | 372 | 10,044 |
| Stellar Variable (CV, LBV, LRN, Nova, M-dwarf flare) | SV | 453 | 10,032 |
| Asteroid (Bus-DeMeo taxonomy, synthetic) | Asteroid | 10,000 | 10,000 |
| **Total** | | **7654** | **82,997** |

- ~90% of non-KN training spectra lie at z < 0.15 (redshift bias toward nearby sources)
- Asteroids: synthetic SEDs from Bus-DeMeo taxonomy, GP-extrapolated to cover 387.5–900 nm, solar-spectrum multiplied

### Simulated KN Spectra (test set only — NOT used in training)
- Two-component radiative-transfer grid from Wollaeger et al. (2021): toroidal dynamical ejecta (lanthanide-rich) + peanut-shaped wind component (lanthanide-poor)
- Parameters: m_d ∈ {0.001, 0.003, 0.01, 0.03, 0.1} M☉; v_d ∈ {0.05, 0.15, 0.3}c; m_w ∈ {0.001, 0.003, 0.01, 0.03, 0.1} M☉; v_w ∈ {0.05, 0.15, 0.3}c; θ ∈ {0°, 15°, 30°, 45°, 60°, 75°, 90°}; t ∈ {0.125, 0.25, 0.5, 1.0, 2.0, 3.0} days post-merger → 9450 total spectra
- Fiducial distance: D_L = 40 Mpc (z ≈ 0.010)
- Only 23.4% of simulated KNe have peak brightness within 7DT detection limits (SNR > 5 in at least one filter)
- Detection is biased toward high wind ejecta mass and high wind velocity (brighter, hotter SEDs)

### Observed KN: AT 2017gfo
- ENGRAVE spectroscopic sequence: +1.43 to +7.4 days post-merger (7 epochs), VLT/X-shooter
- Used as held-out anomaly detection test only (not used in training)

### Real 7DT Observations (test set)
- SN 2025fvw (Type Ia, multiepoch), SN 2024diq (Type II, single epoch), AT 2024ett (CV, two epochs)
- Observed in 20-filter configuration, processed with gpPy-GPU pipeline

---

## Methods

### Feature Construction
- Input features: **pairwise color indices** (magnitude differences between filter pairs)
- 20-filter set: N_features = 190 colors; 40-filter set: N_features = 780 colors
- Filters with SNR < 3 treated as nondetections → masked (NaN), not excluded — nondetection patterns are astrophysically informative (constrain SED shape)
- Only sources with ≥1 band at SNR > 5 are included

### Data Augmentation
- Each class augmented to ~10^4 synthetic instances using 7DT-Simulator
- Multipliers: AGN × 63, Type II × 6, Type Ia × 4, Ibc × 29, SLSN × 122, SV × 22, TDE × 27
- 80/20 train-test split; augmented duplicates kept within same partition

### 3.1 Multiclass Classifier (XGBoost)
Three GBDT algorithms benchmarked: LightGBM, CatBoost, XGBoost  
**XGBoost selected** for best performance and SHAP compatibility.

- Handles NaN natively (nondetections as informative patterns)
- Hyperparameter tuning: Optuna (100 trials, maximize macro F₁)
- Metric: macro F₁ = harmonic mean of per-class precision and recall

### 3.2 Anomaly Classifier (Isolation Forest)
- `iForest` trained **exclusively on non-KN samples** (no KN in training)
- Output: anomaly score P_ano
- KNe are identified as statistical outliers from the learned "normal" transient distribution
- Avoids supervised KN training data scarcity problem

### 3.3 Combined (Hybrid) Decision Rule
- Two scores combined in 2D decision space:
  - **x-axis**: `1 − P_max` (XGBoost classifier uncertainty; P_max = highest non-KN class probability)
  - **y-axis**: `P_ano` (iForest anomaly score)
- Optimal thresholds via Youden's J statistic: simultaneously maximize TPR and TNR
- Final decision: AND gate — event must exceed both thresholds to be flagged as KN
- 20-filter thresholds: (1 − P_max,th, P_ano,th) = (0.081, −0.099); 40-filter: (0.051, −0.087)

---

## Results

### 4.1 Multiclass Classification Performance

| Config | Model | F₁ (macro) | Accuracy | Precision | Recall |
|--------|-------|-----------|----------|-----------|--------|
| 20 filters | XGBoost | 0.797 | 0.801 | 0.786 | 0.789 |
| 20 filters | CatBoost | 0.801 | 0.806 | 0.791 | 0.795 |
| 40 filters | XGBoost | 0.836 | 0.839 | 0.828 | 0.830 |
| 40 filters | CatBoost | 0.834 | 0.839 | 0.825 | 0.828 |

Per-class highlights (40-filter XGBoost):
- **Asteroid**: highest F₁ ~ 0.97 (spectrally simple, solar-like)
- **Type Ia**: F₁ ~ 0.90 (Si II λ6355 absorption well-sampled)
- **Type II**: F₁ ~ 0.85 (Hα emission feature)
- **Ibc**: F₁ ~ 0.77 (confusion with Type II at ~11% level)
- **SLSN**: lowest F₁ (scarce training data; confused with Type II and SV)
- **TDE/AGN**: ~10% mutual confusion (shared blue continua)

### 4.2 SHAP Feature and Filter Importance

- Most informative filters cluster around spectral diagnostics: **Hα** (m650–m675), **Si II** (m600–m625), **blue continuum** (m400–m462)
- Red-end filters (> ~725 nm) contribute little due to lower throughput and larger scatter
- Key finding: **only ~40–50% of the most informative filters are needed to retain near-baseline performance**
  - 40-filter: restricting to top 40% (36 filters) preserves full model F₁ (0.839 vs 0.836 baseline)
  - Performance drops only when fewer than top 20 filters (50%) are used
  - The **bottom 20 filters** (lowest SHAP importance) yield F₁ ≈ 0.716 — confirming ranked importance is predictive

Per-class SHAP diagnostics:
1. **Ia**: Si II λ6355 trough (m600–m625), blue continuum (m400–m462)
2. **Ibc**: broad continuum slopes (m500–m550); degenerate with Type II
3. **II**: Hα and adjacent complexes (m637–m662, m675–m700)
4. **SLSN**: blue bands (m450–m462), O II complexes 3500–4500 Å
5. **TDE**: blue optical (m400–m512) + Hα, Balmer/He II λ4686
6. **AGN**: Hα neighborhood and continuum-slope tracers
7. **Asteroid**: G2V-like reflected solar; long-baseline colors (m400–m612)
8. **SV**: importance spread across m500–m675 (heterogeneous class)

Filter importance correlates strongly with filter throughput (sensitivity): filters with maximum response < 30% contribute little to classification.

### 4.3 KN Anomaly Detection Performance

**Simulated KN grid (40-filter, Youden's J threshold):**
- KN recall (TPR) = **0.932** (93.2% of detectable KNe flagged)
- Non-KN false positive rate (FPR) = **0.089** (8.9%)

**AT 2017gfo (real observed KN):**
- All phases except +1.43 days correctly flagged as anomalous in both 20- and 40-filter configurations
- +1.43 day epoch: classified as asteroid by XGBoost (featureless power-law continuum mimics solar-like SED); this is the earliest, most disk-dominated phase
- All epochs fall within Youden's J thresholds in hybrid decision space

**Real 7DT transients:**
- SN 2025fvw (Ia), SN 2024diq (II), AT 2024ett (CV): all correctly flagged as **non-anomalous** (no false KN flags)
- Multiclass labels: SN 2025fvw → Ia ✓; SN 2024diq → TDE (wrong, true: II); AT 2024ett → Ia (wrong, true: CV)

**Hybrid vs. iForest-only:**
- iForest alone (P_ano only): FPR ~ 50% at same KN recall → contamination unacceptable
- Hybrid (AND gate): FPR reduced to ~10% while preserving ~93% KN recall

### 4.4 7DT + LSST Synergy

Adding a single LSST broadband (u, g, r, i, z, or y) to the 20 top-SHAP-ranked 7DT filters:
- Performance stays within ~4% of the full 40-filter baseline
- y-band provides the largest boost (+2.7% F₁ for 40-filter set)
- Overall improvement modest (~1–2%), confirming 7DT medium-bands already capture most information
- Operationally useful: SHAP top-20 7DT filters + 1 LSST band ≈ full 40-filter performance

---

## Key Conclusions

1. **Single-epoch medium-band SEDs are sufficient** to separate KNe from common transients without multiepoch light curves.
2. The hybrid framework achieves **F₁ ~ 0.80–0.82** (20–40 filters) for 8-class transient classification.
3. **>90% KN recall** on simulated and observed KNe (including AT 2017gfo) with FPR ~ 10%.
4. Only **~40–50% of 7DT filters** are needed to retain near-baseline classification; red filters (> ~725 nm) contribute little.
5. Adding a single LSST broadband to the top-ranked 7DT filters reproduces full 40-filter accuracy within 1–2%.
6. The framework is **computationally fast** (thousands of sources per second after training), suitable for real-time alert stream filtering.
7. Framework is generalizable to any photometric system with comparable mediumband coverage.

---

## Limitations

- Training sample biased to **z < 0.15** (90th percentile), limiting performance for distant transients
- Hot, featureless early KNe (< 0.5 days) resemble early core-collapse SNe and CVs — indistinguishable at earliest phases
- Dust extinction not corrected for in training/test spectra
- KN anomaly detection calibrated only to Wollaeger et al. 2021 AT 2017gfo-like morphology; intrinsically faint or lanthanide-dominated KNe may be missed
- Missing filter observations require model retraining (NaN interpretation breaks)
- LBV (~25%) and Type IIb SN (~20%) contamination remains nonnegligible (though better than light-curve-only methods)
- M-dwarf flares: 100% flagged as anomalies but training sample is small (15 spectra)

---

## Future Work

- Incorporate **temporal information** (two-epoch configurations; hour-to-day gaps)
- Add **host galaxy contextual features** (photometric redshift, host SED, local environment)
- Extend to NIR-capable cameras or synergy with NIR facilities for high-z and red KNe
- Augment training with theoretical KN models across broader redshift range
- Develop more flexible (nonlinear) classification boundaries for KN isolation

---

## Software and Tools Used

- `7DT-Simulator` (Paek & Tak 2025): synthetic photometry simulator for 7DT
- `gpPy-GPU` (Paek 2025): GPU-accelerated photometric calibration pipeline
- `XGBoost`, `LightGBM`, `CatBoost`: GBDT classifiers
- `Isolation Forest` (scikit-learn): unsupervised anomaly detection
- `SHAP` (Lundberg & Lee 2017): feature importance/interpretability
- `Optuna`: Bayesian hyperparameter optimization
- `speclit`: synthetic LSST photometry from spectra
- `Astropy`, `NumPy`, `SciPy`, `SWarp`, `SCAMP`, `Source Extractor`
