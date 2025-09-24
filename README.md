# README

**Manuscript:** *The Development and Validity of a Shortened Version of the Wolf Motor Function Test (WMFT-4)*

This repository contains data and analysis code for deriving and validating the 4-item WMFT (WMFT-4) from the original 15-item WMFT using robust preprocessing, factor analysis (CFA), and psychometric testing (validity, MDC/MCID).

---

## What’s new in this revision

- **Unified preprocessing + feature selection in Python** (`01_Preprocessing_DataReduction.py`):  
  Robust multivariate outlier detection (MinCovDet + χ² cutoff) on Box–Cox–transformed features; saves a cleaned analysis set and diagnostics. Also runs SelectKBest + Random Forest with CV, elbow heuristic, and **1,000× bootstrap** stability for item selection. Outputs plots and a selected-feature stability figure.
- **Confirmatory Factor Analysis with bootstrap in R** (`02_factor_analysis_revision1.R`):  
  Compares **one-factor vs two-factor** models on WMFT-15; repeats the comparison for **WMFT-4** (V3, V5, V6, V8). Extracts bootstrapped fit indices (CFI/TLI/RMSEA/SRMR/AIC/BIC) and loading distributions, visualizes SEM paths, and tests model comparison via `anova()`.
- **Expanded validity analyses in Python** (`03_wmft_fm_validity.py`):  
  Adds Pearson correlations with **bootstrapped CIs** for WMFT-4/WMFT-15/FMA-UE and a **non-selected items** composite. Introduces **FMA-UE 3-group** stratification (0–28 / 29–42 / 43–66) with ANOVA, η², **Tukey HSD**, subject-level exports, and publication-ready box-plots.
- **MDC/MCID script updated** (`04_WMFT4_MDC_MCID.R`):  
  Computes distribution-based MCID (0.8× SD of change) and anchor-based MCID using **FMA-UE Δ≥5.25**; clarifies that WMFT is analyzed as **rate** (60/time).

---

## Repository structure

```
├── data/
│   ├── wmft_rate_fm.csv
│   ├── (optional) WMFT_pre_new.csv, WMFT_post_new.csv
│   ├── (optional) FM_new.csv, FM_post_new.csv
│   └── …
├── code/
│   ├── 01_Preprocessing_DataReduction.py
│   ├── 02_factor_analysis_revision1.R
│   ├── 03_wmft_fm_validity.py
│   ├── 04_WMFT4_MDC_MCID.R
│   └── …
├── figures/        # created by scripts (plots, diagrams)
└── README.md
```

**Key intermediate/outputs (created by scripts):**
- `wmft_rate_fm_MVNclean.csv` (+ `…_with_diag` and outlier audit csv) — cleaned dataset for downstream analyses.  
- `mvn_*` PNGs — MVN diagnostics (χ² plot, MD² histogram).  
- `feature_selection_bootstrap.png` — WMFT item selection robustness (1,000 bootstraps).  
- `wmft4_rate_by_3groups_CUSTOM.png`, `wmft15_rate_by_3groups_CUSTOM.png` — box-plots by FMA strata.  
- `wmft_mean_rate_by_subject_3groups_CUSTOM.csv` — subject-level summary.  
- SEM path diagrams saved from R (CFA one-factor vs two-factor; WMFT-15 and WMFT-4).

---

## Software & packages

### Python (≥3.8)
Install with:
```bash
pip install numpy pandas scikit-learn scipy statsmodels matplotlib
```
Used for preprocessing, feature selection, bootstrap stability, correlations, ANOVA/Tukey, figures.

### R (≥4.2)
Install with:
```r
install.packages(c(
  "caret","boot","MASS","Hmisc","MVN","energy","lavaan",
  "lavaanPlot","semPlot","psych","readxl","REdaS","GPArotation","qgraph"
))
```
Used for CFA with bootstrap, SEM plotting, Bartlett/KMO, and MCID/MDC.

---

## Data conventions

- **Rates**: WMFT items are analyzed as **rates** (reps/min = `60 / time[s]`). Scripts auto-convert if they detect time-like values.  
- **Chronicity**: log-transformed prior to CFA.  
- **Outliers**: robust multivariate detection via **MinCovDet** on Box–Cox-transformed features with χ² cutoff (default α=0.001); cleaned rows propagated to downstream steps.

---

## Reproducible workflow

1) **Preprocess + Item Reduction (Python)**
```bash
python code/01_Preprocessing_DataReduction.py
# Edit the script if needed to point to data/wmft_rate_fm.csv
```
Outputs the cleaned CSV (`wmft_rate_fm_MVNclean.csv`), MVN diagnostics, RF CV curves, elbow-based k, and **bootstrap item-selection stability**.

2) **CFA with bootstrap (R)**
```r
source("code/02_factor_analysis_revision1.R")
```
- Bartlett’s test and KMO on transformed dataset  
- **WMFT-15**: one-factor vs two-factor CFA with 1,000× bootstrap summaries (CFI, TLI, RMSEA, SRMR, AIC, BIC), standardized loadings, SEM diagrams, and **χ² model comparison**  
- **WMFT-4** (V3,V5,V6,V8): one- vs two-factor CFA with the same outputs

3) **Validity (Python)**
```bash
python code/03_wmft_fm_validity.py
```
- Pearson correlations + **bootstrapped CIs** for WMFT-4/WMFT-15/FMA-UE  
- Composite of **non-selected items** and its correlations  
- **FMA-UE 3-group** stratification (0–28 / 29–42 / 43–66): ANOVA, η², **Tukey HSD**, box-plots, and subject-level CSV.

4) **MDC & MCID (R)**
```r
source("code/04_WMFT4_MDC_MCID.R")
```
- Distribution-based MCID (0.8× SD of change) for WMFT-4 and WMFT-15  
- **Anchor-based MCID** using **FMA-UE Δ≥5.25** (Page, Fulk, & Boyne, 2012)  
- Notes: script expects pre/post WMFT and FMA CSVs; update file paths at the top accordingly.

---

## Interpretation guide

- **Dimensionality**: Two-factor structure on WMFT-15 supports separable domains (e.g., proximal control vs integrative functional tasks) while the 4-item set remains cohesive and predictive.  
- **Item selection**: V3, V5, V6, V8 are consistently identified as the most informative subset across CV and **bootstrap stability**.  
- **Validity**: WMFT-4 strongly tracks WMFT-15 and FMA-UE; non-selected items form a weaker composite, providing a sanity check that the chosen set is not arbitrary.  
- **Clinical change**: Report both distribution- and anchor-based MCIDs; clarify that rates were used for comparability.

---

## References

- Wolf SL et al. *Stroke.* 2001;32(7):1635–1639.  
- Woodbury M et al. *Neurorehabil Neural Repair.* 2010;24(9):791–801.  
- Bogard K et al. *Neurorehabil Neural Repair.* 2009;23(5):422–428.  
- Page SJ, Fulk GD, Boyne P. *Phys Ther.* 2012;92(5):707–715.

---

## Citation

Kim B, Schweighofer N, Wolf S, Winstein C. *The Development and Validity of a Shortened Version of the Wolf Motor Function Test (WMFT-4).* Manuscript in preparation.

---

## License & data use

Code under MIT unless noted. De-identified data intended for research/education; follow IRB and data-use agreements.

---

## Contact

**Bokkyu Kim** • SUNY Upstate Medical University • kimbo@upstate.edu

---

**Changelist reference**  
This README supersedes the prior version to align with:  
`01_Preprocessing_DataReduction.py` (preprocessing/outliers + selection),  
`03_wmft_fm_validity.py` (validity, bootstraps, stratified ANOVA/Tukey),  
and the updated analysis scope described here replacing the older outline.
