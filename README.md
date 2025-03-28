# README

Welcome to the repository for the manuscript **“The Development and Validity of a Shortened Version of the Wolf Motor Function Test (WMFT-4)”**. This repository contains the data and analysis code used in the study, which introduces a streamlined version of the Wolf Motor Function Test (WMFT) for assessing upper extremity motor function in stroke rehabilitation research.

---

## Overview

### Purpose
- Demonstrate how the WMFT-4 was derived from the original 15-item WMFT using feature selection and machine learning approaches.
- Provide reproducible analysis scripts that outline data preprocessing, factor analysis (EFA/CFA), and psychometric testing (MDC, MCID, validity correlations).

### Key Findings
1. **Factor Analysis**: Confirmed that WMFT time scores reflect two distinct factors (proximal control vs. integrative functional tasks).
2. **Item Reduction**: Identified four representative WMFT tasks – **Extend Elbow (side) Without Weight, Hand to Table (front), Hand to Box (front), and Lift Can** – through a data-driven process using SelectKBest and Random Forest Regressors.
3. **Psychometric Results**: WMFT-4 strongly correlates with the original WMFT-15 and retains comparable validity and reliability.

---

## Repository Structure

```plaintext
├── Factor Analysis Data
│   ├── wmft_all.csv
│   ├── trial_group.csv
│   ├── chronicity_all.csv
│   ├── FM scores.csv
│   └── ...
├── Feature Selection and Data Reduction Analysis
│   ├── wmft_rate_for_data_reduction.csv
│   └── ...
├── WMFT-4 Psychometrics Analaysis Data
│   ├── WMFT_pre_new.csv
│   ├── WMFT_post_new.csv
│   ├── FM_new.csv
│   ├── FM_post_new.csv
│   └── ...
├── code
│   ├── 01_data_preprocessing_and_factor_analysis.R
│   ├── 02_SelectBest.py
│   ├── 03_wmft_fm_validity.py
│   ├── 04_WMFT4_MDC_MCID.R
│   └── ...
├── figures
│   ├── EFA_results.png
│   ├── CFA_results.png
│   └── ...
└── README.md
```

1. **data/**  
   - Contains anonymized datasets used in the manuscript.  
   - Each file includes participant IDs (coded) and item-level WMFT data for baseline and (if available) post-intervention.
2. **code/**  
   - **R Scripts**: Preprocessing, factor analysis (EFA/CFA), psychometric property calculations.  
   - **Python Scripts**: Feature selection (SelectKBest), Random Forest training for item reduction.  
3. **figures/**  
   - Stores outputs and plots (e.g., Scree plots, CFA diagrams, bootstrap frequency charts).

---

## Getting Started

### Prerequisites

- **R (>= 4.2.1)** and RStudio  
  - Packages: `tidyverse`, `MASS`, `psych`, `nFactors`, `lavaan`, `REdaS`  
- **Python (>= 3.8)**
  - Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/WMFT4-factor-analysis.git
   ```
2. Install required R and Python packages:
   ```bash
   # In R
   install.packages(c("tidyverse", "MASS", "psych", "nFactors", "lavaan", "REdaS"))

   # In Python
   pip install numpy pandas scikit-learn matplotlib
   ```

---

## Data and Analysis Workflow

1. **Data Preprocessing & Factor Analysis (01_data_preprocessing_and_factor_analysis.R)**
   - Converts WMFT time scores to rates (60 / time).
   - Applies Box-Cox transformations to address skewness.
   - Performs multivariate outlier detection using robust Mahalanobis distances.
   - Outputs a cleaned dataset for subsequent factor analyses.
   - **EFA**: Determines underlying factor structure (proximal vs. integrative control) and retains two factors using parallel analysis.
   - **CFA**: Compares one-factor vs. two-factor models; evaluates model fit indices (RMSEA, CFI, TLI).

2. **Feature Selection & Item Reduction (02_SelectBest.py)**
   - Uses `SelectKBest` and Random Forest Regressors to identify the number of items that best predict the total WMFT-15 score.
   - Employs elbow method to locate the optimal item subset.
   - Confirms item stability through 1,000 bootstrap iterations.

3. **Psychometric Testing - Validity (03_wmft_fm_validity.py)**
   - Computes **convergent** validity (correlation WMFT-4 vs. WMFT-15).
   - Assesses **concurrent** validity (WMFT-4 vs. Fugl-Meyer Assessment of Upper Extremity).
     
4. **Psychometric Testing - MDC & MCID (03_WMFT4_MDC_MCID.R)**
   - Calculates **MDC (Minimal Detectable Change)** and **MCID (Minimal Clinically Important Difference)** for both WMFT-4 and WMFT-15.

---

## How to Reproduce the Results

1. **Open RStudio** and run:
   ```r
   source("code/01_data_preprocessing_and_factor_analysis.R")
   ```
   - Outputs EFA/CFA results, model fit statistics, factor loading tables.

2. **Run Python script** for item reduction:
   ```bash
   python code/02_SelectBest.py
   ```
   - Generates the four-item solution (WMFT-4) and an elbow plot to confirm the item count.

3. **Finalize psychometric measures**:
   ```bash
   python code/03_wmft_fm_validity.py
   ```
   ```r
   source("code/04_WMFT4_MDC_MCID.R")
   ```
   - Outputs correlation values for validity and calculates MDC/MCID.

Each script is documented internally with comments explaining all relevant steps and parameters.

---

## References

- Wolf SL et al. Assessing Wolf Motor Function Test as Outcome Measure for Research in Patients After Stroke. *Stroke.* 2001;32(7):1635–1639.  
  [Link](https://www.ahajournals.org/doi/10.1161/01.STR.32.7.1635)

- Woodbury M et al. Measurement structure of the Wolf Motor Function Test: implications for motor control theory. *Neurorehabil Neural Repair.* 2010;24(9):791–801.  
  [Link](https://journals.sagepub.com/doi/10.1177/1545968310370749)

- Bogard K et al. Can the Wolf Motor Function Test be streamlined? *Neurorehabil Neural Repair.* 2009;23(5):422–8.  
  [Link](https://journals.sagepub.com/doi/10.1177/1545968308331141)

- Academy of Neurologic Physical Therapy. StrokeEDGE II Documents.  
  [Link](https://www.neuropt.org/practice-resources/neurology-section-outcome-measures-recommendations/stroke)

---

## Citation

If you use or adapt these materials, please cite:

**Kim B, Schweighofer N, Wolf S, Winstein C.** *The Development and Validity of a Shortened Version of the Wolf Motor Function Test (WMFT-4).* Manuscript in preparation.  

---

## License

Unless otherwise indicated, all code in this repository is released under the [MIT License](LICENSE).  
**Important**: The data are de-identified and intended for research and educational purposes. Any further use of these data must comply with relevant institutional review board (IRB) and data use agreements.

---

## Contact

For questions or feedback, please contact:  
**Bokkyu Kim**  
**SUNY Upstate Medical University**  
Email: kimbo@upstate.edu

Thank you for your interest in the WMFT-4 project! We appreciate contributions and feedback to improve the usability and clarity of this repository.

