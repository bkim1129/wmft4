import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.utils import resample

# ---------------------------------------------------------------------------
# Validity Test Code for WMFT-4, Including Non-Selected Items vs. WMFT-15
# ---------------------------------------------------------------------------

# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("Full_data_wmft.csv")  # Adjust path/name if necessary

# If your file uses columns named differently, modify here:
wmft_cols = [f"WMFT{i}" for i in range(1, 16)]  # e.g., WMFT1 ... WMFT15
fma_col = "FMA_UE"

# 2. WMFT-15 AVERAGE
# ---------------------------------------------------------------------------
df["WMFT15_avg"] = df[wmft_cols].mean(axis=1)

# 3. WMFT-4 ITEMS & AVERAGE
# ---------------------------------------------------------------------------
# Selected tasks: WMFT3, WMFT5, WMFT6, WMFT8
wmft4_cols = ["WMFT3", "WMFT5", "WMFT6", "WMFT8"]
df["WMFT4_avg"] = df[wmft4_cols].mean(axis=1)

# 4. NON-SELECTED ITEMS & AVERAGE
# ---------------------------------------------------------------------------
# Tasks *not* in WMFT-4
non_selected_cols = [
    "WMFT1", "WMFT2", "WMFT4", "WMFT7", "WMFT9",
    "WMFT10", "WMFT11", "WMFT12", "WMFT13", "WMFT14", "WMFT15"
]
df["WMFT_nonselected_avg"] = df[non_selected_cols].mean(axis=1)

# 5. PEARSON CORRELATIONS
# ---------------------------------------------------------------------------
# 5a. WMFT-4 vs. WMFT-15
r_conv, p_conv = pearsonr(df["WMFT4_avg"], df["WMFT15_avg"])

# 5b. WMFT-4 vs. FMA-UE
r_conc_4, p_conc_4 = pearsonr(df["WMFT4_avg"], df[fma_col])

# 5c. WMFT-15 vs. FMA-UE
r_conc_15, p_conc_15 = pearsonr(df["WMFT15_avg"], df[fma_col])

# 5d. Non-selected avg vs. WMFT-15
r_non, p_non = pearsonr(df["WMFT_nonselected_avg"], df["WMFT15_avg"])

print("=== Pearson Correlations (r, p-value) ===")
print(f"WMFT-4 vs. WMFT-15:     r = {r_conv:.3f}, p = {p_conv:.4f}")
print(f"WMFT-4 vs. {fma_col}:      r = {r_conc_4:.3f}, p = {p_conc_4:.4f}")
print(f"WMFT-15 vs. {fma_col}:     r = {r_conc_15:.3f}, p = {p_conc_15:.4f}")
print(f"Non-selected vs. WMFT-15: r = {r_non:.3f}, p = {p_non:.4f}")

# 6. BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------------------------
def bootstrap_correlation(data, col_x, col_y, n_boot=1000, ci=95):
    """
    Returns:
      mean_corr: average Pearson correlation across bootstrap samples
      (lower_bound, upper_bound): CI bounds based on percentiles
    """
    corrs = []
    size = len(data)
    alpha = (100 - ci) / 2.0  # For two-sided CI

    for _ in range(n_boot):
        sample = resample(data, replace=True, n_samples=size)
        r, _ = pearsonr(sample[col_x], sample[col_y])
        corrs.append(r)

    corrs = np.array(corrs)
    mean_corr = corrs.mean()
    lower_bound = np.percentile(corrs, alpha)
    upper_bound = np.percentile(corrs, 100 - alpha)
    return mean_corr, (lower_bound, upper_bound)

# Correlation pairs to examine:
pairs_to_test = [
    ("WMFT4_avg", "WMFT15_avg",  "WMFT-4 vs. WMFT-15"),
    ("WMFT4_avg", fma_col,       f"WMFT-4 vs. {fma_col}"),
    ("WMFT15_avg", fma_col,      f"WMFT-15 vs. {fma_col}"),
    ("WMFT_nonselected_avg", "WMFT15_avg", "Non-selected vs. WMFT-15")
]

print("\n=== Bootstrap Correlation Estimates (1,000 iterations) ===")
for xcol, ycol, label in pairs_to_test:
    mean_corr, (low_ci, high_ci) = bootstrap_correlation(df, xcol, ycol, n_boot=1000, ci=95)
    print(f"{label}: mean r = {mean_corr:.3f}, CI = [{low_ci:.3f}, {high_ci:.3f}]")
