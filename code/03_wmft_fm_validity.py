import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.utils import resample
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from ace_tools_open import display_dataframe_to_user


# ---------------------------------------------------------------------------
# Validity Test Code for WMFT-4, Including Non-Selected Items vs. WMFT-15
# ---------------------------------------------------------------------------

# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("wmft_rate_fm_MVNclean.csv")  # Adjust path/name if necessary

# If your file uses columns named differently, modify here:
wmft_cols = [f"V{i}" for i in range(1, 15)]  # e.g., WMFT1 ... WMFT15
fma_col = "FM"

# 2. WMFT-15 AVERAGE
# ---------------------------------------------------------------------------
df["WMFT15_avg"] = df[wmft_cols].mean(axis=1)

# 3. WMFT-4 ITEMS & AVERAGE
# ---------------------------------------------------------------------------
# Selected tasks: WMFT3, WMFT5, WMFT6, WMFT8
wmft4_cols = ["V3", "V5", "V6", "V8"]
df["WMFT4_avg"] = df[wmft4_cols].mean(axis=1)

# 4. NON-SELECTED ITEMS & AVERAGE
# ---------------------------------------------------------------------------
# Tasks *not* in WMFT-4
non_selected_cols = [
    "V1", "V2", "V4", "V7", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15"
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

# 5e. Non-selected avg vs. FMA-UE
r_non_FMA, p_non_FMA = pearsonr(df["WMFT_nonselected_avg"], df[fma_col])

print("=== Pearson Correlations (r, p-value) ===")
print(f"WMFT-4 vs. WMFT-15:     r = {r_conv:.3f}, p = {p_conv:.4f}")
print(f"WMFT-4 vs. {fma_col}:      r = {r_conc_4:.3f}, p = {p_conc_4:.4f}")
print(f"WMFT-15 vs. {fma_col}:     r = {r_conc_15:.3f}, p = {p_conc_15:.4f}")
print(f"Non-selected vs. WMFT-15: r = {r_non:.3f}, p = {p_non:.4f}")
print(f"Non-selected vs. FMA-UE: r = {r_non_FMA:.3f}, p = {p_non_FMA:.4f}")

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


##### Known value cross-domain validity

# Identify columns (expect V1..V15 and FM)
wmft_cols = [c for c in df.columns if c.strip().upper().startswith("V") and c.strip()[1:].isdigit()]
wmft_cols = sorted(wmft_cols, key=lambda x: int(''.join(filter(str.isdigit, x))))
if len(wmft_cols) >= 15:
    wmft_cols = wmft_cols[:15]  # ensure V1..V15 order
else:
    # fallback to explicit names if needed
    wmft_cols = [f"V{i}" for i in range(1,16) if f"V{i}" in df.columns]

# Ensure numeric for WMFT and FM
for c in wmft_cols + ["FM"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Heuristic: if any median > 120, treat as time and convert to rate; else assume already rate
meds = [df[c].median(skipna=True) for c in wmft_cols]
convert_from_time = any(m > 120 for m in meds if pd.notna(m))

if convert_from_time:
    for c in wmft_cols:
        df.loc[df[c] <= 0, c] = np.nan
        df[c] = df[c].clip(upper=120)
        df[c] = 60.0 / df[c]

# WMFT-4 items (3,5,6,8)
wmft4_items = [f"V{i}" for i in [3,5,6,8] if f"V{i}" in df.columns]

# Composites (mean rates)
df["WMFT4_mean_rate"] = df[wmft4_items].mean(axis=1, skipna=True)
df["WMFT15_mean_rate"] = df[wmft_cols].mean(axis=1, skipna=True)

# --- Stratification: manuscript 3-group cutoffs (0–28, 29–42, 43–66) ---
def fm3_group_custom(x):
    if pd.isna(x): return None
    x = float(x)
    if x <= 28:
        return "Severe (0–28)"
    elif x <= 42:
        return "Moderate (29–42)"
    else:
        return "Mild (43–66)"

df["FMA_3grp_custom"] = df["FM"].apply(fm3_group_custom)

# Group summaries with 95% CI
def group_summary(var, grp_col="FMA_3grp_custom"):
    order = ["Severe (0–28)", "Moderate (29–42)", "Mild (43–66)"]
    g = df.groupby(grp_col)[var].agg(["count","mean","std"]).reindex(order)
    ci_rows = []
    for grp in order:
        x = df.loc[df[grp_col]==grp, var].dropna().values
        n = x.size
        if n>1:
            m = x.mean()
            se = x.std(ddof=1)/np.sqrt(n)
            lo, hi = stats.t.interval(0.95, n-1, loc=m, scale=se)
        else:
            lo, hi = (np.nan, np.nan)
        ci_rows.append((grp, lo, hi))
    ci_df = pd.DataFrame(ci_rows, columns=["grp","CI_lower","CI_upper"]).set_index("grp")
    return pd.concat([g, ci_df], axis=1).round(2)

summary_w4 = group_summary("WMFT4_mean_rate")
summary_w15 = group_summary("WMFT15_mean_rate")



display_dataframe_to_user("WMFT-4 mean rate (reps/min) by 3-group FMA-UE strata (0–28, 29–42, 43–66)", summary_w4)
display_dataframe_to_user("WMFT-15 mean rate (reps/min) by 3-group FMA-UE strata (0–28, 29–42, 43–66)", summary_w15)

# ANOVA + eta^2
def oneway_anova(var, grp_col="FMA_3grp_custom"):
    valid = df[[grp_col, var]].dropna()
    order = ["Severe (0–28)", "Moderate (29–42)", "Mild (43–66)"]
    groups = [valid.loc[valid[grp_col]==g, var].values for g in order if g in valid[grp_col].unique()]
    f, p = stats.f_oneway(*groups)
    overall = valid[var].mean()
    ss_between = valid.groupby(grp_col)[var].apply(lambda x: len(x)*(x.mean()-overall)**2).sum()
    ss_total = ((valid[var]-overall)**2).sum()
    eta2 = ss_between/ss_total if ss_total>0 else np.nan
    return f, p, eta2, len(valid)

f4, p4, eta4, n4 = oneway_anova("WMFT4_mean_rate")
f15, p15, eta15, n15 = oneway_anova("WMFT15_mean_rate")

anova_df = pd.DataFrame({
    "Measure": ["WMFT-4 mean rate (reps/min)", "WMFT-15 mean rate (reps/min)"],
    "N (complete)": [n4, n15],
    "F": [f4, f15],
    "p": [p4, p15],
    "eta_squared": [eta4, eta15]
}).round(4)

display_dataframe_to_user("One-way ANOVA across FMA-UE strata (custom 3-group)", anova_df)

# Tukey HSD post-hoc
valid4 = df[["FMA_3grp_custom","WMFT4_mean_rate"]].dropna()
valid15 = df[["FMA_3grp_custom","WMFT15_mean_rate"]].dropna()

tuk4 = pairwise_tukeyhsd(valid4["WMFT4_mean_rate"], valid4["FMA_3grp_custom"], alpha=0.05)
tuk15 = pairwise_tukeyhsd(valid15["WMFT15_mean_rate"], valid15["FMA_3grp_custom"], alpha=0.05)

tuk4_df = pd.DataFrame(data=tuk4._results_table.data[1:], columns=tuk4._results_table.data[0])
tuk15_df = pd.DataFrame(data=tuk15._results_table.data[1:], columns=tuk15._results_table.data[0])

display_dataframe_to_user("Tukey HSD for WMFT-4 mean rate (custom 3-group)", tuk4_df)
display_dataframe_to_user("Tukey HSD for WMFT-15 mean rate (custom 3-group)", tuk15_df)

# Figures
order = ["Severe (0–28)", "Moderate (29–42)", "Mild (43–66)"]

plt.figure()
plt.boxplot([df.loc[df["FMA_3grp_custom"]==g, "WMFT4_mean_rate"].dropna() for g in order], labels=order)
plt.ylabel("WMFT-4 mean rate (reps/min)")
plt.title("WMFT-4 mean rate by FMA-UE strata (0–28, 29–42, 43–66)")
plt.tight_layout()
plt.savefig("wmft4_rate_by_3groups_CUSTOM.png", dpi=300)
plt.show()

plt.figure()
plt.boxplot([df.loc[df["FMA_3grp_custom"]==g, "WMFT15_mean_rate"].dropna() for g in order], labels=order)
plt.ylabel("WMFT-15 mean rate (reps/min)")
plt.title("WMFT-15 mean rate by FMA-UE strata (0–28, 29–42, 43–66)")
plt.tight_layout()
plt.savefig("wmft15_rate_by_3groups_CUSTOM.png", dpi=300)
plt.show()

# Save subject-level outputs
out = df[["FM","FMA_3grp_custom","WMFT4_mean_rate","WMFT15_mean_rate"]].copy()
out_path = "wmft_mean_rate_by_subject_3groups_CUSTOM.csv"
out.to_csv(out_path, index=False)


# Print concise numeric results for the manuscript

# Detect WMFT columns and compute composites
wmft_cols = [c for c in df.columns if c.strip().upper().startswith("V") and c.strip()[1:].isdigit()]
wmft_cols = sorted(wmft_cols, key=lambda x: int(''.join(filter(str.isdigit, x))))
for c in wmft_cols + ["FM"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
wmft4_items = [f"V{i}" for i in [3,5,6,8] if f"V{i}" in df.columns]
df["WMFT4_mean_rate"] = df[wmft4_items].mean(axis=1, skipna=True)
df["WMFT15_mean_rate"] = df[wmft_cols].mean(axis=1, skipna=True)

def fm3_group_custom(x):
    if pd.isna(x): return None
    x = float(x)
    if x <= 28: return "Severe (0–28)"
    elif x <= 42: return "Moderate (29–42)"
    else: return "Mild (43–66)"
df["FMA_3grp_custom"] = df["FM"].apply(fm3_group_custom)

order = ["Severe (0–28)", "Moderate (29–42)", "Mild (43–66)"]
summ4 = df.groupby("FMA_3grp_custom")["WMFT4_mean_rate"].agg(["count","mean","std"]).reindex(order).round(2)
summ15 = df.groupby("FMA_3grp_custom")["WMFT15_mean_rate"].agg(["count","mean","std"]).reindex(order).round(2)

def oneway(var):
    valid = df[["FMA_3grp_custom", var]].dropna()
    groups = [valid.loc[valid["FMA_3grp_custom"]==g, var].values for g in order]
    f, p = stats.f_oneway(*groups)
    overall = valid[var].mean()
    ss_between = valid.groupby("FMA_3grp_custom")[var].apply(lambda x: len(x)*(x.mean()-overall)**2).sum()
    ss_total = ((valid[var]-overall)**2).sum()
    eta2 = ss_between/ss_total if ss_total>0 else np.nan
    return f, p, eta2, len(valid)

f4, p4, eta4, n4 = oneway("WMFT4_mean_rate")
f15, p15, eta15, n15 = oneway("WMFT15_mean_rate")

# Tukey
valid4 = df[["FMA_3grp_custom","WMFT4_mean_rate"]].dropna()
valid15 = df[["FMA_3grp_custom","WMFT15_mean_rate"]].dropna()

tuk4 = pairwise_tukeyhsd(valid4["WMFT4_mean_rate"], valid4["FMA_3grp_custom"], alpha=0.05)
tuk15 = pairwise_tukeyhsd(valid15["WMFT15_mean_rate"], valid15["FMA_3grp_custom"], alpha=0.05)

t4 = pd.DataFrame(tuk4.summary().data[1:], columns=tuk4.summary().data[0])
t15 = pd.DataFrame(tuk15.summary().data[1:], columns=tuk15.summary().data[0])

print("WMFT-4 mean rate by group:\n", summ4.to_string())
print("\nWMFT-15 mean rate by group:\n", summ15.to_string())
print(f"\nANOVA WMFT-4: F({2},{n4-3})={f4:.2f}, p={p4:.3e}, eta^2={eta4:.3f}, N={n4}")
print(f"ANOVA WMFT-15: F({2},{n15-3})={f15:.2f}, p={p15:.3e}, eta^2={eta15:.3f}, N={n15}")
print("\nTukey WMFT-4:\n", t4.to_string(index=False))
print("\nTukey WMFT-15:\n", t15.to_string(index=False))


