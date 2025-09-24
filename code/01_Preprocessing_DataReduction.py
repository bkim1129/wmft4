import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import chi2, boxcox
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
file_path = 'wmft_rate_fm.csv'  # Replace this with the path to your file
wmft_data = pd.read_csv(file_path)

# Select features V1 to V15
selected_features = [f'V{i}' for i in range(1, 16)]
X = wmft_data[selected_features]

# Calculate the mean WMFT rate score (target variable)
y = X.mean(axis=1)

# Remove rows with missing values to ensure X and y have the same number of samples
data_complete = pd.concat([X, y], axis=1).dropna()
X = data_complete[selected_features]
y = data_complete.iloc[:, -1]  # Target variable


## Outlier detection and removal

def _boxcox_transform_frame(X: pd.DataFrame, eps: float = 1e-6):
    """
    Column-wise Box–Cox transform.
    Ensures strictly-positive inputs by shifting each column if needed.
    Returns: transformed DataFrame, dicts of per-column shifts and lambdas.
    """
    Z = X.copy()
    shifts, lambdas = {}, {}

    for c in Z.columns:
        x = Z[c].astype(float).values
        # require positivity
        minv = np.nanmin(x)
        shift = 0.0 if minv > 0 else (0.0 - minv + eps)
        x_pos = x + shift

        # guard against constant columns (Box–Cox undefined)
        if np.nanmax(x_pos) - np.nanmin(x_pos) <= np.finfo(float).eps:
            zt, lam = x_pos, np.nan
        else:
            # boxcox does not accept NaNs; drop/reinsert
            mask = ~np.isnan(x_pos)
            zt = np.empty_like(x_pos)
            zt[:] = np.nan
            zt[mask], lam = boxcox(x_pos[mask])
        Z[c] = zt
        shifts[c] = shift
        lambdas[c] = lam

    return Z, shifts, lambdas

def mvn_outliers(
    X: pd.DataFrame,
    method: str = "mcd",          # "mcd" (robust) or "classical"
    alpha: float = 0.001,         # cutoff tail prob for chi-square
    transform: str = "boxcox",  # "boxcox", None
    random_state: int = 42,
    make_plots: bool = True,
    fig_prefix: str = "mvn_"
):
    """
    MVN-style multivariate outlier detection using squared Mahalanobis distances.
    - transform="boxcox": per-feature Box–Cox with per-column λ and shifts.
    - method="mcd": robust MinCovDet; "classical": sample mean/cov.
    - cutoff = chi2.ppf(1-alpha, df=p).
    """
    # work on complete cases
    idx = X.dropna().index
    Xc = X.loc[idx].copy()

    # transform
    if transform == "boxcox":
        Z, shifts, lambdas = _boxcox_transform_frame(Xc)
    else:
        Z = Xc.values
        shifts, lambdas = {}, {}

    # compute squared MD
    if method.lower() == "classical":
        mu = np.nanmean(Z, axis=0)
        Sigma = np.cov(np.transpose(Z), ddof=1)
        invS = np.linalg.pinv(Sigma)
        diff = Z - mu
        md2 = np.einsum("ij,jk,ik->i", diff, invS, diff)
    else:  # robust
        mcd = MinCovDet(random_state=random_state).fit(Z)
        md2 = mcd.mahalanobis(Z)  # already squared MD

    p = Z.shape[1]
    cutoff = chi2.ppf(1 - alpha, df=p)
    flag = md2 > cutoff
    pvals = 1 - chi2.cdf(md2, df=p)

    table = pd.DataFrame(
        {"MD2": md2, "p_value": pvals, "flag_outlier": flag},
        index=idx
    )
    out_idx = table.index[flag]
    keep_idx = table.index[~flag]

    # optional diagnostics
    if make_plots:
        theo = chi2.ppf(np.linspace(0.5/len(md2), 1-0.5/len(md2), len(md2)), df=p)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(np.sort(theo), np.sort(md2), s=12)
        ax.axhline(cutoff, ls="--")
        ax.set_xlabel(f"Theoretical χ² quantiles (df={p})")
        ax.set_ylabel("Observed MD²")
        ax.set_title(f"MVN {method.upper()} (Box–Cox) chi-square plot")
        plt.tight_layout()
        plt.savefig(f"{fig_prefix}{method}_bcx_chi2plot.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(md2, bins=30, edgecolor="black")
        ax.axvline(cutoff, color="k", ls="--")
        ax.set_xlabel("MD²")
        ax.set_ylabel("Count")
        ax.set_title(f"MVN {method.upper()} (Box–Cox) MD² histogram")
        plt.tight_layout()
        plt.savefig(f"{fig_prefix}{method}_bcx_md2hist.png", dpi=300)
        plt.close(fig)

    # save transform parameters for reproducibility
    if transform == "boxcox":
        pd.DataFrame({"shift": shifts, "lambda": lambdas}).to_csv(f"{fig_prefix}boxcox_params.csv")

    return {
        "clean_index": keep_idx,
        "outlier_index": out_idx,
        "table": table,
        "cutoff": cutoff
    }

# ---- Example usage (replace your earlier MVN calls) ----
# X, y already defined after your complete-case step
mvn_mcd = mvn_outliers(X, method="mcd", alpha=0.001, transform="boxcox",
                       random_state=42, make_plots=True, fig_prefix="mvn_")
# optional sensitivity check
# mvn_cls = mvn_outliers(X, method="classical", alpha=0.001, transform="boxcox",
#                       make_plots=True, fig_prefix="mvn_")

# choose cleaned index (robust MCD recommended)
X = X.loc[mvn_mcd["clean_index"]].copy()
y = y.loc[mvn_mcd["clean_index"]].copy()

# ==== SAVE CLEANED DATA (all columns) ====
# Assumes you already ran mvn_mcd = mvn_outliers(...),
# and your original full dataframe is `wmft_data`.

# 1) rows to KEEP (robust MCD recommended)
clean_idx = mvn_mcd["clean_index"]

# 2) clean dataset with ALL original columns (no extra diagnostics)
clean_allcols = wmft_data.loc[wmft_data.index.intersection(clean_idx)].copy()

# 3) OPTIONAL: attach MVN diagnostics (MD², p, outlier flag) for transparency
mcd_diag = mvn_mcd["table"].rename(
    columns={"MD2": "MD2_mcd", "p_value": "p_mcd", "flag_outlier": "outlier_mcd"}
)
clean_with_diag = clean_allcols.join(mcd_diag, how="left")

# 4) SAVE
out_clean_path = "wmft_rate_fm_MVNclean.csv"
clean_allcols.to_csv(out_clean_path, index=False)
print(f"[SAVE] Clean (all original cols): {clean_allcols.shape} -> {out_clean_path}")

out_clean_diag_path = "wmft_rate_fm_MVNclean_with_diag.csv"
clean_with_diag.to_csv(out_clean_diag_path, index=False)
print(f"[SAVE] Clean + MVN diagnostics: {clean_with_diag.shape} -> {out_clean_diag_path}")

# 5) OPTIONAL: save flagged OUTLIERS for audit
outliers_idx = mvn_mcd["outlier_index"]
outliers_allcols = wmft_data.loc[wmft_data.index.intersection(outliers_idx)].copy()
outliers_with_diag = outliers_allcols.join(mcd_diag, how="left")

out_outliers_path = "wmft_rate_fm_MVNoutliers_with_diag.csv"
outliers_with_diag.to_csv(out_outliers_path, index=False)
print(f"[SAVE] Outliers + diagnostics: {outliers_with_diag.shape} -> {out_outliers_path}")

# ======================= end MVN-style OUTLIERS =======================

# Define the range of k values to test
k_values = list(range(1, 16))
performance_results = []

# Test different values of k
for k in k_values:
    # Perform SelectKBest to select the top k features
    select_k_best = SelectKBest(score_func=f_regression, k=k)
    X_selected = select_k_best.fit_transform(X, y)

    # Set up cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Train a Random Forest Regressor with the selected features
    rf = RandomForestRegressor(random_state=42)
    cross_val_scores = cross_val_score(rf, X_selected, y, cv=kf, scoring='r2')
    mean_r2 = np.mean(cross_val_scores)

    # Store the performance results for each k
    performance_results.append((k, mean_r2))
    print(f'k={k}, Mean R² from Cross-Validation: {mean_r2:.3f}')

# Convert the results to a DataFrame for better visualization
performance_df = pd.DataFrame(performance_results, columns=['k', 'Mean R²'])

plt.figure(figsize=(10, 6))
sns.lineplot(x='k', y='Mean R²', data=performance_df, marker='o')
plt.title('Performance of Random Forest with Different Number of Selected Features (k)')
plt.xlabel('Number of Selected Features (k)')
plt.ylabel('Mean R² Score')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()

performance_diff = np.diff(performance_df['Mean R²'])
elbow_k = np.argmax(performance_diff < 0.01) + 1  # +1 because np.diff reduces length by 1
print(f'\nOptimal k value based on the elbow method: {elbow_k}')

# Perform SelectKBest with the optimal k value
select_k_best_optimal = SelectKBest(score_func=f_regression, k=elbow_k)
X_selected_optimal = select_k_best_optimal.fit_transform(X, y)
selected_feature_names_optimal = X.columns[select_k_best_optimal.get_support()]

# Train and Evaluate Model with the Optimal Features
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rf_optimal = RandomForestRegressor(random_state=42)

# Perform cross-validation to evaluate the model with optimal features
cross_val_scores_optimal = cross_val_score(rf_optimal, X_selected_optimal, y, cv=kf, scoring='r2')
mean_r2_optimal = np.mean(cross_val_scores_optimal)

# Display the selected features and performance results
print(f'\nSelected Features with Optimal k={elbow_k}: {selected_feature_names_optimal.tolist()}')
print(f'Mean R² from Cross-Validation with Optimal Features: {mean_r2_optimal:.3f}')

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

# Full set of features
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestRegressor(random_state=42)

# Cross-validation on full feature set
cross_val_scores_full = cross_val_score(rf, X, y, cv=kf, scoring='r2')
mean_r2_full = np.mean(cross_val_scores_full)
print(f'Mean R² with Full Set of Features: {mean_r2_full:.3f}')

# Cross-validation on selected feature set (using optimal k from previous step)
cross_val_scores_selected = cross_val_score(rf, X_selected_optimal, y, cv=kf, scoring='r2')
mean_r2_selected = np.mean(cross_val_scores_selected)
print(f'Mean R² with Selected Features (k={elbow_k}): {mean_r2_selected:.3f}')

# Fit Random Forest using the selected features
rf.fit(X_selected_optimal, y)

# Extract feature importance
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_feature_names_optimal, 'Importance': importances})

# Sort and display feature importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# Plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

import scipy.stats as stats

# Correlation between each selected feature and the target variable
for feature in selected_feature_names_optimal:
    corr, p_value = stats.pearsonr(X[feature], y)
    print(f'Feature: {feature}, Correlation: {corr:.3f}, P-value: {p_value:.3f}')

from sklearn.utils import resample

n_iterations = 1000
feature_counts = pd.Series(0, index=selected_features)

for i in range(n_iterations):
    # Bootstrap resampling
    X_resample, y_resample = resample(X, y, random_state=i)

    # Run SelectKBest
    select_k_best = SelectKBest(score_func=f_regression, k=elbow_k)
    X_selected = select_k_best.fit_transform(X_resample, y_resample)
    selected_features_resample = X.columns[select_k_best.get_support()]

    # Count how often each feature is selected
    for feature in selected_features_resample:
        feature_counts[feature] += 1

# Display robustness of selected features
print(feature_counts.sort_values(ascending=False))

# --- Feature selection robustness plot (matplotlib only) ---
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1) INPUT: EITHER (A) use an existing pd.Series called `feature_counts` (index = 'V1'..'V15')
#           OR   (B) read from a CSV with columns: Feature, Count
USE_EXISTING_SERIES = 'feature_counts' in globals()

if USE_EXISTING_SERIES:
    df = feature_counts.rename_axis("Feature").reset_index(name="Count")
else:
    # Path to your counts file (edit as needed)
    counts_csv = Path("bootstrap_selection_counts.csv")
    # CSV should have two columns: Feature, Count (e.g., "V6", 1000)
    df = pd.read_csv(counts_csv)

# 2) Optional: map WMFT item codes to human-readable task names
wmft_labels = {
    "V1":  "Forearm to Table (Side)",
    "V2":  "Forearm to Box (Side)",
    "V3":  "Extend Elbow (Side)",
    "V4":  "Extend Elbow (Weight)",
    "V5":  "Hand to Table (Front)",
    "V6":  "Hand to Box (Front)",
    "V7":  "Reach and Retrieve",
    "V8":  "Lift Can",
    "V9":  "Lift Pencil",
    "V10": "Lift Paper Clip",
    "V11": "Stack Checkers",
    "V12": "Flip Cards",
    "V13": "Turn Key in Lock",
    "V14": "Fold Towel",
    "V15": "Lift Basket",
}

df["Task"] = df["Feature"].map(wmft_labels).fillna(df["Feature"])

# 3) Sort by frequency (descending) so the most stable items appear at the top
df = df.sort_values("Count", ascending=False)  # ascending so barh puts largest at top after invert_yaxis

# 4) Make the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Horizontal bars
bars = ax.barh(df["Feature"], df["Count"])

# Annotate each bar with the task label, inside the bar if it fits, else to the right
for rect, label, value in zip(bars, df["Task"], df["Count"]):
    x = rect.get_width()
    y = rect.get_y() + rect.get_height()/2
    # place text slightly inset from the left edge of the bar
    inset = min(10, max(3, 0.02 * ax.get_xlim()[1]))  # adaptive inset
    if x > 0.15 * ax.get_xlim()[1]:  # if bar long enough, put white text inside
        ax.text(inset, y, label, va="center", ha="left", color="white", fontsize=11)
    else:
        ax.text(x + 5, y, label, va="center", ha="left", color="black", fontsize=11)

# Cosmetics
ax.set_title("Feature Selection Robustness via Bootstrapping", fontsize=16)
ax.set_xlabel("Number of Times Selected (out of 1000)", fontsize=12)
ax.set_ylabel("WMFT Items", fontsize=12)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.invert_yaxis()  # largest on top
fig.tight_layout()

# 5) Save
fig.savefig("feature_selection_bootstrap.png", dpi=300)
print("Saved figure: feature_selection_bootstrap.png")
