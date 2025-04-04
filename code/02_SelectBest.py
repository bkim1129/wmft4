import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
file_path = 'data/wmft_rate_for_data_reduction.csv'  # Replace this with the path to your file
wmft_data = pd.read_csv(file_path)

# Select features V1 to V15
selected_features = [f'WMFT{i}' for i in range(1, 16)]
X = wmft_data[selected_features]

# Calculate the mean WMFT rate score (target variable)
y = X.mean(axis=1)

# Remove rows with missing values to ensure X and y have the same number of samples
data_complete = pd.concat([X, y], axis=1).dropna()
X = data_complete[selected_features]
y = data_complete.iloc[:, -1]  # Target variable

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
