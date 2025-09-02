"""
Basin A example (XGBoost + BayesSearchCV + SHAP)
-------------------------------------------------------------------
This script demonstrates the full pipeline for Basin A:
1) load train/validation/test CSVs,
2) hyperparameter tuning with BayesSearchCV using a PredefinedSplit (train vs val),
3) final training on train+val, evaluation on test,
4) SHAP-based interpretability (global importance by CML groups; per-time-step within one group;
   intensity-stratified SHAP for small/moderate/heavy flows).

Assumptions about the CSVs:
- Three files in ./data/:
  Training_dataset_A.csv, Validation_dataset_A.csv, Testing_dataset_A.csv
- Each file contains:
    - a target column (name can be configured; fallback: the penultimate column)
    - an 'event' column (used only for saving predictions grouped by event)
    - feature columns (all other columns except event & target & any trailing bookkeeping column)
- For the SHAP aggregation by CML group:
    - features are ordered as [Group1_21steps, Group2_21steps, ..., Group5_21steps, (optional extra features...)]
    - Set STEPS_PER_GROUP=21 and NUM_GROUPS=5 for Basin A.

Author: Ying Song
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from math import sqrt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import PredefinedSplit
import shap
import matplotlib.pyplot as plt


# ------------------ Input & Output paths ------------------
data_dir = "./data"        # folder containing datasets
out_dir = "./outputs_A"    # folder for saving results
os.makedirs(out_dir, exist_ok=True)

# ------------------ Load datasets ------------------
train = pd.read_csv(os.path.join(data_dir, "Training_dataset_A.csv"), index_col=0)
vali  = pd.read_csv(os.path.join(data_dir, "Validation_dataset_A.csv"), index_col=0)
test  = pd.read_csv(os.path.join(data_dir, "Testing_dataset_A.csv"), index_col=0)

# ------------------ Split features and target ------------------
train_X, train_y = train.iloc[:, 5:-2], train.iloc[:, -2]
val_X,   val_y   = vali.iloc[:, 5:-2], vali.iloc[:, -2]
test_X,  test_y  = test.iloc[:, 5:-2], test.iloc[:, -2]

# ------------------ Prepare for BayesSearchCV ------------------
X_all = pd.concat([train_X, val_X])
y_all = pd.concat([train_y, val_y])

# PredefinedSplit: training = -1, validation = 0
test_fold = [-1] * len(train_X) + [0] * len(val_X)
ps = PredefinedSplit(test_fold)

# Define hyperparameter search space
param_space = {
    'learning_rate': Real(0.05, 0.1),
    'n_estimators': Integer(300, 1000),
    'max_depth': Integer(3, 10),
    'colsample_bytree': Real(0.5, 1.0)
}

# Run BayesSearchCV
opt = BayesSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
    search_spaces=param_space,
    n_iter=10,
    scoring='neg_root_mean_squared_error',
    cv=ps,
    n_jobs=-1,
    verbose=0,
    random_state=42
)

opt.fit(X_all, y_all)
print("Best parameters:", opt.best_params_)

# ------------------ Test set evaluation ------------------
final_model = opt.best_estimator_
y_pred_test = final_model.predict(test_X)

r2 = r2_score(test_y, y_pred_test)
mare = np.mean(np.abs(test_y - y_pred_test) / (np.abs(test_y) + 1e-6))   # mean absolute relative error
re = (y_pred_test.sum() - test_y.sum()) / test_y.sum()
pcc = np.corrcoef(test_y, y_pred_test)[0, 1]

print(f"R²={r2:.3f}, PCC={pcc:.3f}, MARE={mare:.3f}, RE={re:.3f}")

# Save predictions
df_pred = pd.DataFrame({
    'Observed': test_y,
    'Model': y_pred_test,
    'Event': test['event'].values
})
df_pred.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)
# ------------------ SHAP analysis ------------------
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(test_X)

feature_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
feature_importance = np.abs(shap_values.values).mean(axis=0)

num_CML = 5
In = np.zeros(num_CML)
for i in range(num_CML):
    In[i] = np.sum(feature_importance[i*21:(i+1)*21])

# ---- Global importance by groups ----
sorted_idx = np.argsort(In)[::-1]
sorted_importance = In[sorted_idx]
sorted_features = [feature_labels[i] for i in sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features[::-1], sorted_importance[::-1], color="royalblue")
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Feature Importance based on SHAP Values")
plt.savefig(os.path.join(out_dir, "shap_groups.png"), dpi=300)
plt.close()

# ---- Merge SHAP values for each group (per sample) ----
merged_shap_values = np.zeros((shap_values.values.shape[0], len(feature_labels)))
for i in range(num_CML):
    merged_shap_values[:, i] = np.sum(shap_values.values[:, i*21:(i+1)*21], axis=1)

merged_shap_values_obj = shap.Explanation(
    values=merged_shap_values,
    base_values=shap_values.base_values,
    feature_names=feature_labels
)

# Summary plot by groups
shap.summary_plot(merged_shap_values_obj, test_X.iloc[:, :len(feature_labels)], show=False)
plt.savefig(os.path.join(out_dir, "shap_summary_groups.png"), dpi=300)
plt.close()

# ---- Time-step importance within one group (Group 1) ----
group_index = 0
group_shap_values = shap_values.values[:, group_index*21:(group_index+1)*21]
time_shap_importance = np.abs(group_shap_values).mean(axis=0)
time_labels = [f"T-{i}" for i in range(21)]

plt.figure(figsize=(10, 6))
plt.plot(time_labels, time_shap_importance, marker='o', linestyle='-', color="royalblue")
plt.xlabel("Time Steps")
plt.ylabel("Mean Absolute SHAP Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(os.path.join(out_dir, "shap_timecurve_group1.png"), dpi=300)
plt.close()

# ---- SHAP importance under different rainfall intensities ----
In = np.append(In, feature_importance[-2:])  # add sin/cos if present

threshold = test_y.quantile(0.33)
threshold1 = test_y.quantile(0.67)
small_rain_idx = test_y <= threshold
moderate_rain_idx = (test_y > threshold) & (test_y <= threshold1)
big_rain_idx = test_y > threshold1

shap_small = shap_values.values[small_rain_idx, :]
shap_moderate = shap_values.values[moderate_rain_idx, :]
shap_big = shap_values.values[big_rain_idx, :]

def compute_shap_importance(shap_matrix):
    feature_importance = np.abs(shap_matrix).mean(axis=0)
    In = np.zeros((num_CML, 1))
    for i in range(num_CML):
        start_idx = i * 13  # note: change 13 if using different time steps
        end_idx = (i + 1) * 13
        In[i] = np.sum(feature_importance[start_idx:end_idx])
    In = np.append(In, [feature_importance[-2], feature_importance[-1]])  # add sin and cos
    return In

importance_small = compute_shap_importance(shap_small)
importance_moderate = compute_shap_importance(shap_moderate)
importance_big = compute_shap_importance(shap_big)

def sort_by_importance(importance, feature_labels):
    sorted_idx = np.argsort(importance.flatten())[::-1]
    return importance[sorted_idx], [feature_labels[i] for i in sorted_idx]

sorted_importance_small, sorted_features_small = sort_by_importance(importance_small, feature_labels)
sorted_importance_moderate, sorted_features_moderate = sort_by_importance(importance_moderate, feature_labels)
sorted_importance_big, sorted_features_big = sort_by_importance(importance_big, feature_labels)

# Plot by rainfall intensity
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].barh(sorted_features_small[::-1], sorted_importance_small[::-1], color="royalblue")
axes[0].set_title("Feature Importance\n(Small Rain)")
axes[0].set_xlabel("Mean |SHAP|")

axes[1].barh(sorted_features_moderate[::-1], sorted_importance_moderate[::-1], color="orange")
axes[1].set_title("Feature Importance\n(Moderate Rain)")
axes[1].set_xlabel("Mean |SHAP|")

axes[2].barh(sorted_features_big[::-1], sorted_importance_big[::-1], color="lightcoral")
axes[2].set_title("Feature Importance\n(Heavy Rain)")
axes[2].set_xlabel("Mean |SHAP|")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_by_rain_intensity.png"), dpi=300)
plt.close()






