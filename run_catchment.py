"""
Basin Flow Prediction & Interpretability (XGBoost + BayesSearchCV + SHAP)
-------------------------------------------------------------------
Description:
    This script demonstrates the full pipeline for a specified catchment (A-K):
    1) Load train/validation/test CSVs from the specific catchment folder.
    2) Hyperparameter tuning with BayesSearchCV using a PredefinedSplit.
    3) Final training and evaluation on the test set (using standard RMSE).
    4) SHAP-based interpretability: global importance, per-time-step (for the most important group),
       and intensity-stratified SHAP values.

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ------------------ Configuration ------------------
LAGS_MAPPING = {
    'A': 21, 'B': 21, 'C': 13, 'D': 13,
    'E': 21, 'F': 21, 'K': 21
}
NUM_GROUPS = 5  # Number of CML groups

def run_pipeline(catchment: str, base_dir: str = "."):
    """
    Main pipeline for training, evaluating, and interpreting the model.
    """
    # 1. Dynamic parameters
    if catchment not in LAGS_MAPPING:
        raise ValueError(f"Catchment '{catchment}' not supported. Choose from {list(LAGS_MAPPING.keys())}.")

    n_lags = LAGS_MAPPING[catchment]

    # Adapt to the directory structure: ./catchment_A/
    data_dir = os.path.join(base_dir, f"catchment_{catchment}")
    # Output path: ./outputs/catchment_A/
    out_dir = os.path.join(base_dir, "outputs", f"catchment_{catchment}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"--- Processing Catchment {catchment} (N_LAGS={n_lags}) ---")

    # 2. Load datasets
    train_file = os.path.join(data_dir, f"catchment_{catchment}_train.csv")
    vali_file  = os.path.join(data_dir, f"catchment_{catchment}_validation.csv")
    test_file  = os.path.join(data_dir, f"catchment_{catchment}_test.csv")

    try:
        train = pd.read_csv(train_file, index_col=0)
        vali  = pd.read_csv(vali_file, index_col=0)
        test  = pd.read_csv(test_file, index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}\nPlease ensure files are in {data_dir}.")
        return

    # 3. Split features and target
    # Assuming: first 5 columns are metadata, last 2 columns are target & bookkeeping
    train_X, train_y = train.iloc[:, 5:-2], train.iloc[:, -2]
    val_X,   val_y   = vali.iloc[:, 5:-2], vali.iloc[:, -2]
    test_X,  test_y  = test.iloc[:, 5:-2], test.iloc[:, -2]

    # 4. Prepare for BayesSearchCV
    X_all = pd.concat([train_X, val_X])
    y_all = pd.concat([train_y, val_y])

    test_fold = [-1] * len(train_X) + [0] * len(val_X)
    ps = PredefinedSplit(test_fold)

    param_space = {
        'learning_rate': Real(0.005, 0.1),
        'n_estimators': Integer(100, 600),
        'max_depth': Integer(2, 6),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(1e-4, 0.5, prior='log-uniform'),
        'reg_lambda': Real(5, 10, prior='log-uniform'),
        'subsample': Real(0.7, 1.0),
        'min_child_weight': Integer(1, 10)
    }

    print("Starting hyperparameter tuning (This may take a while)...")
    opt = BayesSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
        search_spaces=param_space,
        n_iter=50,
        scoring='neg_root_mean_squared_error',
        cv=ps,
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    opt.fit(X_all, y_all)
    print("Best parameters:", opt.best_params_)

    # 5. Test set evaluation
    final_model = opt.best_estimator_
    y_pred_test = final_model.predict(test_X)

    r2 = r2_score(test_y, y_pred_test)
    rmse = sqrt(mean_squared_error(test_y, y_pred_test))
    re = (y_pred_test.sum() - test_y.sum()) / test_y.sum()
    pcc = np.corrcoef(test_y, y_pred_test)[0, 1]

    print(f"Test Metrics: R²={r2:.3f}, PCC={pcc:.3f}, RMSE={rmse:.3f}, RE={re:.3f}")

    df_pred = pd.DataFrame({
        'Observed': test_y,
        'Model': y_pred_test,
        'Event': test['event'].values
    })
    df_pred.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

    # 6. SHAP Analysis
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(test_X)
    feature_importance = np.abs(shap_values.values).mean(axis=0)

    n_total_features = shap_values.values.shape[1]
    n_extra_features = n_total_features - (NUM_GROUPS * n_lags)

    feature_labels = [f'Group {i+1}' for i in range(NUM_GROUPS)]
    if n_extra_features > 0:
        feature_labels += [f'Extra_{i+1}' for i in range(n_extra_features)]

    # ---- Global importance by groups ----
    In = np.zeros(len(feature_labels))
    for i in range(NUM_GROUPS):
        In[i] = np.sum(feature_importance[i * n_lags : (i + 1) * n_lags])
    if n_extra_features > 0:
        In[NUM_GROUPS:] = feature_importance[-n_extra_features:]

    sorted_idx = np.argsort(In)[::-1]
    sorted_importance = In[sorted_idx]
    sorted_features = [feature_labels[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[::-1], sorted_importance[::-1], color="royalblue")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature Group")
    plt.title(f"Feature Importance based on SHAP Values (Catchment {catchment})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_groups.png"), dpi=300)
    plt.close()

    # ---- Merge SHAP values for each group (per sample) ----
    merged_shap_values = np.zeros((shap_values.values.shape[0], NUM_GROUPS))
    for i in range(NUM_GROUPS):
        merged_shap_values[:, i] = np.sum(shap_values.values[:, i * n_lags : (i + 1) * n_lags], axis=1)

    merged_shap_values_obj = shap.Explanation(
        values=merged_shap_values,
        base_values=shap_values.base_values,
        feature_names=[f'Group {i+1}' for i in range(NUM_GROUPS)]
    )

    shap.summary_plot(merged_shap_values_obj, test_X.iloc[:, :NUM_GROUPS], show=False)
    plt.savefig(os.path.join(out_dir, "shap_summary_groups.png"), dpi=300)
    plt.close()

    # ---- Time-step importance within the MOST IMPORTANT group ----
    most_important_group_idx = sorted_idx[0]
    most_important_group_label = feature_labels[most_important_group_idx]

    print(f"Analyzing temporal importance for the most important group: {most_important_group_label}")

    group_shap_values = shap_values.values[:, most_important_group_idx * n_lags : (most_important_group_idx + 1) * n_lags]
    time_shap_importance = np.abs(group_shap_values).mean(axis=0)
    time_labels = [f"T-{i}" for i in range(n_lags)]

    plt.figure(figsize=(10, 6))
    plt.plot(time_labels, time_shap_importance, marker='o', linestyle='-', color="royalblue")
    plt.xlabel("Time Steps")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.title(f"Temporal Importance ({most_important_group_label}, Catchment {catchment})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_timecurve_{most_important_group_label.replace(' ', '')}.png"), dpi=300)
    plt.close()

    # ---- SHAP importance under different rainfall intensities ----
    threshold_small = test_y.quantile(0.33)
    threshold_big = test_y.quantile(0.67)

    idx_small = test_y <= threshold_small
    idx_moderate = (test_y > threshold_small) & (test_y <= threshold_big)
    idx_big = test_y > threshold_big

    def compute_stratified_importance(shap_matrix):
        imp = np.abs(shap_matrix).mean(axis=0)
        group_imp = np.zeros(len(feature_labels))
        for i in range(NUM_GROUPS):
            group_imp[i] = np.sum(imp[i * n_lags : (i + 1) * n_lags])
        if n_extra_features > 0:
            group_imp[NUM_GROUPS:] = imp[-n_extra_features:]
        return group_imp

    imp_small = compute_stratified_importance(shap_values.values[idx_small, :])
    imp_moderate = compute_stratified_importance(shap_values.values[idx_moderate, :])
    imp_big = compute_stratified_importance(shap_values.values[idx_big, :])

    def sort_by_importance(importance, labels):
        s_idx = np.argsort(importance)[::-1]
        return importance[s_idx], [labels[i] for i in s_idx]

    s_imp_small, s_feat_small = sort_by_importance(imp_small, feature_labels)
    s_imp_moderate, s_feat_moderate = sort_by_importance(imp_moderate, feature_labels)
    s_imp_big, s_feat_big = sort_by_importance(imp_big, feature_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].barh(s_feat_small[::-1], s_imp_small[::-1], color="royalblue")
    axes[0].set_title("Small Flow/Precipitation")
    axes[0].set_xlabel("Mean |SHAP|")

    axes[1].barh(s_feat_moderate[::-1], s_imp_moderate[::-1], color="orange")
    axes[1].set_title("Moderate Flow/Precipitation")
    axes[1].set_xlabel("Mean |SHAP|")

    axes[2].barh(s_feat_big[::-1], s_imp_big[::-1], color="lightcoral")
    axes[2].set_title("Heavy Flow/Precipitation")
    axes[2].set_xlabel("Mean |SHAP|")

    plt.suptitle(f"SHAP Importance by Intensity (Catchment {catchment})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_by_rain_intensity.png"), dpi=300)
    plt.close()

    print(f"Pipeline completed for Catchment {catchment}. Results saved to {out_dir}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline for Catchment Flow Prediction.")
    parser.add_argument("--catchment", type=str, default="A", choices=list(LAGS_MAPPING.keys()),
                        help="Select catchment basin (A, B, C, D, E, F, K)")

    args = parser.parse_args()
    run_pipeline(catchment=args.catchment, base_dir=".")
