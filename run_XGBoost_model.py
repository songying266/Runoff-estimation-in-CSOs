"""
Basin Flow Prediction & Interpretability (XGBoost + SHAP)
-------------------------------------------------------------------
Description:
    This script runs the full pipeline in a flat structure (for-loop).
    It reads pre-split train/validation/test CSV files from the Model_Data_CML_Q folder, trains the XGBoost model,
    evaluates test metrics, and generates SHAP analysis.
"""
# ==============================================================================
# [Documentation Only] How the 'best_params' were obtained:
#
# The optimal hyperparameters for each catchment were determined using
# BayesSearchCV from the 'skopt' library. The tuning pipeline was as follows:
#
# import pandas as pd
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer
# import xgboost as xgb
# from sklearn.model_selection import PredefinedSplit
#
# # 1. Load train and validation sets
# # train_X, train_y = ...
# # val_X, val_y = ...
# # X_all = pd.concat([train_X, val_X])
# # y_all = pd.concat([train_y, val_y])
#
# # 2. Create PredefinedSplit (train = -1, validation = 0)
# # test_fold = [-1] * len(train_X) + [0] * len(val_X)
# # ps = PredefinedSplit(test_fold)
#
# # 3. Define search space
# # param_space = {
# #     'learning_rate': Real(0.005, 0.1),
# #     'n_estimators': Integer(100, 600),
# #     'max_depth': Integer(2, 10),
# #     'colsample_bytree': Real(0.2, 1.0),
# #     'reg_alpha': Real(1e-4, 0.5, prior='log-uniform'),
# #     'reg_lambda': Real(5, 10, prior='log-uniform'),
# #     'subsample': Real(0.7, 1.0),
# #     'min_child_weight': Integer(1, 10)
# # }
#
# # 4. Run BayesSearchCV
# # opt = BayesSearchCV(
# #     estimator=xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
# #     search_spaces=param_space,
# #     n_iter=50,
# #     scoring='neg_root_mean_squared_error',
# #     cv=ps,
# #     n_jobs=-1,
# #     random_state=42
# # )
# # opt.fit(X_all, y_all)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error

# ================= Configuration =================
# Define the base directory and output directory (relative paths for GitHub)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

data_base_dir = os.path.join(base_dir, "data", "Model_Data_CML_Q")
out_dir = os.path.join(base_dir, "outputs")
os.makedirs(out_dir, exist_ok=True)

# Predefined optimal hyperparameters
best_params = {
'A': {'colsample_bytree':0.87,'learning_rate':0.03,'max_depth':3,'n_estimators':357,
      'min_child_weight': 6,'reg_alpha': 0.02,'reg_lambda': 6.39,'subsample': 0.93},
'B': {'colsample_bytree':0.98,'learning_rate':0.04,'max_depth':4,'n_estimators':400,
      'min_child_weight': 6,'reg_alpha': 0.02,'reg_lambda': 6.39,'subsample': 0.93},
'C': {'colsample_bytree':0.93,'learning_rate':0.10,'max_depth':2,'n_estimators':382,
      'min_child_weight': 1,'reg_alpha': 0.007,'reg_lambda': 5,'subsample': 0.805},
'D': {'colsample_bytree':0.99,'learning_rate':0.08,'max_depth':4,'n_estimators':100,
      'min_child_weight': 4,'reg_alpha': 0.001,'reg_lambda': 9.63,'subsample': 0.7},
'E': {'colsample_bytree':0.34,'learning_rate':0.01,'max_depth':5,'n_estimators':381,
      'min_child_weight': 1,'reg_alpha': 0.5,'reg_lambda': 10,'subsample': 0.7},
'F': {'colsample_bytree':0.67,'learning_rate':0.01,'max_depth':6,'n_estimators':300,
      'min_child_weight': 5,'reg_alpha': 0.5,'reg_lambda': 10,'subsample': 0.6},
'K': {'colsample_bytree':0.71,'learning_rate':0.03,'max_depth':6,'n_estimators':400,
      'min_child_weight': 5,'reg_alpha': 0.5,'reg_lambda': 10,'subsample': 0.6}
}

for catchment in ['A','B','C','D','E','F','K']:
    print("="*60)
    print(f"Processing Catchment: {catchment}")
    print("="*60)
    ########################################################
    # 1. Set dynamic parameters based on catchment
    if catchment in ['C', 'D']:
        n_time_steps = 13
    else:
        n_time_steps = 21
    num_CML = 5
    feature_labels = ['Group 1','Group 2','Group 3','Group 4','Group 5','Sin t','Cos t']
    ########################################################
    # 2. Data reading (Load pre-split CSV files directly)
    data_dir = os.path.join(data_base_dir, f"catchment_{catchment}")
    train_file = os.path.join(data_dir, f"catchment_{catchment}_train.csv")
    test_file  = os.path.join(data_dir, f"catchment_{catchment}_test.csv")
    try:
        train = pd.read_csv(train_file, index_col=0)
        test  = pd.read_csv(test_file, index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading data for {catchment}: {e}")
        continue

    # Split features and target
    # Logic: Drop the last two columns (Target and Event)
    train_X, train_y = train.iloc[:, :-2], train.iloc[:, -2]
    test_X,  test_y  = test.iloc[:, :-2], test.iloc[:, -2]
    ########################################################
    # 3. Train Model using optimal parameters
    p = best_params[catchment]
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        learning_rate=p['learning_rate'],
        max_depth=p['max_depth'],
        n_estimators=p['n_estimators'],
        colsample_bytree=p['colsample_bytree']
    )
    # 4. Train on training set
    model.fit(train_X, train_y)
    ########################################################
    # 5. SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(test_X)
    original_shap_values = np.asarray(shap_values.values)

    # --- (1) Global Importance ---
    original_feature_importance = np.mean(np.abs(original_shap_values), axis=0)
    group_importance = np.zeros(num_CML)
    for i in range(0, num_CML):
        group_importance[i] = np.sum(original_feature_importance[i*n_time_steps:(i+1)*n_time_steps])

    global_importance = np.concatenate([group_importance, original_feature_importance[-2:]])
    ascending_order = np.argsort(global_importance)
    ordered_importance = global_importance[ascending_order]
    ordered_feature_labels = np.asarray(feature_labels)[ascending_order]

    plt.figure(figsize=(10, 6))
    plt.barh(ordered_feature_labels, ordered_importance, color="royalblue")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance based on SHAP Values ({catchment})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'SHAP_{catchment}.tif'), dpi=300)
    #plt.show()

    # --- (2) Visualize CML contribution for each sample (Summary Plot) ---
    merged_shap_values = np.zeros((original_shap_values.shape[0], len(feature_labels)))
    for i in range(num_CML):
        merged_shap_values[:, i] = np.sum(original_shap_values[:, i*n_time_steps:(i+1)*n_time_steps], axis=1)

    # Assign Sin t and Cos t directly
    merged_shap_values[:, -2:] = original_shap_values[:, -2:]

    # Construct feature matrix for coloring points
    color_feature_values = pd.concat([test_X.iloc[:, :num_CML], test_X.iloc[:, -2:]], axis=1).copy()
    color_feature_values.columns = feature_labels

    summary_order = ascending_order[::-1]
    ordered_shap_values = (merged_shap_values[:, summary_order])
    ordered_color_values = (color_feature_values.iloc[:, summary_order].to_numpy(dtype=float))
    ordered_feature_labels = (np.asarray(feature_labels)[summary_order].tolist())

    plt.figure(figsize=(10, 6))
    shap.summary_plot(ordered_shap_values, features=ordered_color_values, feature_names=ordered_feature_labels,
                      sort=False, max_display=len(feature_labels), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'SHAPsample_{catchment}.tif'), dpi=300)
    #plt.show()