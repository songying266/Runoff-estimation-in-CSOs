This package contains an example script and instructions to reproduce the machine learning and SHAP analysis for sub-catchment A.

Python script:

run_catchment A.py demonstrates the full pipeline for sub-catchment A:

1) load train/validation/test CSVs,

2) hyperparameter tuning with BayesSearchCV using a PredefinedSplit (train vs val),

3) final training on train+val, evaluation on test,

4) SHAP-based interpretability (global importance by CML groups; per-time-step within one group; intensity-stratified SHAP for small/moderate/heavy flows).

Input data-each CSV file should include:

Features: input variables (CML group time series, and possibly additional features such as sin/cos of time).

Target column: runoff/discharge values (the second-to-last column).

Event column: event identifiers (column name 'event').


Output:

predictions_test.csv – observed vs. modelled runoff on the test set.

shap_groups.png – global importance of CML groups.

shap_summary_groups.png – SHAP summary plot by CML groups.

shap_timecurve_group1.png – SHAP importance curve over time steps (example for Group 1).

shap_by_rain_intensity.png – SHAP importance under small / moderate / heavy rainfall conditions.

Console output: best hyperparameters from BayesSearchCV, and test set metrics (NSE, PCC, RMSE, RE).


Notes:

The example is provided only for catchment A.

For other catchments, the code can be easily adapted by replacing the input CSVs and adjusting the number of groups/time steps if needed.

