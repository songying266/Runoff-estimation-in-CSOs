This package contains an example script and instructions to reproduce the machine learning and SHAP analysis for sub-catchment A.

Python script:

run_catchment.py demonstrates the full pipeline for sub-catchments:

1) load train/validation/test CSVs,

2) hyperparameter tuning with BayesSearchCV using a PredefinedSplit (train vs val),

3) final training on train+val, evaluation on test,

4) SHAP-based interpretability (global importance by CML groups; per-time-step within one group; intensity-stratified SHAP for small/moderate/heavy flows).


Input data-each CSV files include:

Features: input variables (CML group time series, and additional features such as sin/cos of time).

Target column: runoff/discharge values (the second-to-last column).

Event column: event identifiers (column name 'event').

Output:

predictions_test.csv – observed vs. modelled runoff on the test set.

shap_groups.png – global importance of CML groups.

shap_summary_groups.png – SHAP summary plot by CML groups.

shap_timecurve_{GroupName}.png – SHAP importance curve over time steps (dynamically named after the most important group identified in the global analysis, e.g., shap_timecurve_Group1.png).

shap_by_rain_intensity.png – SHAP importance under small / moderate / heavy rainfall conditions.


If you want to run the model using RG data instead, please simply open run_catchment.py, locate the data_dir variable definition (around Line 47), and change "CML_inputs" to "RGs_inputs".


