# Basin Flow Prediction & Interpretability (XGBoost + SHAP)
This package contains an example script and instructions to reproduce the machine learning and SHAP analysis. The pipeline uses predefined optimal hyperparameters to train the model and interpret it using SHAP.

## Python script: `run_XGBoost_model.py`
The script demonstrates the full pipeline for the sub-catchments (A-K):
1) **Load data**: Read pre-split train/test CSVs.
2) **Train & Evaluate**: Train the XGBoost model using optimal parameters and evaluate on the test set (R², PCC, RMSE, RE).
3) **SHAP-based interpretability**: Generate global importance aggregated by CML groups, and a sample-level SHAP summary plot.

## Input data
The data is stored in the `data/` directory. Each CSV file should include:
*   **Features**: Input variables (event-based CML group time series, event-based RG time seriesand additional features such as sin/cos of time).
*   **Target column**: Runoff/discharge values (the second-to-last column).
*   **Event column**: Event identifiers (column name `'event'`, the last column).

## Output
Results and plots are saved to `./outputs/`:
*   `predictions_test.csv` – Observed vs. modelled runoff on the test set.
*   `SHAP_{catchment}.tif` – Global feature importance based on aggregated SHAP values.
*   `SHAPsample_{catchment}.tif` – Sample-level SHAP summary plot by CML groups.

## Note on RG Data
The provided data folders include both `Model_Data_CML_Q` and `Model_Data_RG_Q`. 
*   The main script is specifically designed for **CML data** because the SHAP aggregation logic relies on the 5 CML groups.
*   If you want to train the model using **RG data** instead of CML, simply open `run_XGBoost_model.py`, locate `data_base_dir` (around Line 28), and change `"Model_Data_CML_Q"` to `"Model_Data_RG_Q"`. *(Note: The SHAP plotting logic may need manual adjustment for RG feature structures).*