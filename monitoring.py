import pandas as pd
import numpy as np
import pickle
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPerformancePreset
from evidently.metrics import *

# Load the reference(training) data :
reference_data = pd.read_excel(r"C:\Users\ZiedTriki\OneDrive - Brand Delta\Desktop\Zied DS\final_mlops_credit_default\src\Credit_Card_Approval_prediction.xlsx")

# Load the new current data :
current_data = pd.read_excel(r"C:\Users\triki\Desktop\MLOps\final_mlops_credit_default\data_to_test_and_predict.xlsx")

# Load your trained model from MLflow model registry
model = mlflow.pyfunc.load_model(r"models:/Champion_Credit_Card_Approval_Model/1")

# Load the preprocessor
with open(r"mlruns\963014787172925059\cea202bfa8964234975e8cd70b7a4ecf\artifacts\preprocessor\preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Prepare your reference data
X_ref = reference_data.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID'], axis=1)
y_ref = reference_data['Credit_Card_Approval']

# Prepare your current data
X_current = current_data.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID'], axis=1)
y_current = current_data['Credit_Card_Approval']

# Preprocess the data
X_ref_processed = preprocessor.transform(X_ref)
X_current_processed = preprocessor.transform(X_current)

# Get feature names
feature_names = preprocessor.get_feature_names_out()

# Create DataFrames with processed data
reference_data = pd.DataFrame(X_ref_processed, columns=feature_names)
reference_data['target'] = y_ref

current_data = pd.DataFrame(X_current_processed, columns=feature_names)
current_data['target'] = y_current

# Make predictions
reference_data['prediction'] = model.predict(X_ref_processed)
current_data['prediction'] = model.predict(X_current_processed)

# Data Drift Report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(reference_data=reference_data, current_data=current_data)
data_drift_report.save_html("data_drift_report.html")

# Target Drift Report
target_drift_report = Report(metrics=[
    TargetDriftPreset(),
])

target_drift_report.run(reference_data=reference_data, current_data=current_data)
target_drift_report.save_html("target_drift_report.html")

# Data Quality Report
data_quality_report = Report(metrics=[
    DataQualityPreset(),
])

data_quality_report.run(reference_data=reference_data, current_data=current_data)
data_quality_report.save_html("data_quality_report.html")

# Classification Performance Report
classification_performance_report = Report(metrics=[
    ClassificationPerformancePreset(),
])

classification_performance_report.run(reference_data=reference_data, current_data=current_data)
classification_performance_report.save_html("classification_performance_report.html")

# Custom Metrics Report
custom_report = Report(metrics=[
    ColumnDriftMetric(column_name="Annual_income"),
    ColumnDriftMetric(column_name="Employed_days"),
    ColumnDriftMetric(column_name="Family_Members"),
    ColumnDriftMetric(column_name="Housing_type"),
    ColumnDriftMetric(column_name="Type_Occupation"),
    ClassificationClassBalance(),
    ClassificationConfusionMatrix(),
    ClassificationQualityMetric(),
    ProbClassificationQualityMetric(),
])

custom_report.run(reference_data=reference_data, current_data=current_data)
custom_report.save_html("custom_report.html")

# Function to check data drift and print alerts
def check_data_drift(report):
    data_drift_metric = report.metrics[0]
    if data_drift_metric.result.dataset_drift:
        print("ALERT: Data drift detected!")
        for feature, drift in data_drift_metric.result.feature_result.items():
            if drift.drift_detected:
                print(f"Drift detected in feature: {feature}")

check_data_drift(data_drift_report)