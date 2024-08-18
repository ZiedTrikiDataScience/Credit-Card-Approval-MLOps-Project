import datetime
import io
import logging
import pickle
import random
import smtplib
import time
import uuid
from datetime import timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import mlflow
import numpy as np
import pandas as pd
import psycopg
import pytz
from evidently import ColumnMapping
from evidently.metrics import (ColumnDriftMetric, DatasetDriftMetric,
                               DatasetMissingValuesMetric)
from evidently.report import Report
from prefect import flow, task
# Add imports for alerting and retraining
from prefect.tasks import task_input_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 20
rand = random.Random()

create_table_statement = """
drop table if exists model_metrics;
create table model_metrics(
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    timestamp timestamp
)
"""

# Define metric thresholds
PREDICTION_DRIFT_THRESHOLD = 0.5
NUM_DRIFTED_COLUMNS_THRESHOLD = 3
SHARE_MISSING_VALUES_THRESHOLD = 0.2

reference_data = pd.read_parquet(r"./monitoring/reference_dataset.parquet")

run_id = "963014787172925059/cea202bfa8964234975e8cd70b7a4ecf"
model = mlflow.pyfunc.load_model(f'./mlruns/{run_id}/artifacts/model')

preprocessor_path = f'./mlruns/{run_id}/artifacts/preprocessor/preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

current_data = pd.read_excel(r"./data_to_test_and_predict.xlsx")

# Identify numeric and categorical columns
numeric_features = reference_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = reference_data.select_dtypes(include=['object']).columns.tolist()

# Apply the processing to the new current data
prediction_dataset = current_data.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID', 'Birthday_count'], axis=1)

dataset_columns = list(prediction_dataset.columns)
selected_features = ['Annual_income', 'Employed_days', 'Family_Members', 'Housing_type', 'Type_Occupation']

# Apply the preprocessor
X_prediction = preprocessor.transform(prediction_dataset)
X_prediction = pd.DataFrame(X_prediction, columns=dataset_columns)
processed_prediction_columns = X_prediction[selected_features]

column_mapping = ColumnMapping(
    prediction='predictions',
    numerical_features=numeric_features,
    categorical_features=categorical_features,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='predictions'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, current_data, processed_prediction_columns):
    current_data['predictions'] = model.predict(processed_prediction_columns)

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    timestamp = datetime.datetime.now()

    curr.execute(
        "INSERT INTO model_metrics(prediction_drift, num_drifted_columns, share_missing_values, timestamp) VALUES (%s, %s, %s, %s)",
        (prediction_drift, num_drifted_columns, share_missing_values, timestamp)
    )

    return prediction_drift, num_drifted_columns, share_missing_values

@task
def send_alert(message):
    # Configure your email settings
    sender_email = "your_email@example.com"
    receiver_email = "receiver_email@example.com"
    password = "your_email_password"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Model Monitoring Alert"

    msg.attach(MIMEText(message, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def check_if_model_needs_retraining(prediction_drift, num_drifted_columns, share_missing_values):
    if (prediction_drift > PREDICTION_DRIFT_THRESHOLD or 
        num_drifted_columns > NUM_DRIFTED_COLUMNS_THRESHOLD or 
        share_missing_values > SHARE_MISSING_VALUES_THRESHOLD):
        return True
    return False

@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            prediction_drift, num_drifted_columns, share_missing_values = calculate_metrics_postgresql(curr, current_data, processed_prediction_columns)
            
            if check_if_model_needs_retraining(prediction_drift, num_drifted_columns, share_missing_values):
                alert_message = f"Model needs retraining. Metrics: Prediction Drift = {prediction_drift}, Drifted Columns = {num_drifted_columns}, Missing Values = {share_missing_values}"
                send_alert(alert_message)
                retrain_model()

        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=10)
        logging.info("Data is Sent!")

@task
def retrain_model():
    # Import the necessary libraries and functions from your training script
    from retrain_model import (evaluate_model, load_and_merge_data,
                               preprocess_data, train_model)
    
    original_data_path = r"C:\Users\triki\Desktop\MLOps and GenAi\Credit Card Approval MLOps Project\src\Credit_Card_Approval_prediction.xlsx"
    new_data_path = r"./data_to_test_and_predict.xlsx"  # Path to your new current data
    
    data = load_and_merge_data(original_data_path, new_data_path)
    
    X, y = preprocess_data(data)
    
    best_model, preprocessor, rfe = train_model(X, y)
    
    accuracy, report = evaluate_model(best_model, X, y, preprocessor, rfe)
    
    # Log the new model with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    logging.info(f"Model retrained with accuracy: {accuracy}")

@flow
def model_monitoring_flow():
    while True:
        batch_monitoring_backfill()
        time.sleep(3600)  # Run every hour

if __name__ == '__main__':
    model_monitoring_flow()