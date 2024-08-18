import datetime
import io
import logging
import pickle
import random
import time
import uuid

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 20
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

reference_data = pd.read_parquet(r"./monitoring/reference_dataset.parquet")

run_id = "963014787172925059/cea202bfa8964234975e8cd70b7a4ecf"
model = mlflow.pyfunc.load_model(f'./mlruns/{run_id}/artifacts/model')

print("model is: " , model, "\n","\n","\n","\n")

preprocessor_path = f'./mlruns/{run_id}/artifacts/preprocessor/preprocessor.pkl'

with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

print("model is", model)
print("preprocessor is", preprocessor)

current_data = pd.read_excel(r"./data_to_test_and_predict.xlsx")



# Identify numeric and categorical columns
numeric_features = reference_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = reference_data.select_dtypes(include=['object']).columns.tolist()


# Apply the processing to the new current data :

prediction_dataset = current_data
prediction_dataset = prediction_dataset.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID', 'Birthday_count'], axis=1)

dataset_columns = list(prediction_dataset.columns)
selected_features = ['Annual_income' ,'Employed_days' ,'Family_Members' ,'Housing_type','Type_Occupation']

# Apply the preprocessor
X_prediction = preprocessor.transform(prediction_dataset)

X_prediction = pd.DataFrame(X_prediction, columns= dataset_columns)

processed_prediction_columns = X_prediction[selected_features]


column_mapping = ColumnMapping(
    prediction='predictions',
    numerical_features=numeric_features,
    categorical_features=categorical_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='predictions'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
]
)
@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i , current_data , processed_prediction_columns):
	
	#current_data.fillna(0, inplace=True)
	current_data['predictions'] = model.predict(processed_prediction_columns)

	report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into dummy_metrics(prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s)",
		(prediction_drift, num_drifted_columns, share_missing_values)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 100):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i , current_data, processed_prediction_columns)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("Data is Sent!")

if __name__ == '__main__':
	batch_monitoring_backfill()