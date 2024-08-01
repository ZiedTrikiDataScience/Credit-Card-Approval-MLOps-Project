import os
import json
import pickle

import pandas as pd
from prefect import flow, task
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.model_profile import Profile
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
from evidently.model_profile.sections import (
    DataDriftProfileSection, 
    RegressionPerformanceProfileSection
)


@task
def load_data(filename):

    # Load model
    model_file = os.getenv('MODEL_FILE', './models/lin_reg.bin')
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Load data
    data = pd.read_csv(filename)
    data['started_at'] = pd.to_datetime(data['started_at'])
    data['ended_at'] = pd.to_datetime(data['ended_at'])

    data[['rideable_type', 'start_station_id', 'end_station_id']] = data[
        ['rideable_type', 'start_station_id', 'end_station_id']
    ].fillna(-1)

    # Add target column
    data['target'] = data['ended_at'] - data['started_at']
    data.target = data.target.apply(lambda td: td.total_seconds() / 60)
    data = data[(data.target >= 1) & (data.target <= 120)]

    # Feature transformation
    features = ['rideable_type', 'start_station_id', 'end_station_id']
    x_pred = dv.transform(data[features].to_dict(orient='records'))

    # Predict
    data['prediction'] = model.predict(x_pred)

    return data


@task
def run_evidently(ref_data, target_data):
    profile = Profile(
        sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()]
    )
    mapping = ColumnMapping(
        prediction='prediction',
        categorical_features=['rideable_type', 'start_station_id', 'end_station_id'],
        datetime_features=[],
    )
    profile.calculate(ref_data, target_data, mapping)

    dashboard = Dashboard(
        tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)]
    )
    dashboard.calculate(ref_data, target_data, mapping)

    return json.loads(profile.json()), dashboard


@task
def save_html_report(result):
    result[1].save('evidently_report_example.html')


@flow
def batch_analyze():
    target_data = load_data('./data/202206-capitalbikeshare-tripdata.csv')
    ref_data = load_data('./data/202201-capitalbikeshare-tripdata.csv')
    result = run_evidently(ref_data, target_data)
    save_html_report(result)


batch_analyze()

# reference data is what you trained on
# target data is your new data that you're monitoring for drifting
