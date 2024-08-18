from unittest import mock

import numpy as np
import pandas as pd
import pytest
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Annual_income': [50000, 60000, np.nan],
        'Employed_days': [365, 730, 1095],
        'Housing_type': ['Own', 'Rent', 'Rent'],
        'Type_Occupation': ['Tech', 'Services', 'Tech'],
    })

@pytest.fixture
def preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('abs_transformer', FunctionTransformer(np.abs))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('count_frequency', CountFrequencyEncoder())
    ])

    return ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, ['Annual_income', 'Employed_days']),
            ('categorical', categorical_transformer, ['Housing_type', 'Type_Occupation'])
        ])

def test_preprocessing_pipeline(preprocessor, sample_data):
    processed_data = preprocessor.fit_transform(sample_data)
    assert processed_data.shape == (3, 4), "The processed data should have 4 features."
    assert not np.isnan(processed_data).any(), "There should be no missing values after preprocessing."


def test_rfe_feature_selection():
    model = RandomForestClassifier(n_estimators=10)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)

    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(X, y)

    assert rfe.n_features_ == 5, "RFE should select 5 features."
    assert rfe.support_.sum() == 5, "5 features should be marked as selected."



def test_model_training():
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    
    model = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_train)
    accuracy = accuracy_score(y_train, preds)
    
    assert 0.5 <= accuracy <= 1.0, "The training accuracy should be between 0.5 and 1.0."




def test_integration_pipeline(preprocessor):
    # Mock the data with correct column names as expected by the preprocessor
    X_mock = pd.DataFrame({
        'Annual_income': np.random.rand(100) * 100000,
        'Employed_days': np.random.randint(0, 3650, size=100),
        'Housing_type': np.random.choice(['Own', 'Rent'], size=100),
        'Type_Occupation': np.random.choice(['Tech', 'Services', 'Other'], size=100)
    })
    
    y_mock = np.random.randint(0, 2, size=100)

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X_mock)
    
    # Train the model
    model = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_processed, y_mock)
    
    # Make predictions
    preds = model.predict(X_processed)
    accuracy = accuracy_score(y_mock, preds)
    
    # Assert that the accuracy is reasonable
    assert accuracy > 0.5, "The pipeline should produce a model with accuracy greater than 0.5."