import pickle

import mlflow
import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder)
from xgboost import XGBClassifier


def load_and_merge_data(original_data_path, new_data_path):
    """
    Load both original and new data, then merge them for training.
    """
    original_data = pd.read_excel(original_data_path)
    new_data = pd.read_excel(new_data_path)
    
    # Ensure the new data has the same structure as the original data
    new_data = new_data[original_data.columns]
    
    # Concatenate the datasets
    merged_data = pd.concat([original_data, new_data], axis=0, ignore_index=True)
    
    return merged_data

def preprocess_data(data):
    X = data.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID'], axis=1)
    y = data['Credit_Card_Approval']
    
    X['Birthday_count'].fillna(0, inplace=True)
    X['Age'] = np.abs(np.floor(X['Birthday_count'] / 365)).astype(int)
    X = X.drop(columns=['Birthday_count'], axis=1)
    X['Employed_days'] = np.abs(X['Employed_days'])
    
    return X, y

def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('abs_transformer', FunctionTransformer(np.abs))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('count_frequency', CountFrequencyEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def select_features(X, y, n_features=5):
    model = RandomForestClassifier(max_depth=5, n_estimators=500)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return rfe

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def objective(params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(nested=True):
        mlflow.set_tag('Model', 'XGBoost_Credit_Card_Approval')
        mlflow.log_params(params)
        
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            reg_alpha=params['reg_alpha'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)

        preds_test = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, preds_test)
        
        mlflow.log_metric("Test_Accuracy", accuracy_test)
        
        return {'loss': -accuracy_test, 'status': STATUS_OK, 'model': model}

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = create_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    rfe = select_features(X_train_processed, y_train)
    X_train_selected = rfe.transform(X_train_processed)
    X_test_selected = rfe.transform(X_test_processed)
    
    X_train_resampled, y_train_resampled = apply_smote(X_train_selected, y_train)
    
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 20, 80, 10)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 6, 1)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
        'reg_alpha': hp.uniform('reg_alpha', 0, 5),
    }
    
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, X_train_resampled, y_train_resampled, X_test_selected, y_test),
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    
    best_model = trials.best_trial['result']['model']
    
    return best_model, preprocessor, rfe

def evaluate_model(model, X, y, preprocessor, rfe):
    X_processed = preprocessor.transform(X)
    X_selected = rfe.transform(X_processed)
    predictions = model.predict(X_selected)
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)
    return accuracy, report

def save_model(model, preprocessor, model_path, preprocessor_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    

def main(original_data_path, new_data_path):
    mlflow.set_experiment("Credit_Card_Approval")
    
    data = load_and_merge_data(original_data_path, new_data_path)

    X, y = preprocess_data(data)
    
    best_model, preprocessor, rfe = train_model(X, y)
    
    accuracy, report = evaluate_model(best_model, X, y, preprocessor, rfe)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    with mlflow.start_run():
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.xgboost.log_model(best_model, artifact_path="Models_Mlflow")
    
    save_model(best_model, preprocessor, "xgboost_credit_card_approval.pkl", "preprocessor.pkl")
    
    print("Model, preprocessor, and RFE saved.")

if __name__ == "__main__":
    original_data_path = r"./src/Credit_Card_Approval_prediction.xlsx"
    new_data_path = r"./data_to_test_and_predict.xlsx"
    main(original_data_path, new_data_path)