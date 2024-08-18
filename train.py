import math
import os
import pickle

import mlflow
import mlflow.sklearn
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder)
from xgboost import XGBClassifier

##############################################################
################ Local Server ################################

# mlflow.set_tracking_uri("file:///c:/Users/ZiedTriki/OneDrive%20-%20Brand%20Delta/Desktop/Zied%20DS/final_mlops_credit_default/mlruns")

tracking_uri = mlflow.get_tracking_uri()

print(tracking_uri)
###############################################################



################################################################
################ Aws EC2 Server ################################
################################################################


#os.environ["AWS_PROFILE"] = os.environ["AWS_PROFILE"] = "credit_card_approval_profile"

#TRACKING_SERVER_HOST = "ec2-203-0-113-25.compute-1.amazonaws.com" 

#mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

#print(tracking_uri)
#################################################################



# Set the experiment name
experiment_name = "Credit_Card_Approval"
# MLflow setup
mlflow.set_experiment(experiment_name)



# Read the data
data = pd.read_excel(r"./src/Credit_Card_Approval_prediction.xlsx")


# Identify features and target
X = data.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID'], axis=1)
y = data['Credit_Card_Approval']


# Train Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  stratify=y, random_state=42)



# Initial Processing Train : 
X_train['Birthday_count'].fillna(0, inplace=True)  # Replace NaN with 0
X_train['Age'] = np.abs(np.floor(X_train['Birthday_count'] / 365)).astype(int)
X_train = X_train.drop(columns=['Birthday_count'], axis=1)
X_train['Employed_days'] = np.abs(X_train['Employed_days'])

# Initial Processing Test : 
X_test['Birthday_count'].fillna(0, inplace=True)  # Replace NaN with 0
X_test['Age'] = np.abs(np.floor(X_test['Birthday_count'] / 365)).astype(int)
X_test = X_test.drop(columns=['Birthday_count'], axis=1)
X_test['Employed_days'] = np.abs(X_test['Employed_days'])


# Create the test and train dataset to be used for reference and train data creation :
test_data = X_test.copy()
test_data['Credit_Card_Approval'] = y_test.values

train_data = X_train.copy()
train_data['Credit_Card_Approval'] = y_train.values


# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns
all_columns = list(numeric_features) + list(categorical_features)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('abs_transformer', FunctionTransformer(np.abs))

])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('count_frequency', CountFrequencyEncoder())
])

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])


# Fit the preprocessor and transform the data
X_processed = preprocessor.fit_transform(X_train)


X_train = pd.DataFrame(X_processed, columns=all_columns)

X_processed = X_train

# Get feature names after preprocessing
numeric_feature_names = numeric_features.tolist()
categorical_feature_names = preprocessor.named_transformers_['categorical']['count_frequency'].get_feature_names_out(categorical_features)
feature_names = numeric_feature_names + categorical_feature_names

# Initialize RFE with the Random Forest model and specify the number of features to select
model = RandomForestClassifier(max_depth=5, n_estimators=500)
rfe = RFE(estimator=model, n_features_to_select=5)

# Fit RFE
rfe = rfe.fit(X_processed, y_train)

# Get the ranking of features
ranking = rfe.ranking_
selected_features = np.array(feature_names)[rfe.support_]

print("\n", "\n", "Selected Features are:", selected_features , "\n","\n")


# Evaluate the model with selected features
X_selected = rfe.transform(X_processed)

"""
#############################################################
# Define the final preprocessor with the selected variables :
#############################################################
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('count_frequency', CountFrequencyEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, ['Annual_income', 'Employed_days', 'Family_Members']),
        ('categorical', categorical_transformer, ['Housing_type','Type_Occupation'])
    ])

selected_dataset = pd.DataFrame(X, columns=selected_features)

preprocessor.fit(selected_dataset)

#############################################################
#############################################################
"""

# Apply SMOTE for imbalance handling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected , y_train)
X_train , y_train = X_resampled, y_resampled



# Apply Preprocessing on the test data :
X_test = preprocessor.transform(X_test)
X_test = pd.DataFrame(X_test, columns=feature_names)
X_test = X_test[selected_features]

# X_test = X_test['Annual_income' ,'Employed_days' ,'Family_Members' ,'Housing_type','Type_Occupation']

# Final X_train and X_test :
X_test = np.array(X_test)

print("X-train shape ;" , X_train.shape)
print("y_train shape ;" , y_train.shape)
print("X-test shape ;" , X_test.shape)
print("y_test shape ;" , y_test.shape)





# Define the objective function for Hyperopt
def objective(params):
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

        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, preds_train)
        accuracy_test = accuracy_score(y_test, preds_test)
        
        mlflow.log_metric("Train_Accuracy", accuracy_train)
        mlflow.log_metric("Test_Accuracy", accuracy_test)
        


        with open("preprocessor.pkl", "wb") as f_out:
            pickle.dump(preprocessor, f_out)
        mlflow.log_artifact("preprocessor.pkl", artifact_path="preprocessor")

        mlflow.xgboost.log_model(model, artifact_path="Models_Mlflow")
        
        return {'loss': -accuracy_test, 'status': STATUS_OK}

# Define the search space for hyperparameters
space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 20, 80, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 6, 1)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5 ),
    'reg_alpha': hp.uniform('reg_alpha', 0, 5),
  
}

# Perform hyperparameter optimization
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials,
    rstate=np.random.default_rng(42)
)





# Train the final model with the best parameters
best_model = XGBClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    reg_alpha=best_params['reg_alpha'],
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

#####################################################
######## Make Pipeline ##############################
#####################################################
"""
pipeline = make_pipeline(
    preprocessor,
    best_model)

pipeline.fit(X_train, y_train)

# Evaluate the final model
preds_train = pipeline.predict(X_train)
preds_test = pipeline.predict(X_test)

"""
best_model.fit(X_train, y_train)

# Evaluate the final model
preds_train = best_model.predict(X_train)
preds_test = best_model.predict(X_test)

accuracy_train = accuracy_score(y_train, preds_train)
accuracy_test = accuracy_score(y_test, preds_test)

report = classification_report(y_test, preds_test)

print(f"Best parameters: {best_params}")
print(f"Train Accuracy: {accuracy_train}")
print(f"Test Accuracy: {accuracy_test}")
print("Classification Report:")
print(report)

# Log the final model with MLflow
with mlflow.start_run() as run:
    with open("preprocessor.pkl", "wb") as f_out:
        pickle.dump(preprocessor, f_out)
    mlflow.log_artifact("preprocessor.pkl", artifact_path="preprocessor")

    mlflow.xgboost.log_model(best_model, artifact_path="Models_Mlflow")

    mlflow.log_params(best_params)
    mlflow.log_metric("Train_Accuracy", accuracy_train)
    mlflow.log_metric("Test_Accuracy", accuracy_test)

    mlflow.log_text(report, "classification_report.txt")

# Save the model to a file
model_path = "xgboost_credit_card_approval.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"Model saved to {model_path}")


# Create the predictions column in the current dataset (train data)
current_data_processed = preprocessor.transform(train_data.drop('Credit_Card_Approval', axis=1))
current_data_processed = pd.DataFrame(current_data_processed, columns=feature_names)
current_data_selected = current_data_processed[selected_features]
current_preds = best_model.predict(current_data_selected)
train_data["predictions"] = current_preds
train_data.to_parquet("monitoring/reference_dataset.parquet")
train_data.to_excel("monitoring/reference_dataset.xlsx")

# Create the predictions column in the reference dataset (test data)
reference_data_processed = preprocessor.transform(test_data.drop('Credit_Card_Approval', axis=1))
reference_data_processed = pd.DataFrame(reference_data_processed, columns=feature_names)
reference_data_selected = reference_data_processed[selected_features]
reference_preds = best_model.predict(reference_data_selected)
test_data["predictions"] = reference_preds
test_data.to_parquet("monitoring/current_dataset.parquet")
test_data.to_excel("monitoring/current_dataset.xlsx")

print("Current and reference datasets with predictions have been saved.")