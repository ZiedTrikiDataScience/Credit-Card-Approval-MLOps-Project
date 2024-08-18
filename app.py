import pickle

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from mlflow import MlflowClient

################################################################
################ Local Artifact Store ##########################
################################################################

# Load the trained model from the local model registry
#model = mlflow.pyfunc.load_model(r"models:/Champion_Credit_Card_Approval_Model/1")

# Docker :

run_id = "963014787172925059/cea202bfa8964234975e8cd70b7a4ecf"
model = mlflow.pyfunc.load_model(f'./mlruns/{run_id}/artifacts/model')

print("model is: " , model, "\n","\n","\n","\n")

# Load the preprocessor from the local path
# preprocessor_path = r"C:\Users\ZiedTriki\OneDrive - Brand Delta\Desktop\Zied DS\final_mlops_credit_default\mlruns\963014787172925059\cea202bfa8964234975e8cd70b7a4ecf\artifacts\preprocessor\preprocessor.pkl"
#preprocessor_path = r"\mlruns\963014787172925059\cea202bfa8964234975e8cd70b7a4ecf\artifacts\preprocessor\preprocessor.pkl"


# Docker
preprocessor_path = f'./mlruns/{run_id}/artifacts/preprocessor/preprocessor.pkl'

with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

print("model is", model)
print("preprocessor is", preprocessor)
#########################################################


################################################################
################ AWS S3 Artifact Store ##########################
################################################################

# Set the MLflow tracking URI
# TRACKING_SERVER_HOST = "ec2-203-0-113-25.compute-1.amazonaws.com" 

# mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

# client = mlflow.tracking.MlflowClient()

# RUN_ID = '963014787172925059/cea202bfa8964234975e8cd70b7a4ecf'

# preprocessor_path = client.download_artifacts(run_id = RUN_ID, path='artifacts/preprocessor/preprocessor.pkl')

# with open(preprocessor_path, 'rb') as f:
#    preprocessor = pickle.load(f)

# Load the trained model from the local model registry
# model = mlflow.pyfunc.load_model(f"s3://credit_card_approval/0/{RUN_ID}/artifacts/models/Champion_Credit_Card_Approval_Model/1")

# print("model is", model)
# print("preprocessor is", preprocessor)
#########################################################



# Define the Flask app :

app = Flask('Credit_Card_Approval')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from the request    
    file = request.files['file']
    
    try:
        # Read the Excel file
        prediction_dataset = pd.read_excel(file)
        prediction_dataset = prediction_dataset.drop(columns=['Credit_Card_Approval', 'Ind_ID', 'EMAIL_ID', 'Birthday_count'], axis=1)
        
        dataset_columns = list(prediction_dataset.columns)
        selected_features = ['Annual_income' ,'Employed_days' ,'Family_Members' ,'Housing_type','Type_Occupation']
        
        # Apply the preprocessor
        X_prediction = preprocessor.transform(prediction_dataset)

        X_prediction = pd.DataFrame(X_prediction, columns= dataset_columns)

        X_prediction = X_prediction[selected_features]
                
        # Make prediction
        predictions = model.predict(X_prediction)
        
        # Convert predictions to list for JSON serialization
        predictions_list = predictions.tolist()
        
        # Create a result list with human-readable outcomes
        results = ['Credit Card Approved' if pred == 1 else 'Credit Card not Approved' for pred in predictions_list]
        
        # Return the predictions and results as JSON
        return jsonify({
            'predictions': predictions_list,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
 