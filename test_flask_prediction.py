import requests
import pandas as pd

url = 'http://localhost:5001/predict'

# Path to your Excel file
excel_file_path = r"data_to_test_and_predict.xlsx"

# Read the Excel file
df = pd.read_excel(excel_file_path)

print(f"Loaded dataset with {len(df)} rows")

# Prepare the file for sending
files = {'file': open(excel_file_path, 'rb')}



# Send the request
response = requests.post(url, files=files)





if response.status_code == 200:
    result = response.json()
    predictions = result['predictions']
    results = result['results']
    
    # Add predictions and results to the dataframe
    df['Prediction'] = predictions
    df['Result'] = results
    
    # Display results
    print(df)
    
    # Optionally, save the results to a new Excel file
    df.to_excel('prediction_results.xlsx', index=False)
    print("Results saved to 'prediction_results.xlsx'")
else:
    print(f"Error: {response.json()['error']}")