import requests
import pandas as pd
import numpy as np
import datetime
from sklearn.datasets import load_iris
import os
import datetime
import pickle
import uuid
from domino.data_sources import DataSourceClient

# UPDATE: Your Model Parameters
external_datasource = "demo-bucket"
datasourceType = "s3"
DMM_datasource_name = "se-demo-bucket"
domino_url = "demo2.dominodatalab.com"
DMM_model_id = "6628103c965e21e5b0d56b29"

# Today's date
date = datetime.datetime.today()
month = date.month
day = date.day - 1
year = date.year

# Load data for scoring
data = load_iris()
df = pd.DataFrame(data = data['data'], columns = data.feature_names)
df['variety'] = data['target']

scoring_data = df[data.feature_names].copy()

# Jitter the scoring data
for row in scoring_data.iterrows():
    for c in scoring_data.columns:
        scoring_data[c] = np.maximum(0.1, scoring_data[c] + np.random.normal()/25)

# Load the "external" model
file_name = "/mnt/code/models/xgb_iris.pkl"
model = pickle.load(open(file_name, "rb"))

# Get model predictions (numeric)
scoring_data = scoring_data.values.tolist()
model_predictions = model.predict(scoring_data)

# Create the scoring dataset for model moniotring

# Data that was scored, model predictions (as strings), timestamp and event ID for model qulaity monitoring.  
predictions = pd.DataFrame(scoring_data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)' ])
predictions['predictions'] = model_predictions
predictions['variety'] = [data.target_names[y] for y in predictions['predictions']]
predictions.drop('predictions', axis=1, inplace=True)
predictions['timestamp']= datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
event_ids = [uuid.uuid4() for x in range(predictions.shape[0])]
predictions['event_id'] = event_ids

# Save version to Domino Dataset for future reference
predictions.to_csv('/mnt/data/{}/external_iris_scoring_data_{}_{}_{}.csv'.format(os.environ.get('DOMINO_PROJECT_NAME'), month, day, year), index=False)

print("Scoring data saved to project's Domino Dataset")

# Create the "dummy" ground truth dataset

ground_truth = pd.DataFrame(columns=['event_id', 'iris_ground_truth'])
ground_truth['event_id'] = predictions['event_id']
ground_truth['iris_ground_truth'] = predictions['variety']

# These row labels help find some diferent iris types in our initial scoring data
end_index = predictions.shape[0]
mid_index = int(round(predictions.shape[0] / 2, 0))

# Simulate some classifcation errors. This makes our confusion matrix interesting.
ground_truth.iloc[0, 1] = 'virginica'
ground_truth.iloc[1, 1] = 'versicolor'
ground_truth.iloc[mid_index-1, 1] = 'versicolor'
ground_truth.iloc[mid_index, 1] = 'virginica'
ground_truth.iloc[end_index-2, 1] = 'setosa'
ground_truth.iloc[end_index-1, 1] = 'setosa'

# Save each version locally 
ground_truth.to_csv('/mnt/data/{}/external_iris_ground_truth_{}_{}_{}.csv'.format(os.environ.get('DOMINO_PROJECT_NAME'), month, day, year), index=False)

print("Ground truth data saved to project's Domino Dataset")
print("Done!")
