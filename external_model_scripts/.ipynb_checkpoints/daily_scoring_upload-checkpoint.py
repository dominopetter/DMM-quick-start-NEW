# import pandas as pd
# import numpy as np
# import random
# import math
# import pickle
# import json
# import os

import requests
import datetime
from domino.data_sources import DataSourceClient

# In this example, the scoring data has been sent to an external model, the external model has returned predictions.
# A dataset with the scoring data, model predictions and a prediction id are uploaded to the DMM data source (s3 in this example)

# UPDATE: Your Model Parameters
external_datasource = "demo-bucket"
datasourceType = "s3"
DMM_datasource_name = "se-demo-bucket"
domino_url = "demo2.dominodatalab.com"
DMM_model_id = "65b0525c54ac3acc8cb495d1"
prediction_data_dir = "65b04f6b1266902edb95b260"

# instantiate a client and fetch the datasource instance
object_store = DataSourceClient().get_datasource("{}".format(external_datasource)) # Update

# 
object_store.upload_file("iris_ground_truth_{}_{}_{}.csv".format(month, day, year), "/mnt/code/data/iris_ground_truth_{}_{}_{}.csv".format(month, day, year))













import os
import json
import requests

# Your Domino API key
API_key = os.environ['MY_API_KEY']

# Your Model Monitoring Model ID, created when the model was registered in Step 2.
model_id='65bc2e5f198fe4d19631c582'

# Your organizations's Domino url
your_domino_url = 'demo2.dominodatalab.com'

# Your DMM datasource name
datasource_name = 'se-demo-bucket'

# Your DMM datasource type
datasource_type = 's3'

# The updated path to your prediction dataset
prediction_dataset_name = "iris_ground_truth_1_25_2024.csv"
prediction_dataset_path = "iris_ground_truth_1_25_2024.csv"
prediction_dataset_fileFormat = "csv"

# Set up call headers
headers = {
           'X-Domino-Api-Key': API_key,
           'Content-Type': 'application/json'
          }

prediction_registration_request = {
    "datasetDetails": {
        "name": prediction_dataset_name,
        "datasetType": "file",
        "datasetConfig": {
            "path": prediction_dataset_path,
            "fileFormat": prediction_dataset_fileFormat
        },
        "datasourceName": datasource_name,
        "datasourceType": datasource_type
    }
}

# Make api call
ground_truth_response = requests.request("PUT", prediction_data_url, headers=headers, data = json.dumps(prediction_registration_request))
 
# Print response
print(ground_truth_response.text.encode('utf8'))
 
print('DONE!')



# import requests
# import pandas as pd
# import numpy as np
# import datetime
# from sklearn.datasets import load_iris
# import os
# import datetime


# data = load_iris()

# df = pd.DataFrame(data = data['data'], columns = data.feature_names)
# df['variety'] = data['target']

# scoring_data = df[data.feature_names].copy()

# # Jitter the scoring data
# for row in scoring_data.iterrows():
#     for c in scoring_data.columns:
#         scoring_data[c] = np.maximum(0.1, scoring_data[c] + np.random.normal()/25)
        
# # Save in Domino for future reference

# date = str(datetime.datetime.today()).split()[0]

# # Retreive ingested predictions from Domino Dataset
# date = datetime.datetime.today()
# month = date.month
# day = date.day
# year = date.year

# scoring_data.to_csv('/mnt/data/{}/iris_ground_truth_{}_{}_{}.csv'.format(os.environ.get('DOMINO_PROJECT_NAME'), month, day, year), index=False)

# scoring_data = scoring_data.values.tolist()

# model_url = os.environ.get('MODEL_URL')
# model_auth_token = os.environ.get('MODEL_AUTH_TOKEN')

# response = requests.post(model_url,
#     auth=(
#         model_auth_token,
#         model_auth_token
#     ),
#     json={
#         "data": scoring_data
#     }
# )

    
# print(response.status_code)
# print(response.headers)
# print(response.json())