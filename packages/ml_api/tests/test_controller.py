from classification_model.config import config as model_config
from classification_model.preprocessing.data_management import load_dataset
from classification_model import __version__ as _version

import json
from flask import jsonify



def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


# def test_prediction_endpoint_returns_prediction(flask_test_client):
#     # Given
#     # Load the test data from the classification_model package
#     # This is important as it makes it harder for the test
#     # data versions to get confused by not spreading it
#     # across packages.
#     test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)  
#     post_json = test_data[0:1].to_json(orient='records')
    

#     # When
#     response = flask_test_client.post('/v1/predict/classification',
#                                       json=post_json)

#     # Then
#     assert response.status_code == 200
#     response_json = json.loads(response.data)
#     prediction = response_json['predictions']
#     response_version = response_json['version']
#     assert prediction == 0
#     assert response_version == _version

# def test_prediction_endpoint_returns_prediction(flask_test_client):
#     # Given
#     # Load the test data from the classification_model package
#     # This is important as it makes it harder for the test
#     # data versions to get confused by not spreading it
#     # across packages.
#     test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)  
#     #post_json = test_data[0:1].to_json(orient='records')
#     test_data = test_data[0:1]
#     cols = test_data.columns
#     json_issue = [c for c in cols if test_data[0:1][c].dtype=='int64']
#     post_json = {} 
#     for i in cols:
#         if i in json_issue:
#             post_json[i]=int(test_data[0:1][i])
#         else:
#             post_json[i]=test_data[0:1][i][0]
#     post_json = str([post_json])
        
    

#     # When
#     response = flask_test_client.post('/v1/predict/classification',
#                                       json=post_json)

#     # Then
#     assert response.status_code == 200
#     response_json = json.loads(response.data)
#     prediction = response_json['predictions']
#     response_version = response_json['version']
#     assert prediction == 0
#     assert response_version == _version




