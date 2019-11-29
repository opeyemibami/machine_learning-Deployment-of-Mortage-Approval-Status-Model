from flask import Blueprint, request, jsonify
from classification_model.predict import make_prediction
from ml_api.api.config import get_logger

###############################
#for flasgger purpose 
import pandas as pd
from classification_model.predict import make_prediction_swagger
from api.app import create_app_for_Flasgger
prediction_app_swagger,swagger = create_app_for_Flasgger()

###############################

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/v1/predict/classification', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        result = make_prediction(input_data=json_data)
        _logger.info(f'Outputs: {result}')

        predictions = result.get('predictions')[0]
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'version': version})


@prediction_app_swagger.route('/additional/outputchannel/classification',methods=["POST"])
def predict_loan_status():
    """Example file endpoint returning a prediction of loan status
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"))
    testfile_json = input_data.to_json(orient='records')

    
    subject = make_prediction_swagger(input_data=testfile_json)
    #subject = make_prediction(input_data=testfile_json)
    
    prediction = subject.get('predictions')
    # prediction_to_int = list()
    # for output in prediction:
    #     prediction_to_int.append(int(output))
    return str(list(prediction))
    #return jsonify({'prediction':prediction_to_int})
    
    
