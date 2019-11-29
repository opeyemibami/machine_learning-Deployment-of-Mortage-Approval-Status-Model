import numpy as np
import pandas as pd

from classification_model.preprocessing.data_management import load_pipeline
from classification_model.config import config
from classification_model.preprocessing.validation import validate_inputs
from regression_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_loan_status_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _loan_status_pipe.predict(validated_data[config.FEATURES])
    output = list(prediction)
    response = {'predictions': output}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {response}')

    return response


#for swagger #############################################

#pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_loan_status_pipe_swagger = load_pipeline(file_name='classification_model.pkl')

def make_prediction_swagger(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _loan_status_pipe_swagger.predict_proba(validated_data[config.FEATURES])
    output = list(prediction)
    prob = np.array(output)
    prob = prob[:,1]
    accepted_loan_value = list(np.round((prob * validated_data['loan_amount']),2))
    status = list(np.where(prob>=0.5,"Accepted","Declined"))
    result = list()
    for index,value in enumerate(status):
        if(value=='Accepted'):
            result.append([value,accepted_loan_value[index]])
        else:
            result.append([value,0])



    # response = {'predictions': output}
    response = {'predictions': result}

    return response
###########################################################