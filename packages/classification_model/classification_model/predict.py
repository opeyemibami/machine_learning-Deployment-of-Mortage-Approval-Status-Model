import numpy as np
import pandas as pd

from classification_model.preprocessing.data_management import load_pipeline
from classification_model.config import config

pipeline_file_name = 'classification_model.pkl'
_loan_status_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    prediction = _loan_status_pipe.predict(data[config.FEATURES])
    output = list(prediction)
    response = {'predictions': output}

    return response

