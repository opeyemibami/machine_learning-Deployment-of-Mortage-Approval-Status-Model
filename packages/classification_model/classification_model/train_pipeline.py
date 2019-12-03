import numpy as np
from sklearn.model_selection import train_test_split

from classification_model import pipeline
from classification_model.preprocessing.data_management import (
    load_dataset, save_pipeline,save_pipeline_swagger)
from classification_model.config import config  
from classification_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)




def run_training() -> None:
    """Train the model."""

     # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)


    # read training data
    #data = pd.read_csv(TRAINING_DATA_FILE)
    # divide train and test

    
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data[config.FEATURES],
    #     data[config.TARGET],
    #     test_size=0.33, random_state=42)  # we are setting the seed here

    pipeline.loan_status_pipe.fit(data[config.FEATURES],data[config.TARGET])
    
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.loan_status_pipe)
    return


# For swagger 
def run_training_swagger() -> None:
    """Train the model."""

     # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)


    pipeline.loan_status_pipe.fit(data[config.FEATURES],data[config.TARGET])
    save_pipeline_swagger(pipeline_to_persist=pipeline.loan_status_pipe)
    return


if __name__ == '__main__':
    run_training()
    run_training_swagger()
