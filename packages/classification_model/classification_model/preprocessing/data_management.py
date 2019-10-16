import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from classification_model.config import config
#from classification_model.config import logging_config
from classification_model import __version__ as _version 
import logging
_loggger = logging.getLogger(__name__)

def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    # save_file_name = 'classification_model.pkl'
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _loggger.info(f'saved pipeline: {save_file_name}')

    print('saved pipeline')



def load_pipeline(*, file_name: str
                  ) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, '__init__.py']:
            model_file.unlink()
