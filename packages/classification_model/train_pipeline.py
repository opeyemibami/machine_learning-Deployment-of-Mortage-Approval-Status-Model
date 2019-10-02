# import pathlib

# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib

# import pipeline


# PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
# TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
# DATASET_DIR = PACKAGE_ROOT / 'datasets'

# # data
# TESTING_DATA_FILE = DATASET_DIR /'test.csv'
# TRAINING_DATA_FILE = DATASET_DIR /'train.csv'
# TARGET = 'accepted'


# # variables
# FEATURES = ['loan_type', 'property_type','loan_purpose','occupancy', 'loan_amount',
#        'preapproval', 'msa_md', 'state_code', 'county_code','applicant_ethnicity','applicant_race','applicant_sex',
#        'applicant_income', 'population', 'minority_population_pct',
#        'ffiecmedian_family_income', 'tract_to_msa_md_income_pct',
#        'number_of_owner-occupied_units', 'number_of_1_to_4_family_units',
#        'lender', 'co_applicant']


import numpy as np
from sklearn.model_selection import train_test_split

from classification_model import pipeline
from classification_model.preprocessing.data_management import load_dataset, save_pipeline
from classification_model.config import config





# def save_pipeline(*, pipeline_to_persist) -> None:
#     """Persist the pipeline."""

#     save_file_name = 'classification_model.pkl'
#     save_path = TRAINED_MODEL_DIR / save_file_name
#     joblib.dump(pipeline_to_persist, save_path)

#     print('saved pipeline')

def run_training() -> None:
    """Train the model."""

     # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # read training data
    #data = pd.read_csv(TRAINING_DATA_FILE)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.33, random_state=42)  # we are setting the seed here

    pipeline.loan_status_pipe.fit(X_train[config.FEATURES],y_train)

    save_pipeline(pipeline_to_persist=pipeline.loan_status_pipe)


if __name__ == '__main__':
    run_training()
