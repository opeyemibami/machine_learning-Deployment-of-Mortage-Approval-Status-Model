import pathlib

import classification_model


PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_models.log'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'
TARGET = 'accepted'


# variables
FEATURES = ['loan_type', 'property_type','loan_purpose','occupancy', 'loan_amount',
       'preapproval', 'msa_md', 'state_code', 'county_code','applicant_ethnicity','applicant_race','applicant_sex',
       'applicant_income', 'population', 'minority_population_pct',
       'ffiecmedian_family_income', 'tract_to_msa_md_income_pct',
       'number_of_owner-occupied_units', 'number_of_1_to_4_family_units',
       'lender', 'co_applicant']


CATEGORICAL_VARS = ['loan_type',
					'property_type',
					'loan_purpose',
					'occupancy',
					'preapproval',
					'state_code',
					'county_code',
					'applicant_ethnicity',
					'applicant_race',
					'applicant_sex',
					'co_applicant']

VAR_TO_BE_LABELENCODED = ['co_applicant']

NUMERICAL_VARS_FOR_MISSINGNESS_FEAT = ['applicant_income','population','number_of_owner-occupied_units','number_of_1_to_4_family_units']


NUMERICAL_VARS_FOR_MEAN_IMPUTING  =['applicant_income','population','ffiecmedian_family_income','tract_to_msa_md_income_pct','number_of_1_to_4_family_units']

NUMERICAL_VARS_FOR_MEDIAN_IMPUTING = ['minority_population_pct','number_of_owner-occupied_units']

# ONEHOTENCODER_VARS = ['loan_purpose','applicant_ethnicity']
ONEHOTENCODER_VARS = ['applicant_ethnicity']


INTERQUANTILE_OUTLIER_VARS =['loan_amount']

GUASSIAN_OUTLIER_VARS =['applicant_income']

INCOME_LOAN_RATIO_VARS = ['loan_amount','applicant_income']

DROP_FEATURES = ['applicant_race'] 


NUMERICAL_NA_NOT_ALLOWED = [
    feature for feature in FEATURES
    if feature not in NUMERICAL_VARS_FOR_MEAN_IMPUTING + NUMERICAL_VARS_FOR_MEDIAN_IMPUTING 
    ]

PIPELINE_NAME = 'gradientboosting_classification'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'