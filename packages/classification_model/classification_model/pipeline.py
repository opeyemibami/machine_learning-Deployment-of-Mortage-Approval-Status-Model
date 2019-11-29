from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from classification_model import data_preprossesors as pp

import logging


_logger = logging.getLogger(__name__)


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

ONEHOTENCODER_VARS = ['loan_purpose','applicant_ethnicity']



INTERQUANTILE_OUTLIER_VARS =['loan_amount']

GUASSIAN_OUTLIER_VARS =['applicant_income']

INCOME_LOAN_RATIO_VARS = ['loan_amount','applicant_income']

DROP_FEATURES = ['applicant_race']



NUMERICAL_VARS_WITH_NA = ['applicant_income','population','minority_population_pct',
						'ffiecmedian_family_income','tract_to_msa_md_income_pct',
						'number_of_owner-occupied_units','number_of_1_to_4_family_units']



loan_status_pipe = Pipeline(
	[	
		('numerical_missingness_feature',pp.NumericalMissingnessFeature(variables=NUMERICAL_VARS_FOR_MISSINGNESS_FEAT)),
	 	('numerical_imputer',pp.NumericalImputer(mean_variables=NUMERICAL_VARS_FOR_MEAN_IMPUTING,median_variables=NUMERICAL_VARS_FOR_MEDIAN_IMPUTING)),
	 	('InterquantileOutlier_doctor', pp.InterquantileOutlier(variable=INTERQUANTILE_OUTLIER_VARS)),
	 	('GuassianOutlier_doctor', pp.GaussianOutlier(variable=GUASSIAN_OUTLIER_VARS)),
	 	('onehotencoder',pp.OneHotEncoding(variable=ONEHOTENCODER_VARS)),
	 	('drop_features',pp.DropUnecessaryFeatures(variables_to_drop=DROP_FEATURES)),
	 	('Income_Loan_Ratio_Feature_Generator',pp.IncomeLoanRatioFeatureGenerator(variable=INCOME_LOAN_RATIO_VARS)),
	 	('label_encoding',pp.LabelEncoding(variable=VAR_TO_BE_LABELENCODED)),
	 	#('st_scaler',pp.StdScaler()),
	 	#('pipeline_breakpoint',pp.FinallyTraining()),
	 	('Classifier',GradientBoostingClassifier(n_estimators= 300,learning_rate=0.1,random_state=42))
	]

)




