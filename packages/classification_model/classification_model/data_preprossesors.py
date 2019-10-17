import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder , StandardScaler
#for testing purposes 
from classification_model.config import config




class NumericalMissingnessFeature(BaseEstimator, TransformerMixin):
	"""
	Numerical missingness feature creator.
	
	"""

	def __init__(self, variables=None):
		if not isinstance(variables, list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit(self, X, y=None):
		
		return self 

	def transform(self, X):
		X = X.copy()
		for feature in self.variables:
			X[feature +str('_NA')]= np.where(X[feature].isnull(),1,0)
		return X


class NumericalImputer(BaseEstimator, TransformerMixin):
	"""Numerical missing value imputer."""

	def __init__(self, mean_variables=None,median_variables=None):
		if not isinstance(mean_variables, list):
			self.mean_variables = [mean_variables]
		else:
			self.mean_variables = mean_variables
		if not isinstance(median_variables, list):
			self.median_variables = [median_variables]
		else:
			self.median_variables = median_variables

	def fit(self, X, y=None):
		# persist mean in a dictionary
		self.mean_imputer_dict_ = {}
		for feature in self.mean_variables:
			self.mean_imputer_dict_[feature] = X[feature].mean()
		
		# persist median in a dictionary
		self.median_imputer_dict_ = {}
		for feature in self.median_variables:
			self.median_imputer_dict_[feature] = X[feature].median()
		return self

	def transform(self, X):
		X = X.copy()
		for feature in self.mean_variables:
			X[feature].fillna(self.mean_imputer_dict_[feature], inplace=True)
		for feature in self.median_variables:
			X[feature].fillna(self.median_imputer_dict_[feature], inplace=True)
		return X


class LabelEncoding(BaseEstimator, TransformerMixin):
	"""String to numbers categorical encoder."""

	def __init__(self, variable=None):
		if not isinstance(variable, list):
			self.variable = [variable]
		else:
			self.variable = variable
		self.enc = LabelEncoder()

	def fit(self, X, y):
		self.enc_dic_ = {}
		for col in self.variable:
			self.enc.fit(X[col])
			self.enc_dic_[col+'_classes']=self.enc.classes_
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		for feature in self.variable:
			self.enc.classes_ = self.enc_dic_[feature+'_classes']
			X[feature] = self.enc.transform(X[feature])

		return X





class OneHotEncoding(BaseEstimator, TransformerMixin):			
	"""String to numbers categorical encoder."""

	def __init__(self, variable=None):
		if not isinstance(variable, list):
			self.variable = [variable]
		else:
			self.variable = variable
			
	def fit(self, X, y):
		self.onehot_dic_ = {}
		for col in self.variable:
			self.onehot_dic_[col+'_fitted'] = OneHotEncoder(sparse=False,dtype=np.uint8)
			self.onehot_dic_[col+'_fitted'].fit((np.array(X[col])).reshape(-1, 1))

		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		for col in self.variable:
			self.heading = list()
			for i in list(self.onehot_dic_[col+'_fitted'].categories_[0]):  #generating the new columns
				self.heading.append(col+'_'+str(i)) 
			dummies = self.onehot_dic_[col+'_fitted'].transform((np.array(X[col])).reshape(-1, 1))
			X.drop(columns=[col],axis=1,inplace=True)
			dummies = pd.DataFrame(data=dummies,columns=self.heading,dtype=np.int)
			X = pd.concat([X,dummies],axis=1)    #concatinating the dummy to X
		X	
		return X


class InterquantileOutlier(BaseEstimator, TransformerMixin):
	"""String to numbers categorical encoder."""

	def __init__(self, variable=None):
		if not isinstance(variable, list):
			self.variable = [variable]
		else:
			self.variable = variable
			

	def fit(self, X, y):

		self.outlier_replace_dic_ = {}
		for col in self.variable:
			self.IQR = X[col].quantile(0.75) - X[col].quantile(0.25)
			self.lower_fence = X[col].quantile(0.25) - (self.IQR * 1.0)
			self.Upper_fence = X[col].quantile(0.75) + (self.IQR * 1.5)
			self.outlier_replace_dic_[col+'_values']=[self.lower_fence,self.Upper_fence]

		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		for col in self.variable:
			X.loc[X[col]>self.Upper_fence,col]= self.outlier_replace_dic_[col+'_values'][1] #replacing this outliers with Upper_fence(which is the boundary obtained by IQR assumption)
			X.loc[X[col]<self.lower_fence,col]= self.outlier_replace_dic_[col+'_values'][0] #replacing this outliers with lower_fence(which is the boundary obtained by IQR assumption)

		return X

class GaussianOutlier(BaseEstimator, TransformerMixin):
	"""String to numbers categorical encoder."""

	def __init__(self, variable=None):
		if not isinstance(variable, list):
			self.variable = [variable]
		else:
			self.variable = variable

	def fit(self, X, y):

		self.outlier_replace_dic_ = {}
		for col in self.variable:
			self.upper_b = X[col].mean() + 3*(X[col].std())
			self.outlier_replace_dic_[col + str('upper')]=self.upper_b

		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		for col in self.variable:
			X.loc[X[col]>self.upper_b,col]= self.outlier_replace_dic_[col + str('upper')] #replacing this outliers with Upper_fence(which is the boundary obtained by Gaussian assumption)

		return X

class IncomeLoanRatioFeatureGenerator(BaseEstimator, TransformerMixin):
	"""String to numbers categorical encoder."""
		#variable = [loan_amount,applicant_income]
	def __init__(self, variable=None):
		if not isinstance(variable, list):
			self.variable = [variable]
		else:
			self.variable = variable

	def fit(self, X, y):

		return self

	def transform(self, X):
		""" applicant income loan ratio feature engineering  """
		X = X.copy()
		X['income_loan_ratio'] = X[self.variable[0]] / (X[self.variable[1]] + 1) 
					
		return X


class StdScaler(BaseEstimator, TransformerMixin):
	"""Scaling all features"""

	def __init__(self, variable=None):
		if not isinstance(variable, list):
			self.variable = [variable]
		else:
			self.variable = variable
			
		self.st_scaler = StandardScaler()

	def fit(self, X, y):
		self.X_cols = list(X.columns)
		self.st_scaler.fit(X)

		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X = pd.DataFrame(self.st_scaler.transform(X))
		X.columns = self.X_cols

		return X

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

	def __init__(self, variables_to_drop=None):
		self.variables = variables_to_drop

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X = X.drop(self.variables, axis=1)

		return X

class FinallyTraining(BaseEstimator, TransformerMixin):
	"""Scaling all features"""

	def __init__(self, variable=None):
		self.message = 'all data checkpoint passed'

	def fit(self, X, y):
		self.training_message = 'Now training, you might wanna go for a coffe break'
		print(self.training_message)
		return self

	def transform(self, X):
		self.testing_message = 'prediction is also possible with this pipeline, you might wanna go for a coffe break'
		print(self.testing_message)
		# print(X.columns)
		# print(X.shape)
		#X.to_csv(config.DATASET_DIR / 'cleaned_X.csv',index=False)
		return X




# class TargetEncoder(BaseEstimator, TransformerMixin):
#     """String to numbers categorical encoder."""

#     def __init__(self, variables=None):
#         if not isinstance(variables, list):
#             self.variables = [variables]
#         else:
#             self.variables = variables

#     def fit(self, X, y):
#         temp = pd.concat([X, y], axis=1)
#         temp.columns = list(X.columns) + ['target']

#         # persist transforming dictionary
#         self.encoder_dict_ = {}

#         for var in self.variables:
#             t = temp.groupby([var])['target'].mean().sort_values(
#                 ascending=True).index
#             self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

#         return self

#     def transform(self, X):
#         # encode labels
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].map(self.encoder_dict_[feature])

#         # check if transformer introduces NaN
#         if X[self.variables].isnull().any().any():
#             null_counts = X[self.variables].isnull().any()
#             vars_ = {key: value for (key, value) in null_counts.items()
#                      if value is True}
#             raise ValueError(
#                 f'Categorical encoder has introduced NaN when '
#                 f'transforming categorical variables: {vars_.keys()}')

#         return X


# class CategoricalImputer(BaseEstimator, TransformerMixin):
# 	"""Categorical data missing value imputer."""

# 	def __init__(self, variables=None) -> None:
# 		if not isinstance(variables, list):
# 			self.variables = [variables]
# 		else:
# 			self.variables = variables

# 	def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategoricalImputer':
			
# 		"""Fit statement to accomodate the sklearn pipeline."""

# 		return self

# 	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
# 		"""Apply the transforms to the dataframe."""

# 		X = X.copy()
# 		for feature in self.variables:
# 			X[feature] = X[feature].fillna('Missing')

# 		return X