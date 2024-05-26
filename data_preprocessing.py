# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy import stats

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import optuna
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from feature_engineering import perform_feature_engineering

import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 

numerical_variables = ['CreditScore','Age', 'Balance','EstimatedSalary' ]
target_variable = 'Exited'
categorical_variables = ['Geography', 'Gender', 'Tenure','NumOfProducts', 'HasCrCard','IsActiveMember']

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.1)
    Q3 = data[column].quantile(0.9)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    return filtered_data, rows_deleted

def preprocess_data(train_data, test_data, original_data):
    # Drop null values from original_data
    original_data = original_data.dropna()
    original_data.drop('RowNumber', axis=1, inplace=True)
    print(original_data.isnull().sum())

    # Combine original_data with train_data
    train_data = pd.concat([train_data, original_data], axis=0).reset_index(drop=True)
    
    # Perform feature engineering
    train_data, test_data = perform_feature_engineering(train_data, test_data)

    #**** Outlier Detection ****#
    rows_deleted_total = 0
    for column in numerical_variables:
        train_data, rows_deleted = remove_outliers_iqr(train_data, column)
        rows_deleted_total += rows_deleted
        print(f"Rows deleted for {column}: {rows_deleted}")

    print(f"Total rows deleted: {rows_deleted_total}")
    
    #**** Transformation ****#
    # [FOR TRAIN]
    # Identify features with skewness greater than 0.75
    # Get the index of the data to be transformed
    skewed_features = train_data[numerical_variables].skew()[train_data[numerical_variables].skew() > 0.75].index.values

    # Print the list of variables to be transformed
    print("Features to be transformed (skewness > 0.75):")
    print(skewed_features)

    # Apply log1p transformation to skewed features
    train_data[skewed_features] = np.log1p(train_data[skewed_features])

    # [FOR TEST]
    # Identify features with skewness greater than 0.75
    # Get the index of the data to be transformed
    skewed_features = test_data[numerical_variables].skew()[test_data[numerical_variables].skew() > 0.75].index.values

    # Print the list of variables to be transformed
    print("Features to be transformed (skewness > 0.75):")
    print(skewed_features)

    # Apply log1p transformation to skewed features
    test_data[skewed_features] = np.log1p(test_data[skewed_features])

    #**** Feature Encoding ****#
    # Selecting specific columns for encoding
    columns_to_encode = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember','Geo_Gender','Customer_Status']
    train_data_to_encode = train_data[columns_to_encode]
    test_data_to_encode = test_data[columns_to_encode]

    # Dropping selected columns for scaling
    train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
    test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

    # Use pandas get_dummies to one-hot encode 'Geography' and 'Gender' in train_data
    train_data_encoded = pd.get_dummies(train_data_to_encode, columns=['Geography', 'Gender','NumOfProducts', 'HasCrCard','IsActiveMember','Geo_Gender','Customer_Status'], drop_first=True)

    # Use pandas get_dummies to one-hot encode 'Geography' and 'Gender' in test_data
    test_data_encoded = pd.get_dummies(test_data_to_encode, columns=['Geography', 'Gender','NumOfProducts', 'HasCrCard','IsActiveMember','Geo_Gender','Customer_Status'], drop_first=True)
    
    print(train_data_encoded.head())
    print(test_data_encoded.head())
    train_data_encoded.to_csv('outputs/encoded_train_data.csv', index=False)
    test_data_encoded.to_csv('outputs/encoded_test_data.csv', index=False)


    #**** Feature Scaling ****#
    # Initialize MinMaxScaler
    minmax_scaler = MinMaxScaler()

    # Fit the scaler on the training data
    minmax_scaler.fit(train_data_to_scale.drop(['Exited'], axis=1))

    # Scale the training data
    scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['Exited'], axis=1))
    scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['Exited'], axis=1).columns)

    # Scale the test data using the parameters from the training data
    scaled_data_test = minmax_scaler.transform(test_data_to_scale)
    scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)

    print(scaled_train_df.head())
    print(scaled_test_df.head())
    scaled_train_df.to_csv('outputs/scaled_train_data.csv', index=False)
    scaled_test_df.to_csv('outputs/scaled_test_data.csv', index=False)

    # Concatenate train datasets
    train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

    # Concatenate test datasets
    test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)

    print(train_data_combined.head())
    print(test_data_combined.head())
    return train_data_combined, test_data_combined, train_data, test_data
