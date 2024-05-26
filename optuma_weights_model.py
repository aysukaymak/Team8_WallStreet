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

import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 

def build_optuma_weights_model(train_data_combined, test_data_combined,xgb_model, xgb_selected_features, lgb_model, lgb_selected_features, cat_model, cb_selected_features):
    # Define objective function for Optuna to optimize
    X = train_data_combined.drop('Exited', axis=1)
    y = train_data_combined['Exited']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model.fit(X_train[[feature for feature in xgb_selected_features if feature != 'Exited']], y_train, verbose=0)
    lgb_model.fit(X_train[[feature for feature in lgb_selected_features if feature != 'Exited']], y_train)
    cat_model.fit(X_train[[feature for feature in cb_selected_features if feature != 'Exited']], y_train, verbose=0)

    def objective(trial):
        xgb_weight = trial.suggest_uniform('xgb_weight', 0, 1)
        lgb_weight = trial.suggest_uniform('lgb_weight', 0, 1)
        cat_weight = trial.suggest_uniform('cat_weight', 0, 1)

        # Normalize weights to sum up to 1
        total_weight = xgb_weight + lgb_weight + cat_weight
        xgb_weight /= total_weight
        lgb_weight /= total_weight
        cat_weight /= total_weight

        # Ensemble predictions
        ensemble_pred_proba = (
            xgb_weight * xgb_model.predict_proba(X_test[[feature for feature in xgb_selected_features if feature != 'Exited']])[:, 1] +
            lgb_weight * lgb_model.predict_proba(X_test[[feature for feature in lgb_selected_features if feature != 'Exited']])[:, 1] +
            cat_weight * cat_model.predict_proba(X_test[[feature for feature in     cb_selected_features if feature != 'Exited']])[:, 1]
        )

        # Assuming y_test is available
        auc_score = roc_auc_score(y_test, ensemble_pred_proba)

        return auc_score

    # Optimize using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Get the best weights
    best_weights = study.best_params
    xgb_weight = best_weights['xgb_weight']
    lgb_weight = best_weights['lgb_weight']
    cat_weight = best_weights['cat_weight']
    
    total_weight = xgb_weight + lgb_weight + cat_weight
    xgb_weight /= total_weight
    lgb_weight /= total_weight
    cat_weight /= total_weight
    
    print('xgb_weight: ',xgb_weight)
    print('lgb_weight: ',lgb_weight)
    print('cat_weight: ',cat_weight)

    xgb_model.fit(train_data_combined[xgb_selected_features].drop('Exited', axis=1), y)
    xgb_pred_proba = xgb_model.predict_proba(test_data_combined[[feature for feature in xgb_selected_features if feature != 'Exited']])[:, 1]

    lgb_model.fit(train_data_combined[lgb_selected_features].drop('Exited', axis=1), y)
    lgb_pred_proba = lgb_model.predict_proba(test_data_combined[[feature for feature in lgb_selected_features if feature != 'Exited']])[:, 1]

    cat_model.fit(train_data_combined[cb_selected_features].drop('Exited', axis=1), y)
    cb_pred_proba = cat_model.predict_proba(test_data_combined[[feature for feature in cb_selected_features if feature != 'Exited']])[:, 1]

    #ensemble_pred_proba = (xgb_pred_proba * 0) + (lgb_pred_proba * 0) + (cb_pred_proba * 1) 
    ensemble_pred_proba = (xgb_pred_proba * xgb_weight) + (lgb_pred_proba * lgb_weight) + (cb_pred_proba * cat_weight) 
    
    return ensemble_pred_proba
