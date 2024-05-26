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

from optuma_weights_model import build_optuma_weights_model

import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 

# Set a seed for reproducibility
seed = 42

# Initialize all the classification models in the requested format
log_reg = LogisticRegression(random_state=seed, max_iter=1000000)
svc = SVC(random_state=seed, probability=True)
lda = LinearDiscriminantAnalysis()
gnb = GaussianNB()
bnb = BernoulliNB()
knn = KNeighborsClassifier()
gauss = GaussianProcessClassifier(random_state=seed)
rf = RandomForestClassifier(random_state=seed)
et = ExtraTreesClassifier(random_state=seed)
xgb = XGBClassifier(random_state=seed)
lgb = LGBMClassifier(random_state=seed)
dart = LGBMClassifier(random_state=seed, boosting_type='dart')
cb = CatBoostClassifier(random_state=seed, verbose=0)
gb = GradientBoostingClassifier(random_state=seed)
hgb = HistGradientBoostingClassifier(random_state=seed)


def tune_hyperparameters(train_data_combined, test_data_combined,xgb_selected_features, lgb_selected_features, cb_selected_features, xgb_model, lgb_model, cat_model):
    # Catboost Optuna Hyperparameter Tuning
    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[cb_selected_features].drop('Exited', axis=1)
    y = train_data_combined['Exited']

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2,    log=True),
            'random_state': 42,
            'verbose': 0,
            'eval_metric': 'AUC',
        }

        model = CatBoostClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # XGB Optuna Hyperparameter Tuning
    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[xgb_selected_features].drop('Exited', axis=1)
    y = train_data_combined['Exited']

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'random_state': trial.suggest_categorical('random_state', [42]),
            'tree_method': 'hist',  # Use GPU for training
            'device': 'cuda',
            'eval_metric': 'auc',  # Evaluation metric
            'verbosity': 2,  # Set verbosity to 0 for less output
        }

        model = XGBClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # LGB Optuna Hyperparameter Tuning
    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[lgb_selected_features].drop('Exited', axis=1)
    y = train_data_combined['Exited']

    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
        }

        model = lgb.LGBMClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
        
    ensemble_pred_proba=build_optuma_weights_model(train_data_combined, test_data_combined,xgb_model, xgb_selected_features, lgb_model, lgb_selected_features, cat_model, cb_selected_features)
    return ensemble_pred_proba
