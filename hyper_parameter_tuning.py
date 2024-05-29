# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy import stats

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, f1_score, recall_score, precision_score
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


def tune_hyperparameters(train_data_combined, test_data_combined, xgb_selected_features, lgb_selected_features, cb_selected_features, xgb_model, lgb_model, cat_model):
    # Initialize empty lists to store results
    cat_results = []
    xgb_results = []
    lgb_results = []
    
    import optuna
    
    # Catboost Optuna Hyperparameter Tuning
    # Assuming 'X' is your feature matrix and 'y' is your target variable
    X = train_data_combined[cb_selected_features].drop('Exited', axis=1)
    y = train_data_combined['Exited']

    def objective_cat(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2, log=True),
            'random_state': 42,
            'verbose': 0,
            'eval_metric': 'AUC',
        }

        model = CatBoostClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        
        # Append results to list
        cat_results.append({
            'trial': trial.number,
            'iterations': params['iterations'],
            'depth': params['depth'],
            'min_data_in_leaf': params['min_data_in_leaf'],
            'learning_rate': params['learning_rate'],
            'AUC': auc,
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            'Accuracy': acc
        })
        
        print(f"AUC: {auc}, F1: {f1}, Recall: {recall}, Precision: {precision}, Accuracy: {acc}")
        
        return auc  # Change this to optimize another metric if needed
    
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(objective_cat, n_trials=100)
    
    # Convert results to DataFrame
    cat_results_df = pd.DataFrame(cat_results)
    cat_results_df.to_csv('outputs/cat_results.csv', index=False)

    """print('CatBoost Best trial:')
    trial_cat = study_cat.best_trial
    print('Value: ', trial_cat.value)
    print('Params: ')
    for key, value in trial_cat.params.items():
        print(f'    {key}: {value}')"""
        
    # XGB Optuna Hyperparameter Tuning
    X = train_data_combined[xgb_selected_features].drop('Exited', axis=1)
    y = train_data_combined['Exited']

    def objective_xgb(trial):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'random_state': trial.suggest_categorical('random_state', [42]),
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'auc',
            'verbosity': 2,
        }

        model = XGBClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        # Append results to list
        xgb_results.append({
            'trial': trial.number,
            'max_depth': params['max_depth'],
            'min_child_weight': params['min_child_weight'],
            'learning_rate': params['learning_rate'],
            'n_estimators': params['n_estimators'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'AUC': auc,
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            'Accuracy': acc
        })
        
        print(f"AUC: {auc}, F1: {f1}, Recall: {recall}, Precision: {precision}, Accuracy: {acc}")
        
        return auc  # Change this to optimize another metric if needed

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=100)
    
    # Convert results to DataFrame
    xgb_results_df = pd.DataFrame(xgb_results)
    xgb_results_df.to_csv('outputs/xgb_results.csv', index=False)

    """print('XGBoost Best trial:')
    trial_xgb = study_xgb.best_trial
    print('Value: ', trial_xgb.value)
    print('Params: ')
    for key, value in trial_xgb.params.items():
        print(f'    {key}: {value}')"""
        
    # LGB Optuna Hyperparameter Tuning    
    X = train_data_combined[lgb_selected_features].drop('Exited', axis=1)
    y = train_data_combined['Exited']
    
    import optuna
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    def objective_lgb(trial):
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

        model = LGBMClassifier(**params)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        # Append results to list
        lgb_results.append({
            'trial': trial.number,
            'max_depth': params['max_depth'],
            'min_child_samples': params['min_child_samples'],
            'learning_rate': params['learning_rate'],
            'n_estimators': params['n_estimators'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'AUC': auc,
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            'Accuracy': acc
        })
        
        print(f"AUC: {auc}, F1: {f1}, Recall: {recall}, Precision: {precision}, Accuracy: {acc}")
        
        return auc  # Change this to optimize another metric if needed

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=100)
    
    # Convert results to DataFrame
    lgb_results_df = pd.DataFrame(lgb_results)
    lgb_results_df.to_csv('outputs/lgb_results.csv', index=False)

    """print('LightGBM Best trial:')
    trial_lgb = study_lgb.best_trial
    print('Value: ', trial_lgb.value)
    print('Params: ')
    for key, value in trial_lgb.params.items():
        print(f'    {key}: {value}')"""
        
    # Combine all results into one DataFrame
    all_results_df = pd.concat([cat_results_df, xgb_results_df, lgb_results_df], keys=['CatBoost', 'XGBoost', 'LightGBM'], names=['Model'])
    all_results_df.to_csv('outputs/all_results.csv', index=False)

    # Build ensemble model
    ensemble_pred_proba = build_optuma_weights_model(train_data_combined, test_data_combined, xgb_model, xgb_selected_features, lgb_model, lgb_selected_features, cat_model, cb_selected_features)
    return ensemble_pred_proba
