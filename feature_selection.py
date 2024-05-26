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


# Function to display feature importance
def display_feature_importance(model, X, y, top_n=34,percentage=3, plot=False):
    # Fit the model
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Calculate threshold based on percentage of the top feature importance
    threshold = percentage / 100 * feature_importance_df.iloc[0]['Importance']
    
    # Select features that meet the threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
    
    if plot==True:
        # Set seaborn color palette to "viridis"
        sns.set(style="whitegrid", palette="viridis")
    
        # Display or plot the top features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
        plt.title('Feature Importance for {}'.format(type(model).__name__))
        plt.savefig(f'outputs/feature_importance_for{format(type(model).__name__)}.png', bbox_inches='tight')
        
        print("Selected Features at threshold {}%; {}".format(percentage,selected_features))
    
    # Add 'Exited' to the list of selected features
    selected_features.append('Exited')
        
    return selected_features

def select_features(train_data_combined, test_data_combined, train_data):
    # Add the 'Exited' column back to the scaled training data
    train_data_combined['Exited'] = train_data['Exited'].values

    # Select numeric columns
    numeric_cols = train_data_combined.select_dtypes(include=['number', 'bool'])

    # Calculate the correlation matrix
    corr_matrix = numeric_cols.corr()

    # Create a heatmap using Seaborn with smaller font size for annotations
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})
    plt.title('Correlation Plot of Numeric Columns in train_data_combined')
    plt.savefig(f'outputs/correlation_of_highly_correlated_features.png', bbox_inches='tight')

    #Plotting Feature Importance for different % thresholds of acceptable Importance Values
    X = train_data_combined.drop('Exited',axis=1)
    y = train_data_combined['Exited']

    # List of trial percentages
    trial_percentages = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22, 26, 30]
        # List to store AUC scores for each trial percentage
    auc_scores = []

    # List to store selected features for each model and trial percentage
    selected_features_xgb = []
    selected_features_lgb = []
    selected_features_cat = []
    
    # Loop over each trial percentage
    for percentage in trial_percentages:
        # Get selected features for each model
        xgb_selected_features = display_feature_importance(XGBClassifier(), X, y, percentage=percentage)
        lgb_selected_features = display_feature_importance(lgb.LGBMClassifier(), X, y, percentage=percentage)
        cat_selected_features = display_feature_importance(CatBoostClassifier(), X, y, percentage=percentage)

        # Append selected features to the respective lists
        selected_features_xgb.append(xgb_selected_features)
        selected_features_lgb.append(lgb_selected_features)
        selected_features_cat.append(cat_selected_features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit models on training data
        xgb_model = XGBClassifier()
        lgb_model = lgb.LGBMClassifier()
        cat_model = CatBoostClassifier()

        xgb_model.fit(X_train[[feature for feature in xgb_selected_features if feature != 'Exited']], y_train, verbose=0)
        lgb_model.fit(X_train[[feature for feature in lgb_selected_features if feature != 'Exited']], y_train)
        cat_model.fit(X_train[[feature for feature in cat_selected_features if feature != 'Exited']], y_train, verbose=0)

        # Predict probabilities on the test set
        xgb_pred_proba = xgb_model.predict_proba(X_test[[feature for feature in xgb_selected_features if feature != 'Exited']])[:, 1]
        lgb_pred_proba = lgb_model.predict_proba(X_test[[feature for feature in lgb_selected_features if feature != 'Exited']])[:, 1]
        cat_pred_proba = cat_model.predict_proba(X_test[[feature for feature in cat_selected_features if feature != 'Exited']])[:, 1]

        # Calculate AUC scores and append to the list
        auc_xgb = roc_auc_score(y_test, xgb_pred_proba)
        auc_lgb = roc_auc_score(y_test, lgb_pred_proba)
        auc_cat = roc_auc_score(y_test, cat_pred_proba)

        auc_scores.append((auc_xgb, auc_lgb, auc_cat))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting lines for each model
    plt.plot(trial_percentages, [auc[0] for auc in auc_scores], label='XGB', marker='o')
    plt.plot(trial_percentages, [auc[1] for auc in auc_scores], label='LGB', marker='o')
    plt.plot(trial_percentages, [auc[2] for auc in auc_scores], label='CatBoost', marker='o')

    plt.xlabel('Trial Percentages')
    plt.ylabel('AUC Score')
    plt.title('Model Performance for Different Feature Selection Percentages')
    plt.legend()
    plt.savefig(f'outputs/model_performance_for_different_feature_selection.png', bbox_inches='tight')

    xgb_model = XGBClassifier()
    xgb_selected_features = display_feature_importance(xgb_model, X, y, percentage=0, plot=True)

    lgb_model = lgb.LGBMClassifier()
    lgb_selected_features = display_feature_importance(lgb_model, X, y, percentage=2, plot=True)

    cat_model = CatBoostClassifier()
    cb_selected_features = display_feature_importance(cat_model, X, y, percentage=0,  plot=True)

    return xgb_selected_features, lgb_selected_features, cb_selected_features, xgb_model, lgb_model, cat_model
