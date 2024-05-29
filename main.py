# main.py
import pandas as pd

# Import functions from each module
from load_data import load_dataset
from overview_data import overview_data
from eda import perform_eda
from feature_engineering import perform_feature_engineering
from data_preprocessing import preprocess_data
from feature_selection import select_features
from hyper_parameter_tuning import tune_hyperparameters
from optuma_weights_model import build_optuma_weights_model

def main():
    # Load the dataset
    train_data, test_data, original_data = load_dataset()
    
    # Overview of the data
    overview_data(train_data, test_data, original_data)
    
    # Perform EDA
    perform_eda(train_data, test_data, original_data)
    
    # Preprocess/feature engineering the data
    train_data_combined, test_data_combined, train_data, test_data= preprocess_data(train_data, test_data, original_data)
    
    # Select important features
    xgb_selected_features, lgb_selected_features, cb_selected_features, xgb_model, lgb_model, cat_model = select_features(train_data_combined, test_data_combined, train_data)
    
    # Tune hyperparameters and build/train the optuma weights model
    ensemble_pred_proba=tune_hyperparameters(train_data_combined, test_data_combined, xgb_selected_features, lgb_selected_features, cb_selected_features, xgb_model, lgb_model, cat_model)
    
    # Assuming 'test_data_combined' is the DataFrame for the test set
    ensemble_predictions = pd.DataFrame({
        'id': test_data['id'],
        'Exited': ensemble_pred_proba  # Fill in the predicted probabilities
    })

    # Save the submission DataFrame to a CSV file
    ensemble_predictions.to_csv('outputs/predictions.csv', index=False)
        
if __name__ == "__main__":
    main()
