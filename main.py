# main.py

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
    
if __name__ == "__main__":
    main()
