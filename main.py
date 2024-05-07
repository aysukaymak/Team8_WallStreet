# main.py

# Import functions from each module
from load_data import load_dataset
from overview_data import overview_data
from eda import perform_eda
from feature_engineering import perform_feature_engineering
from data_preprocessing import preprocess_data
from feature_selection import select_features
from hyperparameter_tuning import tune_hyperparameters
from optuma_weights_model import build_optuma_weights_model

def main():
    # Load the dataset
    data = load_dataset()
    
    # Overview of the data
    overview_data(data)
    
    # Perform EDA
    perform_eda(data)
    
    # Perform feature engineering
    data = perform_feature_engineering(data)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Select important features
    selected_features = select_features(X_train, y_train)
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train)
    
    # Build and train the Optuma weights model
    model = build_optuma_weights_model(best_params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

if __name__ == "__main__":
    main()
