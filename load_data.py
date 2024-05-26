# Importing Libraries
import pandas as pd

def load_dataset():
    # Loading data files
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    original_data = pd.read_csv('data/original_data.csv')
    return train_data, test_data, original_data
