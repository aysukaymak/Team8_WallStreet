import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
train_rows, train_columns = train_data.shape
test_rows, test_columns = test_data.shape

#Printing number of rows and columns in training data and test data
print("Training Data:")
print(f"Number of Rows: {train_rows}")
print(f"Number of Columns: {train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {test_rows}")
print(f"Number of Columns: {test_columns}\n")

#Column data for training data and test data
print("\nTraining Data Info:")
print(train_data.info())

print("\nTest Data Info:")
print(test_data.info())

#Printing the first five rows of training data and test data
print("\nTraining Data First Five Rows of Information:")
print(train_data.head())

print("\nTest Data First Five Rows of Information:")
print(test_data.head())

#Checking for NULL values in training data and Test data
print("\nTraining Data Null Values:")
null_values_train_data = train_data.isna()
print(null_values_train_data.sum())

print("\nTest Data Null Values:")
null_values_test_data = test_data.isna()
print(null_values_test_data.sum())

#Checking for duplicate values in training data and test data
print("\nTraining Data Duplicate Values:")
print(train_data.duplicated().sum())

print("\nTest Data Duplicate Values:")
print(test_data.duplicated().sum())