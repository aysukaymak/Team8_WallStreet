import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

#Printing top 5 of the datasets
print(train_df.head())
print(test_df.head())

#Having a look at the description and info of all the dataset
print(f"\nDescribe table:\n {train_df.describe()}\n\n")
print(f"Information of table:\n{train_df.info}\n\n")

# Checking the number of rows and columns
print(f"Number of rows and columns of train data:{train_df.shape}\n")
print(f"Number of rows and columns of test data:{test_df.shape}\n\n")

#Creating a table for missing values, unique values and data types of the features
missing_values_train = pd.DataFrame({'Feature': train_df.columns,
                              '[TRAIN] Number of Missing Values': train_df.isnull().sum().values})
missing_values_test = pd.DataFrame({'Feature': test_df.columns,
                             '[TEST] Number of Missing Values': test_df.isnull().sum().values})
unique_values = pd.DataFrame({'Feature': train_df.columns,
                              'Number of Unique Values[FROM TRAIN]': train_df.nunique().values})
feature_types = pd.DataFrame({'Feature': train_df.columns,
                              'DataType': train_df.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')
print(merged_df)

# Count duplicate rows in datasets
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
