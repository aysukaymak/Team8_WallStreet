import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

#load train data
df_train = pd.read_csv("data/train.csv")

#summary of records
print(df_train.info())

#first 5 rows of dataset
print(df_train.head())


print("\nNumber of rows and columns of dataset:")
print(df_train.shape)

# Get statistical summary of data
print("\nStatistical summary of data:")
print(df_train.describe())

# Check missing values in your data
print("\nMissing values in the dataset:")
print(df_train.isnull().sum())

#Count duplicate rows in datasets
print("\nPrint Number of duplicate records:")
train_duplicates = df_train.duplicated().sum()
print(train_duplicates)