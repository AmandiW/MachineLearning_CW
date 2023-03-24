import pandas as pd

# Reading the dataset into a dataframe
df = pd.read_csv("ML_DatasetUpdated.csv")

# Checking the initial dataset shape and info
print("Initial Dataset shape = ", df.shape)

# Checking for missing and null values
print("Missing values in the dataset =", df.isnull().sum())

# Checking for duplicate rows
print("Duplicate rows in the dataset = ", df.duplicated().sum())

# Cleaning the dataset by removing missing values and duplicate rows
clean_data = df.dropna().drop_duplicates()

# Printing the cleaned dataset info
print("Cleaned dataset shape = ", clean_data.shape)

# Saving the cleaned dataset as a new CSV file
clean_data.to_csv("ML_DatasetCleaned.csv", index=False)


