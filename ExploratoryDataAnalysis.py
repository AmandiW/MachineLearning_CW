import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
df = pd.read_csv("MLDataset.csv")

# Performing a quick exploration
print(df.head())
print(df.info)
print(df.describe())
print(df.shape)

# Check and count missing data and null values
total_missing = df.isnull().sum().sum()
total_null = df.isna().sum().sum()
print("Total missing values: ", total_missing)
print("Total null values: ", total_null)
