import re
import pandas as pd

# Read feature names from file and store in a list
with open("C:\\Users\\HP\\Downloads\\spambase\\spambase.names") as spam:
    text = spam.read()
    feature_names = re.findall(r'\n(\w*_?\W?):', text)

# Add "spam" column name to feature names list
feature_names.append('spam')

# Load the data from file using pandas, using feature names as column headers
data = pd.read_csv("C:\\Users\\HP\\Downloads\\spambase\\spambase.data", header=None, names=feature_names)

# Print the first few rows of the dataset to check if it was combined properly
print(data.head())
print(data.shape)

# Save the updated dataset to a CSV file
data.to_csv("C:\\Users\\HP\\Downloads\\MachineLearning_CW\\MachineLearning_CW\\ML_DatasetUpdated.csv", index=False)

