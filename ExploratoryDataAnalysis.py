import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
df = pd.read_csv("ML_DatasetCleaned.csv")

# Get the number of rows and columns in the data
print("Rows: ", df.shape[0])
print("Columns: ", df.shape[1])

# Get the first five rows of the dataset
print(df.head())

# Check for missing values in the dataset
print(df.isnull().sum())

# Get the summary of the dataset
print(df.describe())

# Plotting the heatmap using the correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.show()

# Plot boxplot
sns.boxplot(data=df)
plt.show()

# Plot histograms
df.hist(bins=30)
plt.show()


# # Checking and removing outliers using IQR
# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# # The 25th (Q1) and 75th (Q3) percentile is used to find the IQR
# IQR = Q3 - Q1
# # The IQR is used to find the lower limit and upper limit
# lower_limit = Q1 - 1.5 * IQR
# upper_limit = Q3 + 1.5 * IQR
# print(df[(df < lower_limit)| (df > upper_limit)])

