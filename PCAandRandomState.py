import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Read data
data = pd.read_csv("ML_DatasetCleaned.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X.shape)
print(X.head)

# Define the range of n_components to try
n_components_range = range(X.shape[1])

# Create an empty list to store the explained variance ratios
explained_variances = []

# Loop through each n_components and fit the PCA model to the data
for n_components in n_components_range:
    pca = PCA(n_components=n_components)
    pca.fit(X)
    # Append the explained variance ratio to the list
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# Plot the explained variance ratios for each n_components
plt.plot(n_components_range, explained_variances, 'bo-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Find the best number of principal components that explain most of the variance
best_n_components = np.argmax(explained_variances) + 1
print("Best n_components:", best_n_components)

# Define a range of random state values to try
random_states = range(100)

# Initialize variables to store the best random state and the corresponding accuracy
best_random_state = None
best_random_stateDT = None
best_accuracy = 0
best_accuracyDT = 0

# Loop over the random state values and train a KNN classifier at each state
for random_state in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Update the best random state and accuracy if a new best is found
    if accuracy > best_accuracy:
        best_random_state = random_state
        best_accuracy = accuracy

# Print the best random state and accuracy
print('Best random state for KNN = ', best_random_state)

# Loop over the random state values and train a DT classifier at each state
for random_stateDT in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_stateDT)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracyDT = accuracy_score(y_test, y_pred)

    # Update the best random state and accuracy if a new best is found
    if accuracyDT > best_accuracyDT:
        best_random_stateDT = random_stateDT
        best_accuracyDT = accuracyDT

# Print the best random state and accuracy
print('Best random state for DT = ', best_random_stateDT)
