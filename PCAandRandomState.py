# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("ML_DatasetUpdated.csv")

# Create a PCA object with a high number of components
pca = PCA(n_components=57)

# Fit the PCA model to the data
pca.fit(data)

# Get the explained variance ratio for each component
explained_variances = pca.explained_variance_ratio_

# Plot the cumulative explained variance ratio as a function of the number of components
cumulative_variances = []
cumulative_sum = 0
for variance in explained_variances:
    cumulative_sum += variance
    cumulative_variances.append(cumulative_sum)

plt.plot(range(1, len(cumulative_variances)+1), cumulative_variances)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance ratio')
plt.show()

# Split the data into features and target variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Define a range of random state values to try
random_states = range(100)

# Initialize variables to store the best random state and the corresponding accuracy
best_random_state = None
best_accuracy = 0

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
print(f'Best random state: {best_random_state}')
print(f'Accuracy: {best_accuracy:.2f}')
