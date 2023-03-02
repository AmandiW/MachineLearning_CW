import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer

# Load data
data = pd.read_csv("ML_DatasetUpdated.csv")

# Split the data into features and target variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Used a scaler to improve accuracy
scaler = QuantileTransformer()
X_scaled = scaler.fit_transform(X)

# Perform PCA analysis. Used elbow method to find best n_components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=56)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))

# Calculate accuracy using accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy =", accuracy)