import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("ML_DatasetCleaned.csv")

# Split the data into features and target variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Used a scaler to improve accuracy
scaler = QuantileTransformer()
X_scaled = scaler.fit_transform(X)

# Perform PCA analysis. Used elbow method to find best n_components
pca = PCA(n_components=57)
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=14)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on train set and print evaluation metrics
y_pred_train = knn.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(classification_report(y_train, y_pred_train))
print("Train set accuracy:", accuracy_train)
print("")
# Predict on test set and print evaluation metrics
y_pred_test = knn.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(classification_report(y_test, y_pred_test))
print("Test set accuracy:", accuracy_test)


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_test)
print("")
print("AUC =", auc)
