import pandas as pd
from sklearn.decomposition import PCA
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
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

# # Train DT model
# knn = DecisionTreeClassifier()
# knn.fit(X_train, y_train)
#
# y_pred = knn.predict(X_test)
#
# print(classification_report(y_test, y_pred))
#
# # Calculate accuracy using accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print("DT Model Accuracy =", accuracy)

# Training the DT Model
DT_clf = DecisionTreeClassifier(random_state=0, criterion='gini')
DT_clf.fit(X_train, y_train)

# Checking accuracy of training and testing data
predictions_test = DT_clf.predict(X_test)
predictions_train = DT_clf.predict(X_train)
accuracy_test = accuracy_score(y_test, predictions_test)
accuracy_train = accuracy_score(y_train, predictions_train)
print(accuracy_test, accuracy_train)

# Plotting the final Decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(DT_clf, filled=True)
# plt.show()

# Evaluating the test dataset
print(classification_report(y_test, predictions_test))
print(confusion_matrix(y_test, predictions_test))

# Evaluating the train dataset
print(classification_report(y_train, predictions_train))
print(confusion_matrix(y_train, predictions_train))

# Pruning the Decision Tree
path = DT_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Visualizing the impurity of leaves
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
# plt.show()

clf_array = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clf_array.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
    clf_array[-1].tree_.node_count, ccp_alphas[-1]))
clf_array = clf_array[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clf_array]
depth = [clf.tree_.max_depth for clf in clf_array]

# Checking accuracy after pruning
train_scores = [clf.score(X_train, y_train) for clf in clf_array]
test_scores = [clf.score(X_test, y_test) for clf in clf_array]

# Fitting the DT Classifier after pruning
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.016)
clf.fit(X_train, y_train)

# Accuracy of test and train data after pruning
prediction_testP = clf.predict(X_test)
print(accuracy_score(y_test, prediction_testP))
prediction_trainP = clf.predict(X_train)
print(accuracy_score(y_train, prediction_trainP))
