import pandas as pd
from sklearn.decomposition import PCA
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer

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
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=22)

# Training the DT Model
DT_clf = DecisionTreeClassifier(random_state=0, criterion='gini')
DT_clf.fit(X_train, y_train)

# Checking accuracy of training and testing data
predictions_test = DT_clf.predict(X_test)
predictions_train = DT_clf.predict(X_train)
accuracy_test = accuracy_score(y_test, predictions_test)
accuracy_train = accuracy_score(y_train, predictions_train)

# Plotting the final Decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(DT_clf, filled=True)
plt.title('DT before Pruning')
# plt.show()

# Evaluating the train dataset
print(classification_report(y_train, predictions_train))
print(confusion_matrix(y_train, predictions_train))
print("Train accuracy before pruning = ", accuracy_train)

# Evaluating the test dataset
print(classification_report(y_test, predictions_test))
print(confusion_matrix(y_test, predictions_test))
print("Test accuracy before pruning = ", accuracy_test)

# Pruning the Decision Tree
path = DT_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Visualizing the impurity of leaves
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

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
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

node_counts = [clf.tree_.node_count for clf in clf_array]
depth = [clf.tree_.max_depth for clf in clf_array]

# Checking accuracy after pruning
train_scores = [clf.score(X_train, y_train) for clf in clf_array]
test_scores = [clf.score(X_test, y_test) for clf in clf_array]

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.grid()

# Fitting the DT Classifier after pruning
clf = DecisionTreeClassifier(random_state=22, ccp_alpha=0.016)
clf.fit(X_train, y_train)

# Plotting the ROC curve
dt_probs = clf.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, dt_probs)
auc_score_dt = auc(fpr_dt, tpr_dt)
print("AUC after pruning = ", auc_score_dt)


# Plotting the ROC curve
def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_dt, tpr_dt, color='orange', label='AUC = %0.2f' % auc_score_dt)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


plot_roc_curve(fpr_dt, tpr_dt)

# Accuracy of test and train data after pruning
prediction_trainP = clf.predict(X_train)
print(classification_report(y_train, prediction_trainP))
print("Train accuracy after pruning = ", accuracy_score(y_train, prediction_trainP))

prediction_testP = clf.predict(X_test)
print(classification_report(y_test, prediction_testP))
print("Test accuracy after pruning = ", accuracy_score(y_test, prediction_testP))

# Visualizing after pruning

plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True)
plt.title("DT after pruning")
plt.show()
