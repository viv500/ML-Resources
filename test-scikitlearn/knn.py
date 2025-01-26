# ================================
# KNN model (K Nearest Neighbors)
# ================================

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# K-Nearest Neighbors (KNN):
# ---------------------------
# K-Nearest Neighbors (KNN) is SUPERVISED and used for classification and regression tasks.
# It works by finding the 'k' closest data points (neighbors) to a new data point and making predictions 
# based on the majority class (classification) or average (regression) of those neighbors.

# When to use it:
# 1. When the dataset has a simple structure and when the decision boundary is highly non-linear.
# 2. When you need an intuitive, easy-to-understand model that doesn’t require much training.
# 3. For small to medium-sized datasets, as the algorithm performs poorly with large datasets due to high computation costs.
# 4. When working with data where the relationships between features and output are local, rather than global.

# Limitations:
# - KNN can be slow with large datasets, especially when making predictions as it requires searching through the entire dataset.
# - It is sensitive to irrelevant or redundant features, and feature scaling (like normalization) is important.
# - Choosing the right value for 'k' is crucial: a small 'k' may overfit, while a large 'k' may smooth out important distinctions.


# A type of SUPERVISED machine learning
# Supervised Learning: The model is trained on labeled data (input-output pairs), 
#                      where the goal is to predict the output (target variable) based on the input features.
#                      Examples include classification and regression.

# Unsupervised Learning: The model is trained on unlabeled data, and the goal is to find hidden patterns or structures
#                       in the data. 
#                       Examples include clustering and dimensionality reduction.


df = pd.read_csv("./500hits.csv", encoding = "latin-1")

# Cleanup

# preprocessing: what features shouldnt impact our model?
# get rid of caught stealing, player names etc

df = df.drop(columns=['CS', 'PLAYER'])
print(df.head())

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4,test_size=0.2)

minmaxscaler = MinMaxScaler(feature_range=(0, 1))

X_train = minmaxscaler.fit_transform(X_train)
X_test = minmaxscaler.fit_transform(X_test)

# creating the model
knn = KNeighborsClassifier(n_neighbors = 10) #8 closest player to see if a player is HOF or not

# fit our data to the model
knn.fit(X_train, y_train)

# prediction
y_pred = knn.predict(X_test)
print(y_pred)

print(knn.score(X_test, y_test))   # predicts how similar the X_test scores were to y_text (the actual values)


# CONFUSION MATRIX: false positives, negatives
cm = confusion_matrix(y_test, y_pred)
# matrix: 
# __                                                             __
# | True Class 0 Predicted as 0       True Class 0 Predicted as 1 |
# | True Class 1 Predicted as 0       True Class 1 Predicted as 1 |
# __                                                             __


# cLASSIFICATION REPORT
cr = classification_report(y_test, y_pred)
print(cr)

print(cm)

# Classification Report:

#       Precision: Proportion of correct positive predictions out of all positive predictions. (true positive/ true positive + false positive)
#       Recall: Proportion of actual positives correctly identified by the model.
#       F1-Score: Harmonic mean of precision and recall, balancing both metrics. ( (2 * precision * recall) / precision + recall)
#       Support: Number of actual instances for each class in the dataset.
#       Accuracy: Overall percentage of correctly classified instances across all classes.
#       Macro Avg: Unweighted average of precision, recall, and F1-score across all classes.
#       Weighted Avg: Average of precision, recall, and F1-score weighted by the support for each class.
