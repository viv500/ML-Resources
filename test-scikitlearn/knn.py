# ================================
# KNN model (K Nearest Neighbors)
# ================================

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# A K-Nearest Neighbors (KNN) model is a simple, non-parametric algorithm used for classification and regression. 
# It works by finding the 'k' closest data points (neighbors) to a new data point and predicting the output based on the majority class (for classification) or average (for regression) of those neighbors. 
# The closer the neighbors are, the more influence they have on the prediction.

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
