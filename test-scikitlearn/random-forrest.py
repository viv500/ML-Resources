from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Random Forest:
# --------------
# A Random Forest is an ensemble learning method that creates multiple decision trees during training.
# Each tree is trained on a random subset of the training data and features, and during prediction,
# the final result is obtained by averaging the outputs of all the trees (for regression) or by majority voting (for classification).
# It helps in reducing overfitting compared to a single decision tree by combining multiple trees' predictions.

# Key Features:
# 1. It reduces overfitting by averaging out the results from multiple trees.
# 2. It can handle both classification and regression tasks.
# 3. It is more robust and less sensitive to noisy data than a single decision tree.

# Random Forest uses the concept of decision trees but is more powerful and accurate due to its ensemble approach.


# Ensemble:
# ---------
# An ensemble method combines the predictions of multiple models to create a stronger model.
# The idea is that combining multiple models, such as decision trees, can reduce the risk of overfitting 
# and lead to better generalization on unseen data.

# Types of ensemble methods:
# 1. Bagging (e.g., Random Forest): Builds multiple models in parallel, each trained on a random subset of the data, 
#    and aggregates their predictions (e.g., by voting or averaging).
# 2. Boosting (e.g., AdaBoost, Gradient Boosting): Trains models sequentially, with each model focusing on correcting 
#    the errors made by the previous ones.
# 3. Stacking: Combines the outputs of several base models using a higher-level model to make the final prediction.

# Ensemble methods can outperform individual models by leveraging the diversity of multiple models.
import pandas as pd

df = pd.read_csv("./500hits.csv", encoding = "latin-1")
df = df.drop(columns=['CS', 'PLAYER'])

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(rf.score(X_test, y_test)) # the rf calculates its predictions and compares it to the actual values
# 84%

features = pd.DataFrame(rf.feature_importances_, index = X.columns)
features.iloc[:, 0] = features.iloc[:, 0] * 100 # HOW TO MODIFY A DATAFRAME\

print(features)


# Hyperparameters:
# ----------------
# Hyperparameters are the configuration values that control how a machine learning model is trained.
# They are set before training and are not learned from the data. 
# Examples include the learning rate, number of trees in a random forest, and the number of neighbors (k) in KNN.

# Common Hyperparameters:
# 1. **Learning rate**: Determines how quickly the model updates during training.
# 2. **Number of trees** (Random Forest): Specifies how many decision trees to use in the ensemble.
# 3. **Number of neighbors** (KNN): Defines how many nearest neighbors should be considered for prediction.
# 4. **Max depth** (Decision Trees): Limits the depth of the tree to prevent overfitting.
# 5. **Batch size** (Neural Networks): The number of training samples used in one iteration.

# Hyperparameter Tuning:
# ----------------------
# Hyperparameter tuning is the process of selecting the best combination of hyperparameters for a model.
# It involves adjusting hyperparameters to optimize the model's performance.

# Methods for Hyperparameter Tuning:
# 1. **Grid Search**: Tries every possible combination of hyperparameters from a predefined grid.
# 2. **Random Search**: Randomly selects a subset of hyperparameter combinations to evaluate.
# 3. **Bayesian Optimization**: Uses probabilistic models to predict and find optimal hyperparameters more efficiently.

# Importance of Hyperparameter Tuning:
# - **Improves model performance** by selecting the best set of hyperparameters.
# - Can lead to better accuracy, reduced overfitting, and faster training.
# - Requires careful evaluation and cross-validation to prevent overfitting the training data.

rf2 = RandomForestClassifier(n_estimators = 1000,
                             criterion = 'entropy',
                             min_samples_split = 10,
                             max_depth = 6,
                             random_state = 42)

# Splitting and Impurity in Decision Trees:
# -----------------------------------------
# In decision trees, splitting refers to the process of dividing the data into subsets based on certain feature conditions.
# Impurity is a measure of how mixed the data points are within a subset, which determines how useful that split is for classification.

# Splitting:
# ----------
# - The process where a node in the decision tree is split into two or more child nodes based on feature values.
# - The goal of splitting is to separate the data in a way that maximizes the "purity" of the resulting subsets, i.e., each subset should ideally belong to a single class.
# - The decision tree evaluates multiple potential splits (based on different features and threshold values) and selects the one that improves the classification.

# Impurity:
# --------
# - Impurity measures how mixed or impure the data points are within a node. A perfect node is "pure", meaning it contains only data points from a single class.
# - Common measures of impurity include:
#    1. **Gini Index**: Measures the degree of impurity by calculating the probability of a random data point being misclassified.
#    2. **Entropy**: Measures the uncertainty or disorder of the data. It is used to calculate information gain, which is the reduction in uncertainty after the split.
# - Lower impurity values are preferred because they indicate that the resulting subsets are more homogenous.

# Example of how splitting works:
# - A node might be split on a feature such as "age" (e.g., age > 50) to separate the data into two child nodes (one for ages greater than 50, and one for 50 or below).
# - This split reduces the impurity of the nodes, leading to a better decision tree model.

# n_estimators = 1000:
# - Number of decision trees in the random forest. More trees typically result in better performance but with increased computation.

# criterion = 'entropy':
# - The function used to measure the quality of a split. 'entropy' uses information gain (more split purity is better) to decide splits.
#   The alternative is 'gini', which uses the Gini index to measure impurity.

# min_samples_split = 10:
# - The minimum number of samples required to split an internal node. A higher value prevents overfitting by forcing splits only on nodes with more samples.

# max_depth = 14:
# - The maximum depth of each decision tree. Limiting the depth can prevent overfitting by making the tree less complex.

rf2.fit(X_train, y_train)

print(rf2.score(X_test, y_test))
# 89.2% (max depth made a big difference)
