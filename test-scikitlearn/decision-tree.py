import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Look at decisiontree.png

# Decision Tree Classifier:
# --------------------------
# A Decision Tree Classifier is SUPERVISED for both classification and regression tasks.
# It works by splitting the dataset into subsets based on feature values using conditions, 
# forming a tree-like structure where each node represents a decision based on a feature.
# The goal is to create rules that classify data points as accurately as possible.

# When to use it:
# 1. When the data has both numerical and categorical features, as decision trees handle both seamlessly.
# 2. When interpretability is important, as the tree structure provides clear, human-readable rules.
# 3. For datasets with non-linear relationships, as trees can capture complex patterns.

# Limitations:
# - Decision trees are prone to overfitting, especially with noisy or small datasets.
# - They can be unstable as small changes in the data may lead to a completely different tree.
# - Using techniques like pruning, limiting tree depth, or ensemble methods (e.g., Random Forest) can mitigate these issues.


df = pd.read_csv("./500hits.csv", encoding = "latin-1")
df = df.drop(columns=['CS', 'PLAYER'])

X = df.iloc[:, 0:13]
y = df.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4,test_size=0.2)
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print(confusion_matrix(y_test, y_pred))


# IMPORTANCE OF FEATURES FOR MODEL
features = pd.DataFrame(dtc.feature_importances_, index = X.columns)
print(features)