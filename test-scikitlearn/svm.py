import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #i.e Classifier

# Support Vector Machine (SVM):
# -----------------------------
# Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks.
# SVM works by finding the optimal hyperplane that separates the data points of different classes. 
# The key idea is to maximize the margin, i.e., the distance between the closest data points (support vectors) and the hyperplane.
# It is particularly effective for high-dimensional spaces and cases where the number of dimensions exceeds the number of data points.

# When to use it:
# 1. When the data is linearly separable or can be transformed into a linearly separable space using the kernel trick.
# 2. When dealing with high-dimensional data or small datasets with a clear margin of separation between classes.
# 3. For binary classification tasks, but can be adapted for multi-class classification using strategies like one-vs-one or one-vs-all.

# Key Parameters:
# - **C (Regularization parameter)**: Controls the trade-off between achieving a low error on the training data and maximizing the margin.
#   - High C: Low margin, low bias (might overfit).
#   - Low C: High margin, high bias (might underfit).
# - **Kernel**: Defines the function used to transform the data into a higher-dimensional space.
#   - Common kernels: Linear, Polynomial, Radial Basis Function (RBF), Sigmoid.
# - **Gamma**: Defines the influence of a single training example; higher gamma values mean a higher influence.
#             DISTANCE FROM DIVIDING LINE: high gamma -> poitns closer to line

# Limitations:
# - SVM can be computationally expensive and slow for large datasets due to its quadratic complexity.
# - It may struggle with noisy data or overlapping classes since the margin maximization is sensitive to outliers.
# - Choice of kernel and hyperparameters (like C and gamma) can heavily influence model performance and require careful tuning.



mean1 = 55
std_dev1 = 10
num_samples = 500

column1_numbers = np.random.normal(mean1, std_dev1, num_samples)
column1_numbers = np.clip(column1_numbers, 30, 120)
column1_numbers = np.round(column1_numbers).astype(int)

mean2 = 18
std_dev2 = 3

column2_numbers = np.random.normal(mean2, std_dev2, num_samples)
column2_numbers = np.clip(column2_numbers, 12, 26)
column2_numbers = np.round(column2_numbers).astype(int)

column3_numbers = np.random.randint(2, size=num_samples)
column3_numbers[column1_numbers > mean1] = 1

data = {
    'Miles_Per_week': column1_numbers,
    'Farthest_run': column2_numbers,
    'Qualified_Boston_Marathon': column3_numbers
}


df = pd.DataFrame(data)

plt.scatter(df['Miles_Per_week'], df['Farthest_run'], c=df['Qualified_Boston_Marathon']) # c=df means colour ppoints differently based on made it to Boston or not
# plt.show()

X = df.iloc[:, 0:2]
y = df.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)


# without parameters
svm = SVC()
svm.fit(X_train, y_train)


print(svm.score(X_test, y_test))
# 83%


#CREATING MY DATA
# DOUBLE PARENTHESIS !!
my_test1 = pd.DataFrame([[12, 4]], columns=['Miles_Per_week', 'Farthest_run']) #loser
my_test2 = pd.DataFrame([[41, 24]], columns=['Miles_Per_week', 'Farthest_run']) #loser

print(svm.predict(my_test1))
print(svm.predict(my_test2))



# Regularlization 
model_reg0 = SVC(C=0.1)

# GAMMA
model_gamma0 = SVC(gamma=1)
model_gamma0.fit(X_train, y_train)
print(model_gamma0.score(X_test, y_test))
# 75%


# KERNELS
model_kernal = SVC(kernel='linear') # or rbf
model_kernal.fit(X_train, y_train)
print(model_kernal.score(X_test, y_test))
# 81%

