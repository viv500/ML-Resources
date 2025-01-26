from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Logistic Regression Explanation

# 1. **Purpose**
#    - Logistic Regression is used for binary classification tasks (0 or 1, Yes or No).
#    - It models the probability of the target variable being in one class.

# 2. **Sigmoid Function**
#    - Logistic Regression uses the sigmoid function (S shaped) to convert the output into a probability.
#    - The sigmoid function outputs a value between 0 and 1.
#    - Formula: S(z) = 1 / (1 + exp(-z)), where z is mx + b

# 3. **Mathematical Model**
#    - Logistic Regression models the probability of class 1 as: P(y = 1 | X) = σ(w0 + w1*x1 + w2*x2 + ... + wn*xn)
#    - Where:
#        - P(y = 1 | X) is the probability that y = 1 given input features X.
#        - σ is the sigmoid function, and w0, w1, ..., wn are weights for the features.

# 4. **Training the Model**
#    - The model is trained by minimizing the **log-loss (cross-entropy loss)** function.
#    - Log-loss: L = - (y * log(ŷ) + (1 - y) * log(1 - ŷ))
#        - y is the actual label, and ŷ is the predicted probability.

# 5. **Decision Boundary**
#    - The model predicts class 1 if the probability is >= 0.5, else class 0.
#    - The decision boundary is based on the threshold value (commonly 0.5).


d = {'miles_per_week': [37,39,46,51,88,17,18,20,21,22,23,24,25,27,28,29,30,31,32,33,34,38,40,42,57,68,35,36,41,43,45,47,49,50,52,53,54,55,56,58,59,60,61,63,64,65,66,69,70,72,73,75,76,77,78,80,81,82,83,84,85,86,87,89,91,92,93,95,96,97,98,99,100,101,102,103,104,105,106,107,109,110,111,113,114,115,116,116,118,119,120,121,123,124,126,62,67,74,79,90,112],
      'completed_50m_ultra': ['no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','no','yes','yes','yes','yes','no','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','no','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes','yes',]}
df = pd.DataFrame(data=d)

finished_race = ['no', 'yes']
enc = OrdinalEncoder(categories=[finished_race])
df['completed_50m_ultra'] = enc.fit_transform(df[['completed_50m_ultra']])

print(df)

# plt.scatter(df['miles_per_week'], df['completed_50m_ultra'])
# plt.show()

# sns.countplot(x='completed_50m_ultra', data=df)


# splitting data
X = df.iloc[:, 0:1]
y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

print(logistic.score(X_test, y_test))
# 90.4%
