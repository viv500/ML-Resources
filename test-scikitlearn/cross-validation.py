import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# CROSS VALIDATION is splitting your data set into different subsets 
# of testing and training, to see which one gives you the most optimal score

np.random.seed(42)
fastball_speed = np.random.randint(90, 106, size=500)

# surgery
# CONDITIONAL STATEMENT
tommy_john = np.where(fastball_speed > 96, np.random.choice([0, 1], size=500, p=[0.3, 0.7]), 0)
# if a pitcher throws over 96, make it a 30% chance they get it, else 0
print(tommy_john)

d = {'fastball_speed': fastball_speed, 'tommy_john': tommy_john}

df = pd.DataFrame(data=d)
print(df)

X = df.loc[:,['fastball_speed']] # if i wanted more than 1, i could do [:,['', '', '', '']]
y = df.loc[:,['tommy_john']]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

lr = LogisticRegression()

lr.fit(X_train, y_train.values.ravel()) # need to flatten 

print(lr.score(X_test, y_test))




# CROSS VALIDATION
cv_score = cross_val_score(lr, X, y.values.ravel(), cv=10) # i.e. 10 results
print(cv_score)

print(np.mean(cv_score))