from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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


# CREATING A PARAMETER GRID FOR THE ALGORITHM TO TO ITERATE THROUGH
parameter_grid = [{
    'n_estimators' : [1000, 500, 100],
    'criterion' : ['entropy', 'gini'],
    'min_samples_split' : [10, 20, 5],
    'max_depth' : [6, 10, 4]
}]

rf2 = RandomForestClassifier(n_estimators = 1000,
                            criterion = 'entropy',
                             min_samples_split = 10,
                             max_depth = 6,
                             random_state = 42)

gridsearch = GridSearchCV(rf, 
                          param_grid=parameter_grid, 
                          cv=2, 
                          scoring='accuracy',
                          n_jobs=-1)


gridsearch.fit(X_train, y_train)

# FINDING BEST SCORES AND PARAMETERS
print(gridsearch.best_score_)
print(gridsearch.best_params_)