from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


# ================================================
# SIMPLE PIPELINE: Only numeric data
# ================================================

df1 = {'Social_media_followers':[1000000, np.nan, 2000000, 1310000, 1700000, np.nan, 4100000, 1600000, 2200000, 1000000],
    'Sold_out':[1,0,0,1,0,0,0,1,0,1]}

df2 = {'Genre':['Rock', 'Metal', 'Bluegrass', 'Rock', np.nan, 'Rock', 'Rock', np.nan, 'Bluegrass', 'Rock'],
    'Social_media_followers':[1000000, np.nan, 2000000, 1310000, 1700000, np.nan, 4100000, 1600000, 2200000, 1000000],
    'Sold_out':[1,0,0,1,0,0,0,1,0,1]}

df1 = pd.DataFrame(data=df1)
X1 = df1[['Social_media_followers']]
y1 = df1['Sold_out']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=19)


# 1. FILLING IN THE NANS: Simple Imputer
imputer=SimpleImputer(strategy="mean")


lr = LogisticRegression()

  
# MAKING THE PIPELINE
pipe1 = make_pipeline(imputer, lr)

# fit the pipline with your data so it can move it thorugh
pipe1.fit(X1_train, y1_train) 

print(pipe1.score(X1_train, y1_train)) # score of the entire pipline
# 100%: duh its good at this, it was trained at it

print(pipe1.score(X1_test, y1_test))
# 66%