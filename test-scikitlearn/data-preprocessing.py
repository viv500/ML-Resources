# 80-20 split

# ================================
# train_test_split
# ================================

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./500hits.csv", encoding = "latin-1") #no idea


# model to predict if a player is in the HOF

X = df.drop(columns=['PLAYER','HOF']) # dont wanna give them that data
y = df['HOF'] # the y value contains the answer

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)
# exact same randomization for the future


# ================================
# Feature Scaling
# ================================

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# converting numerical values to same scale
# 1. Normalization (x - min(x)/(max(x) - min(x)))
# 2. Standardization (x - mean)/std

# We are going to drop Player names cuz we only care about numbers
df2 = pd.read_csv("./500hits.csv", encoding = "latin-1")

df2 = df2.drop(columns=["PLAYER", "CS"])


# the stats are all over the place, we wanna normalize them

X1 = df2.iloc[:, 0:13]
X2 = df2.iloc[:, 0:13]

scaleStandard = StandardScaler()

X1 = scaleStandard.fit_transform(X1) # fit() calculates paramaters (mean and std) needed to scale the data and fits the scaler to X1
# transform() after fitting, it scales data in X! using computed mean and std and returns 
# the SCALED VERSION of x1

X1 = pd.DataFrame(X1, columns=["YRS", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS"])

print(X1.head())
# data is now cleaned up and scaled


# NORMALIZATION: put everything between 0 and 1
minMaxScaler = MinMaxScaler(feature_range=(0, 1)) # to normalize

X2 = minMaxScaler.fit_transform(X2)
X2 = pd.DataFrame(X2, columns=["YRS", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS"])

print(X2.head())

# Normalization is ideal when you're working with bounded ranges or non-Gaussian data, or when you want values in a fixed range like [0, 1].
# Standard Scaling is preferred when data is roughly Gaussian or when you want to ensure the features are on a comparable scale, especially for distance-based or regularization-sensitive algorithms.



# ================================
# OneHotEncoder
# ================================

from sklearn.preprocessing import OneHotEncoder

# good for categorial data
# convert into nummbers !

# Focuses on nominal data: categorical data with no numerical significance (like colours) -> theres no heirarchy
# vs something like sizes which are comparable (small medium large)

d = {'sales': [100000,222000,1000000,522000,111111,222222,1111111,20000,75000,90000,1000000,10000],
      'city': ['Tampa','Tampa','Orlando','Jacksonville','Miami','Jacksonville','Miami','Miami','Orlando','Orlando','Orlando','Orlando'], 
      'size': ['Small', 'Medium','Large','Large','Small','Medium','Large','Small','Medium','Medium','Medium','Small',]}

df = pd.DataFrame(data=d)


# we're working with cities (nominal data)

print(df['city'].unique()) # most popoular cities in order


# One Hot Encoder -> expand the table to have a column for each city name -> 1 if it is the city 0 otherwise

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas') # use pandas for our output
ohetransform = ohe.fit_transform(df[['city']]) #df with just cities and bools

print(ohetransform.head())

df = pd.concat([df, ohetransform], axis=1) # concat sideways
df.drop(columns=['city'], inplace=True)

print(df.head(1))


# ================================
# Simple Imputer
# ================================
import numpy as np
from sklearn.impute import SimpleImputer

# getting rid of null values instead of getting rid of rows

df = pd.DataFrame({"Values": [1, 2, 3, np.nan, 5]})

print(df.head())

impute = SimpleImputer(strategy='mean')

impute.fit_transform(df)

# strategy can also be median, or most frequency
# strategies
#    1. mean
#    2. median
#    3. most frequent
#    4. constant (extra parameter: fill_value=13) useful for dataframers that contain strings
#   * add_indicator = True to add an indicator if you imputed


# Overfitting:
# --------------
# Overfitting occurs when a model learns the details and noise in the training data to such an extent
# that it negatively impacts the performance of the model on new, unseen data.
# This typically happens when the model is too complex, with too many parameters or features,
# and it "memorizes" the training data rather than generalizing from it.

# When to avoid overfitting:
# - If your model performs well on training data but poorly on testing data.
# - When you notice that your model is too complex (e.g., too many features, too deep in a decision tree).

# Underfitting:
# --------------
# Underfitting occurs when a model is too simple to capture the underlying patterns in the data,
# leading to poor performance on both the training and testing data.
# This typically happens when the model is too rigid or not powerful enough (e.g., a linear model for non-linear data).

# When to avoid underfitting:
# - If your model performs poorly on both the training and testing data.
# - When your model is too simple, such as using linear models for data that has a non-linear relationship.

# Balancing overfitting and underfitting is key to creating a model that generalizes well.


