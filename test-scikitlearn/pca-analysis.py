import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# not every feature has much variance -> i.e. don't effect the model much
# PCA (Principle Component Analysis) transfoms less important features into new components
# This makes your model run faster and could prevent overfitting by eliminating noise


df = pd.read_csv("./500hits.csv")
print(df)

df.drop('PLAYER', axis=1, inplace=True)
print(df)

X = df.iloc[:, 0:14]
y = df.iloc[:, 14]

print(y)

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

df = pd.DataFrame(X_train, 
                  columns=['YRS', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB',
                            'CS', 'BA'])

print(df)

lr1 = LogisticRegression()
lr1.fit(X_train, y_train)


print("Pre PCA Score: ", lr1.score(X_test, y_test))

pca1 = PCA()
X_pca1 = pca1.fit_transform(X_train)

print(pca1.explained_variance_ratio_)
# orders features from most variance to least variance

# The explained variance ratio tells you how much variance each principal component explains
# It helps in identifying the most important features (or components).
# 8. Applying PCA with 95% explained variance ratio
pca2 = PCA(0.95) # Specify that we want to retain 95% of the total variance

X_pca2 = pca2.fit_transform(X_train)

# applying it to the TEST data too
X_test_pca = pca2.transform(X_test)

lr2 = LogisticRegression()
lr2.fit(X_pca2, y_train)

print("Post PCA Score: ", lr2.score(X_test_pca, y_test))