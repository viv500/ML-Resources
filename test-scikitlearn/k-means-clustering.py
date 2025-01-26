import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# an UNSUPERVISED ML model
# learns from a cluster of data points
# goal : minimize the distance between center of cluster, and its data points

# step 1: define number of clusters you want
# step 2: k random points are selected on the data set
# step 3: calculate euclidean disance between a clusters center point and all other points
# step 4: all points get assigned to a specific cluster
# step 5: calculate mean point of each cluster (i.e. average coordinates of all points within a cluster)
# step 6: after finding k mean points, treat it as the k random points you selected in step 2 and repeat
# step 7: once mean points stop changin, algorithm ends


# EBLOW PLOT: (Find image) Chosing K is usually an elbow plot with worst performance (sum of squared distances) for k = 1 and
#  a huge imporvement for k = 2 and then gradual improvement

df = pd.read_csv('test-scikitlearn/force2020_data_unsupervised_learning.csv', index_col = 'DEPTH_MD')

print(df.tail())

# filling missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(df)

# but this is now a numpy array , we need to convert back to a df

df = pd.DataFrame(imputed_data, columns=['RHOB', 'GR', 'NPHI', 'PEF', 'DTC'])
df.index = pd.Index(df.index, name='DEPTH_MD') # to set back the index column

# or couldve done df.dropna(inplace=True)


# we need to scale cuz some features have their own units which may misinterpret variance to the model
scale = StandardScaler()

scaled_data = scale.fit_transform(df)
# cnoverting back to a data frame
df = pd.DataFrame(scaled_data, columns=['RHOB', 'GR', 'NPHI', 'PEF', 'DTC'])
df.index = pd.Index(df.index, name='DEPTH_MD') # to set back the index column


print(df)


k_means = KMeans(n_clusters=3)

# fitting it to our data
k_means.fit(df[['NPHI', 'RHOB']])  # output was an error cuz we didnt look 
# only need fit and not transform cuz the model only need to fit (train) on it, not modify it like for scaling

# output of k means: create a new fields with the clustering
df['k_means_3'] = k_means.labels_

print(df)

X = df['RHOB']
y = df['NPHI']




plt.scatter(X, y, c=df['k_means_3']) # colour coding based on clusters
plt.show()