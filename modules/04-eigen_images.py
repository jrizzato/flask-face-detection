import joblib
from sklearn.decomposition import PCA

X, y = joblib.load("./data/data_10000_norm.joblib")

print(type(X))
print(X[:10])
print(y[:10])

print(X.shape)
print(y.shape)

# now we will apply the PCA to this data
# PCA uses eigenvalues and eigenvectors
# eigenvalues denote the variance, eigenvectors represent the direction
# the values come from the covariance matrix
# so, larger the variance, larger the information available in that particular direction
# the idea of PCA is to identify the directions with the largest variance
# PCA reduces dimensionality by keeping only the principal components with high variance
# and discarding components with low variance (which contain less information)

X1 = X - X.mean(axis=0)
pca = PCA(n_components=None, whiten=True, svd_solver='randomized')
 
print(type(pca))

x_pca = pca.fit_transform(X1)
print(type(x_pca))
print(x_pca.shape)

# this takes a lot of time, so i will save the x_pca and continue in another module

joblib.dump(x_pca, './data/x_pca.joblib')
joblib.dump(pca, './data/pca.joblib')