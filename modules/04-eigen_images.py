import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

X, y = joblib.load("./data/data_10000_norm.joblib")

print(f'type(X): {type(X)}')
print(X[:10])
print(y[:10])

print(f'X.shape: {X.shape}')
print(f'y.shape: {y.shape}')

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
x_pca = pca.fit_transform(X1)

print(type(X1))
print(type(pca))

eigen_ratio = pca.explained_variance_ratio_
eigen_ratio_cum = np.cumsum(eigen_ratio)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(eigen_ratio[:100], 'r>--')
plt.xlabel('nº of components')
plt.ylabel('explained variance ratio')

plt.subplot(1,2,2)
plt.plot(eigen_ratio_cum[:100], 'r>--')
plt.xlabel('nº of components')
plt.ylabel('cumulative explained variance ratio')

plt.show()

# using elbow method consider number of components in between 25 - 30
# since if i consider component between 25 to 30 the explained variance is around 75%
# so, in order to get min 80% variance i considering 50 components

pca_50 = PCA(n_components=50, whiten=True, svd_solver='randomized')
x_pca_50 = pca_50.fit_transform(X1)

print(f'x_pca_50.shape: {x_pca_50.shape}')

joblib.dump(pca_50, './model/pca_50.joblib')

# consider the 50 component and the inverse transform
x_pca_inv = pca_50.inverse_transform(x_pca_50)

print(x_pca_inv.shape)

# consider one image (one row)
eig_img = x_pca_inv[0,:]
eig_img = eig_img.reshape((100,100))
plt.imshow(eig_img,cmap='gray')
plt.show()

def label(y):
    if y==0:
        return 'Male'
    else:
        return 'Female'
    
np.random.randint(1001)
pics = np.random.randint(0,6058,40)
plt.figure(figsize=(15,8))
for i,pic in enumerate(pics):
    plt.subplot(4,10,i+1)
    img = X[pic:pic+1].reshape(100,100)
    plt.imshow(img,cmap='gray')
    plt.title('{}'.format(label(y[pic])))
    plt.xticks([])
    plt.yticks([])
plt.show()

print("="*20+'Eigen Images'+"="*20)
plt.figure(figsize=(15,8))
for i,pic in enumerate(pics):
    plt.subplot(4,10,i+1)
    img = x_pca_inv[pic:pic+1].reshape(100,100)
    plt.imshow(img,cmap='gray')
    plt.title('{}'.format(label(y[pic])))
    plt.xticks([])
    plt.yticks([])
    
plt.show()

joblib.dump((x_pca_50, y, X.mean(axis=0)), './data/data_pca_50_y_mean.joblib')