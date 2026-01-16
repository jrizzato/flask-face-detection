import joblib

x_pca = joblib.load('./data/x_pca.joblib')
pca = joblib.load('./data/pca.joblib')

print(type(x_pca))
print(x_pca.shape)

print(type(pca))

