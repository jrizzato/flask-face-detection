import joblib
import numpy as np

df = joblib.load("./data/df_images_100_100.joblib")
print(df.head())

print()
df.info()

print(df.isnull().sum()) # 0, no missing values

# split data
X = df.iloc[:,1:].values    # features
y = df.iloc[:,0].values     #target values (gender)

# print(X.shape)
# print(type(X))

# normalize features min-max scaler
# xnorm = x - xmin / (xmax - xmin)
# print(X.min()) # 0
# print(X.max()) # 255
# so xnorm = x / X.max()
Xnorm = X / X.max()

# print(Xnorm)

# encode target variables: female = 1, male = 0
ynorm = np.where(y=='female',1,0)

# print(ynorm)

# save the data
joblib.dump((Xnorm, ynorm), './data/data_10000_norm.joblib')