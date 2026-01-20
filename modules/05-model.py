import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X, y, mean = joblib.load("./data/data_pca_50_y_mean.joblib")

print(f'type X_pca_5a: {type(X)}')
print(f'type y: {type(y)}')
print(f'type X: {type(mean)}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(f'X_train.shape: {X_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'y_test.shape: {y_test.shape}')

# training a machine learning model

# define param grid
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"] # radial basis function aka gaussian kernel
}

svc = SVC(probability=True)

# apply grid search cv
# we use gridsearch with 5 CV (cross validtion)
grid = GridSearchCV(svc, param_grid, cv=5, verbose=1, scoring='accuracy', n_jobs=-1)

grid.fit(X_train, y_train)

# show best params
print("grid.best_params_:", grid.best_params_)
print("grid.best_score_:", grid.best_score_)

model = grid.best_estimator_
joblib.dump(model, "./model/svc_model.joblib")
joblib.dump(mean, './model/mean.joblib')

# evaluate the model

y_pred = model.predict(X_test)

print("accuracy_score:", "\n", accuracy_score(y_test, y_pred))
print("classification_report:", "\n", classification_report(y_test, y_pred, target_names=['male','female']))
print("confusion_matrix:", "\n", confusion_matrix(y_test, y_pred))
