#practice machine learning project from elite data science

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

#split target from training features
y = data.quality
x = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123, stratify = y)

#fitting transformer API
scaler = preprocessing.StandardScaler().fit(X_train)

#applying transformer to training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}

#sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

#fit and tune model
clf.fit(X_train, y_train)

#refit on entire training set
#print(clf.refit) since is true no more code needed

#evaluate model pipeline on test data
pred = clf.predict(X_test)
print (r2_score(y_test, pred))
print (mean_squared_error(y_test, pred))

#save model
joblib.dump(clf, 'rf_regressor.pkl')
# for loading: clf2 = joblib.load('rf_regressor.pkl')
