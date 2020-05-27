# -*- coding: utf-8 -*-
"""
Created on Wed May 27 07:11:18 2020

@author: rusha
"""


import models
import shap
import pathlib
from preprocess import PreProcess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
preproc = PreProcess()
preproc.splitStandardize()

X_train,_,y_train,y_test = models.Model('tree').getData()
estimator = models.Model('tree').getXGBModel()
estimator.fit(X_train,y_train)
importances = estimator.feature_importances_
cwd = pathlib.Path().absolute()
indices = np.argsort(importances)[::-1]
cols = X_train.columns.tolist()
std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
     axis=0)
tempcols = []
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))
    tempcols.append(cols[indices[f]])
    
        # Plot the impurity-based feature importances of the forest
plt.figure(figsize = (10,5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.tick_params(axis = 'x', size = 1, labelrotation = 85)
plt.xticks(range(X_train.shape[1]), tempcols)
plt.xlim([-1, X_train.shape[1]])
plt.show()
cwd = pathlib.Path().absolute()
y_pred = np.array(pd.read_csv(str(cwd) + '\\RandomForest\\random_forest_predictions.csv'))
print(f1_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))