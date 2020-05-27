

import models
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt


class FeatureImportance:
    
    def __init__(self):
        self.xgb = models.Model('tree')
        self.estimator = self.xgb.getXGBModel()
        self.X_train, self.X_test, self.y_train, self.y_test = self.xgb.getData()
        self.estimator.fit(self.X_train,self.y_train)
        self.importances = self.estimator.feature_importances_
        self.cwd = pathlib.Path().absolute()
        self.indices = np.argsort(self.importances)[::-1]
        self.std = np.std([tree.feature_importances_ for tree in self.estimator.estimators_],
             axis=0)
        
    def plotFeatures(self):
        cols = self.X_train.columns.tolist()
        features = []
        significance = []
        for f in range(self.X_train.shape[1]):
            features.append(cols[self.indices[f]])
            significance.append(self.importances[self.indices[f]])
        dictcol = {'features':features,'significance':significance}
        feature_importance_df = pd.DataFrame(dictcol)
        feature_importance_df.to_csv(str(self.cwd)+'\\feature_importance.csv', index = False)
        # Plot the impurity-based feature importances of the forest
        plt.figure(figsize = (10,5))
        plt.title("Feature importances")
        plt.bar(range(self.X_train.shape[1]), self.importances[self.indices],
                color="r", yerr=self.std[self.indices], align="center")
        plt.xticks(range(self.X_train.shape[1]), features)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.tick_params(axis = 'x', size = 1, labelrotation = 85)
        plt.savefig(str(self.cwd)+'\\feature_importance.png')
        plt.show()
    
    def trimFeatures(self):
        cols = self.X_train.columns.tolist()
        self.indices = self.indices[0:51]
        finalcols = map(cols.__getitem__,self.indices)
        X_tr = self.X_train[finalcols]
        X_te = self.X_test[X_tr.columns]
        X_tr.to_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\X_train.csv',index = False)
        X_te.to_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\X_test.csv',index = False)
        self.y_train.to_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\y_train.csv', index = False)
        self.y_test.to_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\y_test.csv', index = False)
        
        




