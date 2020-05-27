


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pathlib
import pandas as pd
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, accuracy_score

class Model:
    
    def __init__(self, name):
        self.cwd = pathlib.Path().absolute()
        self.name = name
        if name == 'tree':
            self.classifier = RandomForestClassifier(verbose = True,n_jobs = -1)
            self.param = {'min_child_weight' : [i for i in range(1,4)],'gamma' : [i for i in range(1,4)], 
                      'colsample_bytree' : [i/10 for i in range(3,8)],'max_depth' : [3,4,5],
             'n_estimators' : [500] , 'learning_rate' : [0.05]} 
        
        elif name == 'LogisticRegression':
            self.classifier = LogisticRegression(verbose = True, n_jobs = -1)
            self.param = {'penalty' : ['l2'], 
                          'C' : [0.5,1,1.5],
                          'fit_intercept' : [True, False],
                          'max_iter' : [100,500]
                          }
        
        elif name == 'KNN':
            self.classifier = KNeighborsClassifier( n_jobs = -1)
            self.param = {'n_neighbors' : [i for i in range(0,100,5)],
                          'weights' : ['uniform', 'distance'],
                          }
        
        else :
            self.classifier = RandomForestClassifier(verbose = True, n_jobs = -1)
            self.param = {'n_estimators' : [500],
                          'max_depth' : [3,4,5],
                          'min_samples_split' : [i for i in range(1,4)],
                          'ccp_alpha': [i/2 for i in range(1,4)]}
    
    
    
    def getData(self):
        
        if self.name == 'tree':
            X_train = pd.read_csv(str(self.cwd)+'\\TrainTestData\\X_train.csv')
            X_test = pd.read_csv(str(self.cwd)+'\\TrainTestData\\X_test.csv')
            y_train = pd.read_csv(str(self.cwd)+'\\TrainTestData\\y_train.csv')
            y_test = pd.read_csv(str(self.cwd)+'\\TrainTestData\\y_test.csv')
        else:
            X_train = pd.read_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\X_train.csv')
            X_test = pd.read_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\X_test.csv')
            y_train = pd.read_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\y_train.csv')
            y_test = pd.read_csv(str(self.cwd)+'\\FeatureEnggTrainTestData\\y_test.csv')
        
        return X_train, X_test, y_train, y_test
    
    def gridSearch(self):
        grid = RandomizedSearchCV(self.classifier, self.param, scoring = 'f1')
        X_train, X_test, y_train, y_test = self.getData()
        grid.fit(X_train, y_train)
        return grid.best_params_

    def train(self):
        if os.path.exists(str(self.cwd) + '\\LogisticRegression\\logistic_regression_predictions.csv') and os.path.exists(str(self.cwd) + '\\KNN\\knn_predictions.csv') and os.path.exists(str(self.cwd) + '\\RandomForest\\random_forest_predictions.csv'):
            return
        print('HI should not be here')
        X_train,X_test,y_train, y_test = self.getData()
        best_params = None
        if self.name == 'tree':
            self.classifier = RandomForestClassifier(verbose = True, n_jobs = -1)
            self.classifier.fit(X_train, y_train)
        
        elif self.name == 'LogisticRegression' :
            best_params = self.gridSearch()
            self.classifier = LogisticRegression(**best_params,verbose = True, n_jobs = -1)
            self.classifier.fit(X_train, y_train)
            
        
        elif self.name == 'KNN':
            best_params = self.gridSearch()
            self.classifier = KNeighborsClassifier(**best_params, n_jobs = -1)
            self.classifier.fit(X_train, y_train)
        
        else :
            best_params = self.gridSearch()
            self.classifier = RandomForestClassifier(**best_params,verbose = True, n_jobs = -1)
            self.classifier.fit(X_train, y_train)
    
    def getXGBModel(self):
        return self.classifier
    
    def predict(self):
        X_train,X_test,y_train, y_test = self.getData()
        if self.name == 'LogisticRegression' and os.path.exists(str(self.cwd) + '\\LogisticRegression\\logistic_regression_predictions.csv'):
            y_pred = pd.read_csv(str(self.cwd) + '\\LogisticRegression\\logistic_regression_predictions.csv')
            return y_pred.iloc[:,0]
        elif self.name == 'KNN' and os.path.exists(str(self.cwd) + '\\KNN\\knn_predictions.csv'):
            y_pred = pd.read_csv(str(self.cwd) + '\\KNN\\knn_predictions.csv')
            return y_pred.iloc[:,0]
        elif self.name == 'RandomForest' and os.path.exists(str(self.cwd) + '\\RandomForest\\random_forest_predictions.csv'):
            y_pred = pd.read_csv(str(self.cwd) + '\\RandomForest\\random_forest_predictions.csv')
            return y_pred.iloc[:,0]
        y_pred = self.classifier.predict(X_test)
        if self.name == 'LogisticRegression':
            try:
                os.mkdir(str(self.cwd)+'\\LogisticRegression')
            except:
                print('Predicting....')
            pd.DataFrame(y_pred).to_csv(str(self.cwd) + '\\LogisticRegression\\logistic_regression_predictions.csv', index = False)
        elif self.name == 'KNN':
            try:
                os.mkdir(str(self.cwd)+'\\KNN')
            except:
                print('Predicting....')
            pd.DataFrame(y_pred).to_csv(str(self.cwd) + '\\KNN\\knn_predictions.csv', index = False)
        else:
            try:
                os.mkdir(str(self.cwd)+'\\RandomForest')
            except:
                print('Predicting....')
            pd.DataFrame(y_pred).to_csv(str(self.cwd) + '\\RandomForest\\random_forest_predictions.csv', index = False)
        
        return y_pred
    
    def plot_learning_curve(self, title, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        
        X_train, X_test, y_train, y_test = self.getData()
        
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
    
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self.classifier, X_train, y_train, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
    
        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")
    
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")
    
        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")
        
        if self.name == 'LogisticRegression':
            plt.savefig(str(self.cwd) + '\\LogisticRegression\\losscurve_logistic_regression.png')
        elif self.name == 'KNN':
            plt.savefig(str(self.cwd) + '\\KNN\\losscurve_knn.png')
        else:
            plt.savefig(str(self.cwd) + '\\RandomForest\\losscurve_random_forest.png')
    
        return plt

    def plotAUC(self):
        if self.name == 'LogisticRegression':
            y_pred = pd.read_csv(str(self.cwd) + '\\LogisticRegression\\logistic_regression_predictions.csv')
        elif self.name == 'KNN':
            y_pred = pd.read_csv(str(self.cwd) + '\\KNN\\knn_predictions.csv')
        else:
            y_pred = pd.read_csv(str(self.cwd) + '\\RandomForest\\random_forest_predictions.csv')
        a,b, c,y_test = self.getData()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if self.name == 'LogisticRegression':
            plt.savefig(str(self.cwd) + '\\LogisticRegression\\roc_logistic_regression.png')
        elif self.name == 'KNN':
            plt.savefig(str(self.cwd) + '\\KNN\\roc_knn.png')
        else:
            plt.savefig(str(self.cwd) + '\\RandomForest\\roc_random_forest.png')
        plt.show()
            
    def printAccuracy(self):
        _,_,_,y_test = self.getData()
        if self.name == 'LogisticRegression':
            y_pred = pd.read_csv(str(self.cwd) + '\\LogisticRegression\\logistic_regression_predictions.csv')
        elif self.name == 'KNN':
            y_pred = pd.read_csv(str(self.cwd) + '\\KNN\\knn_predictions.csv')
        else:
            y_pred = pd.read_csv(str(self.cwd) + '\\RandomForest\\random_forest_predictions.csv')
        
        print('Accuracy Score for ' + self.name + ' is ' + str(accuracy_score(y_test, y_pred)))
            
        