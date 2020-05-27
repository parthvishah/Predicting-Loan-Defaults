

from preprocess import PreProcess
from featureimportance import FeatureImportance
import models
import warnings
warnings.filterwarnings("ignore")

def main():
    preproc = PreProcess()
    preproc.splitStandardize()
    preproc.upSample()
    
    print('Now beginning Feature Importance Module')
    print('...')
    print('The dimension of the data is reduced according SHAP values')
    print('--------------------------------------------------------------------')
    
    print('\n')
    
    fi = FeatureImportance()
    print('Below is the importance of each feature')
    fi.plotFeatures()
    print('Trimming the dimension of the original dataset')
    fi.trimFeatures()
    
    print('Feature Importance phase done')
    print('\n-------------------------------------------------------------------')
    
    print('\n')
    
    print('Modelling Phase')
    print('Results will be stored in the respective folder.')
    
    print('\nLogistic Regression')
    m1 = models.Model('LogisticRegression')
    m1.train()
    pred1 = m1.predict()
    print('...')
    m1.plot_learning_curve('LossCurve for LogReg')
    m1.plotAUC()
    m1.printAccuracy()
    print('Logistic Regression Done')
    
    print('\nKNN')
    m2 = models.Model('KNN')
    m2.train()
    pred2 = m2.predict()
    print('...')
    m2.plot_learning_curve('LossCurve for KNN')
    m2.plotAUC()
    m2.printAccuracy()
    print('KNN Done')
    
    print('\nRandom Forest')
    m3 = models.Model('RandomForest')
    m3.train()
    pred3 = m3.predict()
    print('...')
    m3.plot_learning_curve('LossCurve for RandomFor')
    m3.plotAUC()
    m3.printAccuracy()
    print('RandomForest Done')
    
    print('\n')
    print('ALL DONE')
    
    print('-------------------------------------------------------------------')
    
if __name__ == "__main__":
    main()

