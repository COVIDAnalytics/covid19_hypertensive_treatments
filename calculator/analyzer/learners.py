#Julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from interpretableai import iai

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mlflow.sklearn
import xgboost as xgb



def train_oct(X_train, y_train,
              X_test, y_test,
              output_path,
              seed=1):

    oct_grid = iai.GridSearch(
        iai.OptimalTreeClassifier(
            random_seed = seed,
        ),
        max_depth=range(1, 10),
        minbucket=[10, 15, 20, 25, 30, 35],
        ls_num_tree_restarts=200,
    )
    oct_grid.fit_cv(X_train, y_train, n_folds=10, validation_criterion = 'auc')
    best_learner = oct_grid.get_learner()
    best_learner.write_json('%s/learner.json' % output_path)
    best_learner.write_questionnaire('%s/app.html' % output_path)
    best_learner.write_html('%s/tree.html' % output_path)
    best_learner.write_png('%s/tree.png' % output_path)
    in_auc = oct_grid.score(X_train, y_train, criterion='auc')
    out_auc = oct_grid.score(X_test, y_test, criterion='auc')
    in_mis = oct_grid.score(X_train, y_train, criterion='misclassification')
    out_mis = oct_grid.score(X_test, y_test, criterion='misclassification')
    print('In Sample AUC', in_auc)
    print('Out of Sample AUC', out_auc)
    print('In Sample Misclassification', in_mis)
    print('Out of Sample Misclassification', out_mis)
    return in_auc, out_auc, in_mis, out_mis


#INITIALIZE A LIST TO KEEP TRACK OF ALL BEST MODELS DEVELOPED

#DEFINE FUNCTION THAT COMPUTES ACCURACY, TPR, FPR, AND AUC for GIVEN MODEL
def Scores(model, t_X, t_Y, te_X, te_Y):

    # misclassification accuracies
    accTrain = np.round(sum(model.predict(t_X) == t_Y)/len(t_Y),2)
    accTest = np.round(sum(model.predict(te_X) == te_Y)/len(te_Y),2)
    pred_t_Y = model.predict_proba(t_X)[:, 1]
    pred_te_Y = model.predict_proba(te_X)[:, 1]

    is_fpr, is_tpr, _ = roc_curve(t_Y, pred_t_Y)
    isAUC = auc(is_fpr, is_tpr)

    ofs_fpr, ofs_tpr, _ = roc_curve(te_Y, pred_te_Y)
    ofsAUC = auc(ofs_fpr, ofs_tpr)
    return (accTrain, accTest, ofs_fpr, ofs_tpr, isAUC, ofsAUC)

#DEFINE FUNCTION THAT RETURNS TOP 10 PREDICTORS GIVEN A MODEL
def top10(model, trainX):
    try:
        varsImpo = pd.DataFrame({'names':trainX.columns, 'vals':model.feature_importances_})
        varsImpo = varsImpo.sort_values(by='vals', ascending = False)

    except:
        print("The model does not support the method: feature_importances_")

    return varsImpo


#INITIATE 10-FOLD CV

def xgboost_classifier(X_train, y_train, X_test, y_test, param_grid, output_path, seed = 1):
    X_train.sex = X_train.sex.cat.codes.astype('category')
    X_test.sex = X_test.sex.cat.codes.astype('category')
    y_train = y_train.cat.codes.astype('category')
    y_test = y_test.cat.codes.astype('category')

    XGB = xgb.XGBClassifier()
    gridsearch = GridSearchCV(estimator = XGB, param_grid = param_grid, cv = 10, verbose = 1)
    gridsearch.fit(X_train.astype(int), y_train.astype(int), eval_metric="auc")

    #RECORD BEST MODEL
    bestHyp = gridsearch.best_params_
    bestXGB = gridsearch.best_estimator_
    mlflow.sklearn.save_model(bestXGB, output_path, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE) 
    accTrain_XGB, accTest_XGB, ofs_fpr_XGB, ofs_tpr_XGB, isAUC_XGB, ofsAUC_XGB  = Scores(bestXGB, X_train.astype(int), y_train.astype(int), X_test.astype(int), y_test.astype(int))
    print('In Sample AUC', isAUC_XGB)
    print('Out of Sample AUC', ofsAUC_XGB)
    print('In Sample Misclassification', accTrain_XGB)
    print('Out of Sample Misclassification', accTest_XGB)
    return isAUC_XGB, ofsAUC_XGB, accTrain_XGB, accTest_XGB

 
 
def rf_classifier(X_train, y_train, X_test, y_test, param_grid, output_path, seed = 1):
    X_train.sex = X_train.sex.cat.codes.astype('category')
    X_test.sex = X_test.sex.cat.codes.astype('category')
    y_train = y_train.cat.codes.astype('category')
    y_test = y_test.cat.codes.astype('category')

    RF = RandomForestClassifier()
    gridsearch = GridSearchCV(estimator = RF, param_grid = param_grid, cv = 10, verbose = 1)
    gridsearch.fit(X_train, y_train)

    #RECORD BEST MODEL
    bestHypRF = gridsearch.best_params_
    bestRF = gridsearch.best_estimator_
    mlflow.sklearn.save_model(bestRF, output_path, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE) 
    accTrain_RF, accTest_RF, ofs_fpr_RF, ofs_tpr_RF, isAUC_RF, ofsAUC_RF  = Scores(bestRF, X_train.astype(int), y_train.astype(int), X_test.astype(int), y_test.astype(int))
    print('In Sample AUC', isAUC_RF)
    print('Out of Sample AUC', ofsAUC_RF)
    print('In Sample Misclassification', accTrain_RF)
    print('Out of Sample Misclassification', accTest_RF)
    return isAUC_RF, ofsAUC_RF, accTrain_RF, accTest_RF


