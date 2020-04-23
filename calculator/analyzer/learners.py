#Julia

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mlflow.sklearn
import xgboost as xgb

from analyzer.utils import top_features, remove_dir

def train_oct(X_train, y_train,
              X_test, y_test,
              output_path,
              seed=1):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from interpretableai import iai

    oct_grid = iai.GridSearch(
        iai.OptimalTreeClassifier(
            random_seed = seed,
        ),
        max_depth=range(3, 10),
        minbucket=[5, 10, 15, 20, 25, 30, 35],
        ls_num_tree_restarts=200,
    )
    oct_grid.fit_cv(X_train, y_train, n_folds=5, validation_criterion = 'auc')
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
def scores(model, t_X, t_Y, te_X, te_Y):

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


#INITIATE 10-FOLD CV

def xgboost_classifier(X_train, y_train, X_test, y_test, param_grid, output_path, seed = 1):
    y_train = y_train.cat.codes.astype('category')
    y_test = y_test.cat.codes.astype('category')

    XGB = xgb.XGBClassifier()
    gridsearch = GridSearchCV(estimator = XGB, param_grid = param_grid, cv = 10, n_jobs=-1, verbose = 1)
    gridsearch.fit(X_train.astype(np.float64),
                   y_train.astype(int),
                   eval_metric="auc")

    #RECORD BEST MODEL
    bestHyp = gridsearch.best_params_
    bestXGB = gridsearch.best_estimator_

    accTrain_XGB, accTest_XGB, ofs_fpr_XGB, ofs_tpr_XGB, isAUC_XGB, ofsAUC_XGB  = \
            scores(bestXGB,
                   X_train.astype(np.float64),
                   y_train.astype(int),
                   X_test.astype(np.float64),
                   y_test.astype(int)
                   )

    print('In Sample AUC', isAUC_XGB)
    print('Out of Sample AUC', ofsAUC_XGB)
    print('In Sample Misclassification', accTrain_XGB)
    print('Out of Sample Misclassification', accTest_XGB)
    print(pd.DataFrame(bestXGB.get_params().items(), columns = ['Parameter', 'Value']))
    top_features(bestXGB, X_train)

    remove_dir(output_path)
    mlflow.sklearn.save_model(bestXGB, output_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)


    return isAUC_XGB, ofsAUC_XGB, accTrain_XGB, accTest_XGB



def rf_classifier(X_train, y_train, X_test, y_test, param_grid, output_path, seed = 1):

    y_train = y_train.cat.codes.astype('category')
    y_test = y_test.cat.codes.astype('category')

    RF = RandomForestClassifier()
    gridsearch = GridSearchCV(estimator = RF, param_grid = param_grid, n_jobs=-1, cv = 10, verbose = 1)
    gridsearch.fit(X_train, y_train)

    #RECORD BEST MODEL
    bestHypRF = gridsearch.best_params_
    bestRF = gridsearch.best_estimator_

    accTrain_RF, accTest_RF, ofs_fpr_RF, ofs_tpr_RF, isAUC_RF, ofsAUC_RF  = \
            scores(bestRF,
                   X_train.astype(np.float64),
                   y_train.astype(int),
                   X_test.astype(np.float64),
                   y_test.astype(int))

    print('In Sample AUC', isAUC_RF)
    print('Out of Sample AUC', ofsAUC_RF)
    print('In Sample Misclassification', accTrain_RF)
    print('Out of Sample Misclassification', accTest_RF)
    print(pd.DataFrame(bestRF.get_params().items(), columns = ['Parameter', 'Value']))
    top_features(bestRF, X_train)

    remove_dir(output_path)
    mlflow.sklearn.save_model(bestRF, output_path, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    return isAUC_RF, ofsAUC_RF, accTrain_RF, accTest_RF
