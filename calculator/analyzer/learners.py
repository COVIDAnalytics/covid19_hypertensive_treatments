#Julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from interpretableai import iai

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
    in_auc = oct_grid.score(X_train, y_train, criterion='auc')
    out_auc = oct_grid.score(X_test, y_test, criterion='auc')
    print('In Sample AUC', in_auc)
    print('Out of Sample AUC', out_auc)
    return in_auc, out_auc
