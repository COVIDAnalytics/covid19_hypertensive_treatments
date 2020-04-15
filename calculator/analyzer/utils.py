import os
import json
import shutil
import pandas as pd


def create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def top_features(model, X_train, n=20):
    varsImpo = pd.DataFrame({'names':X_train.columns,
                             'vals':model.feature_importances_})

    varsImpo = varsImpo.sort_values(by='vals',
                                    ascending = False)
    return varsImpo[:n]

def export_features_json(X, file_name):
    data = {'numeric': {},
            'categorical': {},
            'checkboxes': {}}

    import ipdb; ipdb.set_trace()




    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
