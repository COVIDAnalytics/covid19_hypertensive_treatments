import os
import json


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def export_features_json(X, file_name):
    data = {'numeric': {},
            'categorical': {},
            'checkboxes': {}}

    import ipdb; ipdb.set_trace()




    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
