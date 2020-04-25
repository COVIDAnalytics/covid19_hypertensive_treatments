import numpy as np


def create_dataset(data_dict,
        comorbidities=True,
        vitals=True,
        lab=True,
        prediction='Outcome'):

    data = data_dict['anagraphics']

    if comorbidities:
        data = data.join(data_dict['comorbidities'])

    if vitals:
        data = data.join(data_dict['vitals'])

    if lab:
        data = data.join(data_dict['lab'])

    X = data.loc[:, data.columns.difference(['Outcome', 'ICU', 'Swab'])]
    y = data.loc[:, prediction]

    X = X.astype(np.float64)
    y = y.astype(int)

    return X, y


