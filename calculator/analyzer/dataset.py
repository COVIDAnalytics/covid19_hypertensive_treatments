import numpy as np


def create_dataset(data_dict, discharge_data = True, 
                        comorbidities_data = True, 
                        vitals_data = True, 
                        lab_tests=True, 
                        anagraphics_data = False, 
                        swabs_data = False,
                        prediction='Outcome'):

    if discharge_data:
        data = data_dict['discharge']
    
    if anagraphics_data:
        data = data_dict['anagraphics']

    if comorbidities_data:
        data = data.join(data_dict['comorbidities'])

    if vitals_data:
        data = data.join(data_dict['vitals'])

    if lab_tests:
        data = data.join(data_dict['lab'])

    if swabs_data:
        data = data.join(data_dict['swab'])

    X = data.loc[:, data.columns.difference(['Outcome', 'ICU', 'Swab'])]
    y = data.loc[:, prediction]

    X = X.astype(np.float32)
    y = y.astype(int)

    return X, y
