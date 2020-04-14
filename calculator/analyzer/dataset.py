def create_dataset(data_dict,
        comorbidities=True,
        vitals=True,
        prediction='outcome'):

    data = data_dict['anagraphics']

    if comorbidities:
        data = data.join(data_dict['comorbidities'])

    if vitals:
        data = data.join(data_dict['vitals'])

    X = data.loc[:, data.columns.difference(['outcome', 'icu'])]
    y = data.loc[:, prediction]

    return X, y

