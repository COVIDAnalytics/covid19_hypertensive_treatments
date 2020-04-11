import pandas as pd


def load_cremona(path):
    anagraphics = pd.read_csv("%s/anagraphics/anagraphics.csv" % path).dropna(how='all')

    # Load drugs
    drugs_cremona = pd.read_csv("%s/therapies/drugs_cremona.csv" % path)
    drugs_orgoglio_po = pd.read_csv("%s/therapies/drugs_oglio_po.csv" % s)
    drugs = drugs_cremona.append(drugs_orgoglio_po).dropna(how='all')

    # Load comorbidities
    comorbidities_data = pd.read_csv('%s/therapies/active_substances_comorbidities.csv' % path)[['Active_substance', 'therapy_for_filtered']]

    # Load vital signs in ER
    er_vital_signs = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    er_vital_signs['SCHEDA_PS'] = er_vital_signs['SCHEDA_PS'].astype(str)

    data = {'anagraphics': anagraphics,
            'drugs': drugs,
            'comorbidities': comorbidities_data,
            'er_vitals': er_vital_signs}
    return data
