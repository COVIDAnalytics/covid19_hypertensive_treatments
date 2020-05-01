import os
import json
import shutil
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

FEATURE_BOUNDS_EXPLANATION = {'Alanine Aminotransferase (ALT)': [2.0, 929.0,'Alanine Aminotransferase (ALT) in U/L'],
                            'Prothrombin Time (INR)': [0.0, 17.0, 'Prothrombin Time Ratio (INR)'],
                            'Blood Creatinine': [0.0, 11.0, 'Blood Creatinine in mg/dL'],
                            'Respiratory Frequency': [14.0, 40.0, 'Number of Breaths per Minute'],
                            'Cholinesterase': [1422.0, 19851.0, 'Cholinesterase in Units per Liter (U/L)'],
                            'ABG: pH': [7.0, 8.0, 'Acid-base balance of the blood'],
                            'Blood Sodium': [115.0, 166.0, 'Blood Sodium in mmol/L'],
                            'CBC: Hemoglobin': [6.0, 19.0, 'Hemoglobin in g/dL'],
                            'Cardiac Frequency': [40.0, 171.0, 'Number of Beats per Minute.'],
                            'ABG: PaO2': [17.0, 305.0, 'Partial Pressure of Oxygen in mmHg'],
                            'Total Bilirubin': [0.0, 5.0, 'Total Bilirubin in mg/dL'],
                            'ABG: MetHb': [0.0, 3.0, 'Methemoglobin fraction'],
                            'ABG: Oxygen Saturation (SaO2)': [80, 100.0, 'Oxygen Saturation (SaO2) in %'],
                            'SaO2': [80, 100.0, 'Oxygen Saturation (SaO2) in %'],
                            'Potassium Blood Level': [2.0, 7.0, 'Potassium Blood Level in mmol/L'],
                            'Blood Urea Nitrogen (BUN)': [4.0, 174.0, 'Blood Urea Nitrogen (BUN) in mg/dL'],
                            'CBC: Leukocytes': [0.0, 36.0, 'Leukocytes in 10^3/muL'],
                            'Body Temperature': [34, 104.0, 'Body temperature measurement. Use the dropdown to select the unit (Fahrenheit or Celsius).'],
                            'CBC: Platelets': [20.0, 756.0, 'Platelets in 10^3/muL'],
                            'CBC: Mean Corpuscular Volume (MCV)': [58.0, 116.0,'Mean Corpuscular Volume (MCV) in fL'],
                            'Blood Calcium': [6.0, 12.0, 'Blood Calcium in mg/dL'],
                            'Age': [0.0, 100.0, 'Age of the patient. Modeled only for adults.'],
                            'C-Reactive Protein (CRP)': [0.0, 567.0, 'C-Reactive Protein (CRP) in mg/L'],
                            'Aspartate Aminotransferase (AST)': [9.0, 941.0,'Aspartate Aminotransferase (AST) in U/L'],
                            'Glycemia': [57.0, 620.0, 'Blood Glucose in mg/dL'],
                            'Systolic Blood Pressure': [20.0, 199.0, 'Systolic Blood Pressure in mmHg'],
                            'Gender': 'Select the gender of the patient', 
                            'Activated Partial Thromboplastin Time (aPTT)' : [0.0, 6.0, 'Activated Partial Thromboplastin Time in minutes'],
                            'Blood Amylase' : [10, 300, 'Blood Amylase Level in Units per Liter (U/L)'],
                            'CBC: Red cell Distribution Width (RDW)': [10, 27, 'Red cell Distribution Width in %']}



def change_SaO2(x):
    if x > 92:
        return 1
    else:
        return 0

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
    varsImpo = varsImpo[:n]

    print("Top %d\n" % n)
    print(varsImpo)
    return varsImpo

def plot_correlation(X, file_name):
    data_corr = X.copy()
    data_corr['Sex'] = data_corr['Sex'].astype(object)
    data_corr.loc[data_corr['Sex'] == 'M', 'Sex'] = 0
    data_corr.loc[data_corr['Sex'] == 'F', 'Sex'] = 1
    X_corr = data_corr.loc[:, data_corr.columns != 'outcome'].astype(np.float64).corr()

    # Plot correlation
    mask = np.triu(np.ones_like(X_corr.corr(), dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(X_corr.corr(), mask=mask, cmap=cmap, center=0,
                xticklabels=True, yticklabels=True,
                vmax=1.0, vmin=-1.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    create_dir(os.path.dirname(file_name))
    plt.tight_layout()
    plt.savefig(file_name)

    # Print correlations > 0.25
    upper = X_corr.corr().where(np.triu(np.ones(X_corr.corr().shape), k=1).astype(np.bool))
    rows, columns = np.where(abs(upper) > 0.8)

    print("Highest correlations (> 0.8)")
    print(list(zip(upper.columns[rows], upper.columns[columns])))


# List of well written names for the Cremona data
comorbidities = ['Multiple Sclerosis',
             'Acidosis', 'Anaemia', 'Asthma', 'Cancer', 'Chronic Heart Condition', 'Chronic Kidney', 'Chronic Liver', 'Chronic Obstructive Lung',
        'Diabetes', 'Epilepsy', 'Glaucoma', 'High Triglycerides', 'Hypercholesterolemia', 'Hypertension', 'Leukemia', 'Neutropenia', 'Osteoporosis', 'Parkinson', 'Rickets']
symptoms = ['Vomit', 'Diarrhea']
numeric = ['SaO2','Age', 'Cardiac Frequency', 'Diastolic Blood Pressure', 'Respiratory Frequency', 'Systolic Blood Pressure','Temperature Celsius']
categorical = ["Sex"]

def impute_missing(df):
    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(df)
    imputed_df = imp_mean.transform(df)
    df = pd.DataFrame(imputed_df, index=df.index, columns=df.columns)
    return df


def export_features_json(X, numeric, categorical,  symptoms, comorbidities, file_name):
    data = {'numeric': [],
            'categorical': [],
            'checkboxes': [],
            'multidrop': []}
    X = impute_missing(X)

    for i in range(len(numeric)):
        data['numeric'].append({"name":numeric[i], 'index' : list(X.columns).index(numeric[i]), "min_val" : FEATURE_BOUNDS_EXPLANATION[numeric[i]][0],
        "max_val" : FEATURE_BOUNDS_EXPLANATION[numeric[i]][1], "default" : np.round(np.median(X[numeric[i]]),2), 'explanation' : FEATURE_BOUNDS_EXPLANATION[numeric[i]][2]})

    for i in range(len(categorical)):
        data['categorical'].append({"name": categorical[i], 'index' : list(X.columns).index(categorical[i]), "vals" : list(np.unique(X[categorical[i]])),
        "default" : np.unique(X[categorical[i]])[0], 'explanation' : FEATURE_BOUNDS_EXPLANATION[categorical[i]] })

    data['checkboxes'].append({'name': "Symptoms", "index": [], "vals" : [], 'explanation': []})
    data['multidrop'].append({'name': "Comorbidities", "index": [], "vals" : [], 'explanation': []})

    for i in range(len(symptoms)):
        data['checkboxes'][0]["index"].append(list(X.columns).index(symptoms[i]))
        data['checkboxes'][0]["vals"].append(symptoms[i])
    data['checkboxes'][0]["explanation"].append("Select the existing symptoms.")


    for i in range(len(comorbidities)):
        data['multidrop'][0]["index"].append(list(X.columns).index(comorbidities[i]))
        data['multidrop'][0]["vals"].append(comorbidities[i])
    data['multidrop'][0]["explanation"].append("Select the existing chronic diseases or conditions.")


    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

    return data


def export_model_imp_json(model, imp, json, cols, path):
    exp = {'model': model,
    'imputer': imp,
    'json': json,
    'columns': list(cols)}
    with open(path, 'wb') as handle:
        pickle.dump(exp, handle, protocol=4)
    return exp


def store_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)


def get_percentages(df, missing_type=np.nan):
    if np.isnan(missing_type):
        df = df.isnull()  # Check what is NaN
    elif missing_type is False:
        df = ~df  # Check what is False

    percent_missing = df.sum() * 100 / len(df)
    return pd.DataFrame({'percent_missing': percent_missing})


def remove_missing(df, missing_type=np.nan, nan_threshold=40, impute=False):
    missing_values = get_percentages(df, missing_type)
    df_features = missing_values[missing_values['percent_missing'] < nan_threshold].index.tolist()

    df = df[df_features]

    if impute:
        imp_mean = IterativeImputer(random_state=0)
        imp_mean.fit(df)
        imputed_df = imp_mean.transform(df)
        df = pd.DataFrame(imputed_df, index=df.index, columns=df.columns)

    return df
