import os
import json
import shutil
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


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


def train_and_evaluate(algorithm, X, y, seed, best_params):

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
        
    best_model = algorithm()
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    accTrain, accTest, ofs_fpr, ofs_tpr, isAUC, ofsAUC  = \
                scores(best_model,
                    X_train,
                    y_train,
                    X_test,
                    y_test)
        
    print('Seed = ', seed)
    print('In Sample AUC', isAUC)
    print('Out of Sample AUC', ofsAUC)
    print('In Sample Misclassification', accTrain)
    print('Out of Sample Misclassification', accTest)
    print('\n')

    return best_model, accTrain, accTest, ofs_fpr, ofs_tpr, isAUC, ofsAUC
    
# List of well written names for the Cremona data
comorbidities = ['Multiple Sclerosis',
             'Acidosis', 'Anaemia', 'Asthma', 'Cancer', 'Chronic Heart Condition', 'Chronic Kidney', 'Chronic Liver', 'Chronic Obstructive Lung',
        'Diabetes', 'Epilepsy', 'Glaucoma', 'High Triglycerides', 'Hypercholesterolemia', 'Hypertension', 'Leukemia', 'Neutropenia', 'Osteoporosis', 'Parkinson', 'Rickets']
symptoms = ['Vomit', 'Diarrhea']
numeric = ['SaO2','Age', 'Cardiac Frequency', 'Diastolic Blood Pressure', 'Respiratory Frequency', 'Systolic Blood Pressure','Temperature Celsius']
categorical = ["Sex"]


def export_features_json(X, numeric, categorical,  symptoms, comorbidities, file_name):
    data = {'numeric': [],
            'categorical': [],
            'checkboxes': [],
            'multidrop': []}

    for i in range(len(numeric)):
        data['numeric'].append({"name":numeric[i], 'index' : list(X.columns).index(numeric[i]), "min_val" : np.min(X[numeric[i]]),
        "max_val" : np.max(X[numeric[i]]), "default" : np.round(np.median(X[numeric[i]]),2), 'explanation' : 'Insert value of ' + numeric[i]})

    for i in range(len(categorical)):
        data['categorical'].append({"name": categorical[i], 'index' : list(X.columns).index(categorical[i]), "vals" : list(np.unique(X[categorical[i]])),
        "default" : np.unique(X[categorical[i]])[0], 'explanation' : '' })

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
