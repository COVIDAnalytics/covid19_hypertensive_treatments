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
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    create_dir(os.path.dirname(file_name))
    plt.tight_layout()
    plt.savefig(file_name)

    # Print correlations > 0.25
    upper = X_corr.corr().where(np.triu(np.ones(X_corr.corr().shape), k=1).astype(np.bool))
    rows, columns = np.where(abs(upper) > 0.25)

    print("Highest correlations (> 0.25)")
    print(list(zip(upper.columns[rows], upper.columns[columns])))


def export_features_json(X, file_name):
    data = {'numeric': {},
            'categorical': {},
            'checkboxes': {}}

    import ipdb; ipdb.set_trace()




    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
