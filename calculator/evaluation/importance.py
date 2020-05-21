import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


from math import sqrt
import matplotlib

title_mapping = {
    'ABG: Oxygen Saturation (SaO2)': 'Oxygen Saturation (\%)',
    'Alanine Aminotransferase (ALT)': 'Alanine Aminotransferase (U/L)',
    'Age': 'Age',
    'Aspartate Aminotransferase (AST)': 'Aspartate Aminotransferase (U/L)',
    'Blood Creatinine': 'Blood Creatinine (mg/dL)',
    'Blood Sodium': 'Sodium (mmol/L)',
    'Blood Urea Nitrogen (BUN)': 'Blood Urea Nitrogen (mg/dL)',
    'Body Temperature': 'Temperature (F)',
    'C-Reactive Protein (CRP)':  'C-Reactive Protein (mg/L)',
    'CBC: Hemoglobin': 'Hemoglobin (g/dL)',
    'CBC: Leukocytes': 'White Blood Cells (1000/muL)',
    'CBC: Mean Corpuscular Volume (MCV)': 'Mean Corpuscular Volume (fL)',
    'CBC: Platelets': 'Platelets (1000/muL)',
    'CBC: Red cell Distribution Width (RDW)': 'Red Cell Distribution Width (%)',
    'Cardiac Frequency': 'Heart Rate (bpm)',
    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
    'Gender': 'Gender (M/F)',
    'Glycemia': 'Blood Glucose (mg/dL)',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'Prothrombin Time (INR)',
    'Systolic Blood Pressure': 'Systolic Blood Pressure (mmHg)',
    'SaO2': 'Oxygen Saturation (\%)',
    'Blood Calcium': 'Calcium (mg/dL)',
    'ABG: PaO2': 'Partial Pressure Oxygen (PaO2)',
    'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory Frequency (bpm)',
    'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total Bilirubin (mg/dL)',
    'Comorbidities': 'Comorbidities',
    'Diabetes': 'Diabetes',
    'Chronic kidney disease': 'Chronic kidney disease',
    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis and other heart disease'
}

title_mapping_summary = {
    'ABG: Oxygen Saturation (SaO2)': 'Oxygen Saturation (\%)',
    'Alanine Aminotransferase (ALT)': 'ALT (U/L)',
    'Age': 'Age',
    'Aspartate Aminotransferase (AST)': 'AST (U/L)',
    'Blood Creatinine': 'Creatinine (mg/dL)',
    'Blood Sodium': 'Sodium (mmol/L)',
    'Blood Urea Nitrogen (BUN)': 'BUN (mg/dL)',
    'Body Temperature': 'Temperature (F)',
    'C-Reactive Protein (CRP)':  'CRP (mg/L)',
    'CBC: Hemoglobin': 'Hemoglobin (g/dL)',
    'CBC: Leukocytes': 'WBC (1000/muL)',
    'CBC: Mean Corpuscular Volume (MCV)': 'MCV (fL)',
    'CBC: Platelets': 'Platelets (1000/muL)',
    'CBC: Red cell Distribution Width (RDW)': 'RDW (\%)',
    'Cardiac Frequency': 'Heart Rate (bpm)',
    'Cardiac dysrhythmias': 'Cardiac\n dysrhythmias',
    'Gender': 'Gender (M/F)',
    'Glycemia': 'Blood Glucose (mg/dL)',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'INR',
    'Systolic Blood Pressure': 'Systolic BP (mmHg)',
    'SaO2': 'Oxygen Saturation (\%)',
    'Blood Calcium': 'Calcium',
    # 'ABG: PaO2': 'Partial Pressure\n Oxygen (PaO2)',
    # 'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory Frequency',
    # 'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total Bilirubin (mg/dL)',
    'Comorbidities': 'Comorbidities',
    'Diabetes': 'Diabetes',
    'Chronic kidney disease': 'Chronic\n kidney disease',
    'Cardiac dysrhythmias': 'Cardiac\ndysrhythmias',
    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis\n and other heart disease'
}


#%% Latex-style image printing

SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    # NB (bart): default font-size in latex is 11. This should exactly match
    # the font size in the text if the figwidth is set appropriately.
    # Note that this does not hold if you put two figures next to each other using
    # minipage. You need to use subplots.
    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 12, # fontsize for x and y labels (was 12 and before 10)
              'axes.titlesize': 12,
              'font.size': 12, # was 12 and before 10
              'legend.fontsize': 12, # was 12 and before 10
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

def get_model_data(model_type, model_lab, website_path, data_path):

    print(model_type)
    print(model_lab)

    #Load model corresponding to model_type and lab
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    seedID = model_file['best seed']

    #Load model corresponding to model_type and lab
    with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    #Extract the inputs of the model
    model = model_file['model']
    features = model_file['json']
    columns = model_file['columns']
    imputer= model_file['imputer']
    test = model_file['test']

    if model_type == 'mortality':
        y = test['Outcome']
    else:
        y=test['Swab']

    X = test.iloc[:,0:(len(test.columns)-1)]




def feature_importance(model_type, model_lab, website_path, data_path, save_path,
                       feature_limit = 100, latex = True, dependence_plot = False,
                       data_filter = None, suffix_filter = ''):
    assert model_type in('mortality','infection'), "Invalid outcome"
    assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

    ## Load model corresponding to *model_type* and *model_lab*.

    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        best_model = pickle.load(file)

    seedID = best_model['best seed']

    #Load model corresponding to model_type and lab
    with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    model = model_file['model']
    data = model_file['train']
    data_test = model_file['test']

    ## Load data: to be replaced once we store X_train and X_test. Currently no imputation

    if data_filter != None:
        print("Raw Data Rows = "+str(data.shape[0]))
        print("Applying filter: "+data_filter)
        data = data.query(data_filter)
        data_test= data.query(data_test)
        print("Filtered Data  Rows = "+str(data.shape[0]))

    if model_type == "mortality":
        X = data.drop(["Outcome"], axis=1, inplace = False)
        y = data["Outcome"]
        X_test = data_test.drop(["Outcome"], axis=1, inplace = False)
        y_test = data_test["Outcome"]

    else:
        X = data.drop(["Swab"], axis=1, inplace = False)
        y = data["Swab"]
        X_test = data_test.drop(["Swab"], axis=1, inplace = False)
        y_test = data_test["Swab"]

    ## Calculate SHAP values (for each observation x feature)
    explainer = shap.TreeExplainer(model,
                                   data=X_test,
                                   );
    shap_values = explainer.shap_values(X);

    # Recode column names
    X.columns = [title_mapping[i] for i in X.columns]




    # Write complete grid as in original SHAP paper

    # Old way with gridspec
    #  fig = plt.figure(constrained_layout=True)
    #  gs = fig.add_gridspec(2, 4)
    #  gs.update(wspace=0.5)  # Reduce space

    # New way with just subplots
    max_display = np.minimum(len(X.columns) - 1, 9)   # -1 because we remove age
    n_cols = 3
    if latex:
        latexify(columns=1)


    fig, axs = plt.subplots(np.ceil(max_display/n_cols).astype(np.int64), n_cols,
                            figsize=(10, 10), constrained_layout=True
                            #  facecolor='w', edgecolor='k'
                            )

    axs = axs.ravel()  # flatten

    # Sort shap values indices to get most important ones
    sort_idx = np.argsort(-np.abs(shap_values).mean(0))

    # Get features to display. Remove age
    feat_display = np.array(X.columns)[sort_idx]
    feat_display = np.delete(feat_display, np.argwhere(feat_display == "Age"))

    for idx in range(max_display):
        ax = axs[idx]
        feat = feat_display[idx]
        shap.dependence_plot(feat,
                             shap_values, X, ax=ax,
                             interaction_index="Age",
                             xmin="percentile(1)", xmax="percentile(99)",
                             #  alpha=0.5,
                             #  x_jitter=0.1,
                             dot_size=3,
                             show=False)
        ax.set_ylabel("SHAP value")
        ax.set_ylim([-2, 2])
        if feat.split(" ", 1)[0] in ['CRP']:
            ax.set_xscale('log')

        # Remove colorbar
        ax.collections[0].colorbar.remove()

        #  import ipdb; ipdb.set_trace()
        ax.hlines(y=0, xmin=X[feat].min(), xmax=X[feat].max(),
                  color="#cccccc", lw=0.5, linestyle="dotted", zorder=-1)

    #  fig.subplots_adjust(right=0.8)


    # Plot colorbar on the right
    cb = fig.colorbar(ax.collections[0], ax=axs.flat)
    cb.set_label("Age", size=13)
    cb.ax.tick_params(labelsize=11)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #  cb.ax.set_aspect((bbox.height - 0.8) * 20)
    cb.ax.set_aspect((bbox.height - 0.8) * 10)


    print("Number of samples %d" % len(X))

    fig.savefig(os.path.join(save_path, 'feature_plot' + suffix_filter+'.pdf'),  bbox_inches='tight')

    plt.close()
    shap.summary_plot(shap_values, X, show=False,
                      max_display=10,
                      plot_size=(10, 5),
                      feature_names=[c[:c.find("(")] if c.find("(") != -1 else c for c in X.columns],
                      plot_type="violin")
    f = plt.gcf()
    f.savefig(os.path.join(save_path, 'summary_plot' + suffix_filter + '.pdf'), bbox_inches='tight')
    plt.clf()



  #
    #  # Test
    #  ax1 = fig.add_subplot(gs[1, 0:2])
    #  shap.summary_plot(shap_values, X, show=False,
    #                    feature_names=ft_recode,
    #                    max_display=feature_limit,
    #                    plot_type="violin",
    #                    ax=ax1)


    #  import ipdb; ipdb.set_trace()








    ## Summarize SHAP values across all features
    # This acts as an alterative to the standard variable importance plots. Higher SHAP values translate to higher probability of mortality.
    #  plt.close()
    #  if latex:
    #      latexify(columns=2)
    #  shap.summary_plot(shap_values, X, show=False,feature_names=ft_recode, max_display=feature_limit,
    #                    plot_type = "violin", plot_size = .7)
    #  f = plt.gcf()
    #  f.savefig(save_path+'summary_plot_top'+str(feature_limit)+suffix_filter+'.pdf', bbox_inches='tight')
    #  plt.clf()
    #  plt.close()

    ## Deep-dive into individual features
    # For a given feature, see how the SHAP varies across its possible values.
    # The interaction_index lets you choose a secondary index to visualize.
    # If omitted, it will automatically find the variable with the highest interaction.

    #  if dependence_plot:
    #      for i in X.columns:
    #          plt.close()
    #          if latex:
    #              latexify(columns=2)
    #          shap.dependence_plot(i, shap_values, X, show=False)
    #          f = plt.gcf()
    #          f.savefig(save_path+'dependence_plot/'+i+suffix_filter+'.pdf', bbox_inches='tight')
    #          plt.clf()
    #          plt.close()

def feature_importance_website(model_type, model_lab, website_path, data_path, save_path,
                       feature_limit = 10):
    assert model_type in('mortality','infection'), "Invalid outcome"
    assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

    ## Load model corresponding to *model_type* and *model_lab*.
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        best_model = pickle.load(file)

    seedID = best_model['best seed']

    #Load model corresponding to model_type and lab
    with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    model = model_file['model']
    data = model_file['train']

    ## Load data: to be replaced once we store X_train and X_test. Currently no imputation
    if model_type == "mortality":
        X = data.drop(["Outcome"], axis=1, inplace = False)
        y = data["Outcome"]
    else:
        X = data.drop(["Swab"], axis=1, inplace = False)
        y = data["Swab"]

    ## Calculate SHAP values (for each observation x feature)
    explainer = shap.TreeExplainer(model);
    shap_values = explainer.shap_values(X);

    ft_recode = []
    for i in X.columns:
        ft_recode.append(title_mapping[i])

    ## Summarize SHAP values across all features
    # This acts as an alterative to the standard variable importance plots. Higher SHAP values translate to higher probability of mortality.
    plt.close()
    shap.summary_plot(shap_values, X, show=False,feature_names=ft_recode, max_display=feature_limit, plot_type = "violin")
    f = plt.gcf()
    f.savefig(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.jpg', bbox_inches='tight')
    plt.clf()
    plt.close()

#%% Evaluate drivers of individual predictions
# # Select index j for prediction to generate.

# shap_values = explainer.shap_values(X)
# shap.force_plot(explainer.expected_value, shap_values, X, link = "logit")

# j=20
# plot = shap.force_plot(explainer.expected_value, shap_values[j], X.iloc[[j]] , link="logit")
# plot

# #%%  As an alternative view, you can trace a 3D plot of the values and their impact on the prediction.
# # shap.decision_plot(explainer.expected_value, shap_values[j], X.iloc[[j]], link='logit')
