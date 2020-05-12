import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


from math import sqrt 
import matplotlib

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

#%% 
def feature_importance(model_type, model_lab, website_path, data_path, save_path, title_mapping, 
                       feature_limit = 100, latex = False):
    assert model_type in('mortality','infection'), "Invalid outcome"
    assert model_lab in('with_lab','without_lab'), "Invalid lab specification"
    
    if latex:
        save_path = save_path+"latex_"

    ## Load model corresponding to *model_type* and *model_lab*.

    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)
        
    model = model_file['model']

    ## Load data: to be replaced once we store X_train and X_test. Currently no imputation
    
    data = pd.read_csv(data_path+model_type+"_"+model_lab+"/train.csv")
    if model_type == "mortality":
        X = data.drop(["Unnamed: 0", "Outcome"], axis=1, inplace = False)
        y = data["Outcome"]
    else:
        X = data.drop(["NOSOLOGICO","Swab"], axis=1, inplace = False)
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
    latexify(fig_width = 3.4)
    shap.summary_plot(shap_values, X, show=False,feature_names=ft_recode, max_display=feature_limit, plot_type = "violin")
    f = plt.gcf()
    f.savefig(save_path+'summary_plot_top'+str(feature_limit)+'.pdf', bbox_inches='tight')
    f.savefig(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.jpg', bbox_inches='tight')
    plt.clf() 
    plt.close()
    
 
    ## Deep-dive into individual features
    # For a given feature, see how the SHAP varies across its possible values. 
    # The interaction_index lets you choose a secondary index to visualize.
    # If omitted, it will automatically find the variable with the highest interaction.

    for i in X.columns:
        plt.close()
        latexify(fig_width = 3.4)
        shap.dependence_plot(i, shap_values, X,show=False)
        f = plt.gcf()
        f.savefig(save_path+'dependence_plot_'+i+'.pdf', bbox_inches='tight')
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