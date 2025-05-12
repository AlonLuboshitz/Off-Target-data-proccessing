'''
Module to interpret data and models
'''
import pandas as pd
import numpy as np
import os
import shap
from file_utilities import create_folder
from features_and_model_utilities import get_feature_name
from plotting import plot_subplots, sub_plot_shap_beeswarn
from train_and_test_utilities import keep_intersect_guides_indices
from interpertation_utilities import *
from scipy.stats import pearsonr
from plotting_utilities import return_colormap


import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
##################### MODEL INTERPERTABILITY #####################

##################### SHAP #####################
def shap_partition():
    X_sub = X_train[:, 600:608]
    X_test_sub = X_test[:, 600:608]

    # Compute clustering on these 8 features
    link = linkage(X_sub.T, method="ward", metric="correlation")

    # Masker and explainer only for last 8 features
    masker = shap.maskers.Partition(X_sub, clustering=link)
    explainer = shap.Explainer(lambda x: model.predict(np.hstack([X_test[:, :600], x])), masker)

    # Explain
    shap_values = explainer(X_test_sub)
    shap.plots.waterfall(shap_values[0])

def get_shaply_values(model, x_background, explainer_type, x_selected = None, only_seq = False):
    """
    Computes SHAP values for the given model using the specified explainer type.
    
    Args:
        model: Trained model to explain.
        x_background (numpy.ndarray or pandas.DataFrame): Background dataset for SHAP.
        explainer_type (str): Type of SHAP explainer to use ('deep', 'gradient', 'kernel').
        x_selected (numpy.ndarray or pandas.DataFrame, optional): Selected dataset to explain.
    
    Returns:
        shap_values (list of numpy arrays): Computed SHAP values for each output class (or regression target).
    """
    
    class SHAPModelWrapper(tf.keras.Model):
        def __init__(self, model, encoded_length=600):
            super().__init__()
            self.model = model
            self.encoded_length = encoded_length
            self.inputs = model.inputs
            self.outputs = model.outputs
        def call(self, X):
            if isinstance(X, list) and len(X) == 1:
                X = X[0]
            if not only_seq:
                X = extract_features(X, encoded_length=self.encoded_length)

            return self.model(X)  # Use this, NOT .predict()

        def predict(self, X, **kwargs):
            return self.call(X).numpy()
    def model_wrapper(X):
        num_of_points = len(X)
        if not only_seq:
                
            X = extract_features(X, encoded_length= 600)
        if isinstance(model,list):
            predictions = np.zeros((len(model),num_of_points))
            for index,model_ in enumerate(model):
                predictions[index] = model_.predict(X).ravel()
            predictions = predictions.mean(axis=0)
            return predictions 
        return model.predict(X)
    if explainer_type == 'deep': # doesnt work
        deep_shap = SHAPModelWrapper(model=model,encoded_length=600)
        explainer = shap.DeepExplainer(deep_shap, x_background)
    elif explainer_type == 'gradient':
        explainer = shap.GradientExplainer(model, x_background)
    elif explainer_type == 'kernel':
        x_background = np.random.permutation(x_background)
        x_background = x_background[:10000]
        explainer = shap.KernelExplainer(model_wrapper, x_background)
    else:
        print('using permutation explainer')
        explainer = shap.PermutationExplainer(model_wrapper, x_background, max_evals = 1217,max_samples=1000) # 608 *2 +1
    if x_selected is None:
        x_selected = x_background
    shap_values = explainer(x_selected)
    
    return shap_values

def transform_to_heatmap(shap_values, seqeunce_length, bits_per_base, additional_features_length = 0):
    """
    Transforms SHAP values into a 2D matrix for heatmap representation.
    """
    if isinstance(shap_values,shap.Explanation):
        shap_values = shap_values.values
    min_shap = shap_values.min()
    max_shap = shap_values.max()
    epigenetics_values = None
    if additional_features_length > 0: # split the shap values to sequence and epigenetics
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        sequence_values = shap_values[:,:seqeunce_length * bits_per_base]
        epigenetics_values = shap_values[:,seqeunce_length * bits_per_base:]
    else: sequence_values = shap_values
    if sequence_values.ndim == 1:
        sequence_values = sequence_values.reshape(1,seqeunce_length , bits_per_base)
    elif sequence_values.ndim == 2:
        sequence_values = sequence_values.reshape(sequence_values.shape[0],seqeunce_length , bits_per_base)
    else:
        raise ValueError("SHAP values should be 1D or 2D")
    return sequence_values, epigenetics_values, min_shap, max_shap

def run_shap(model_path, background_data_path, explainer_type, output_path, explain_data_path = None,
             num_of_points=None, specific_indices=None, specific_guides=None,
             only_seq=False, plot_all_guides = True, plot_single_guides = True, 
             set_background_from_explain = False):
    '''
    Runs on the data and model given and extract shap values
    Plots the bars, beeswarm and waterfall plots
    
    Args:
        model_path (str): path to the model/folder of models
        data_path (str): path to the data
        explainer_type (str): type of the explainer to use
        output_path (str): path to save the plots
        num_of_points (int, optional): Number of first indices to use from x_background.
        specific_indices (list, optional): Specific indices to extract from x_background.
        specific_guides (list, optional): Specific guides to extract from x_background.
        only_seq (bool, optional) defualt True: If True, only the sequence features will be used otherwise split to sequence and epigenetics.
    '''
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    
    if not (plot_all_guides or plot_single_guides):
        raise ValueError("At least one of the plot options should be True")
    models,model_path = get_model(model_path,"deep")
    model = models[0]
    
    x_background,y,guides,otss_dict = get_data(background_data_path,features)
    
    if explain_data_path:
        x_explain,y,guides,otss_dict = get_data(explain_data_path,features)
        if set_background_from_explain:
            x_background = x_explain
    else: x_explain = x_background
    if specific_guides is None:
        specific_guides = guides
        guide_idx = keep_intersect_guides_indices(guides,specific_guides)
    
    whole_background = np.concatenate(x_background)
    whole_selected = []
    additional_features = whole_background.shape[1] - 600 if not only_seq else 0
    shap_vals = []
    output_path = create_folder(output_path,'SHAP_values')

    for idx in guide_idx:
        sgrna = specific_guides[idx]
        sg_x_background = x_explain[idx]
        sg_y = y[idx]
        sg_otss = otss_dict[sgrna]
        sg_x_selected, sgrna_otss = filter_data_for_interpertation(sg_x_background, sg_y, sg_otss, number_of_points=num_of_points)
        whole_selected.append(sg_x_selected)
        # NOTE: Background set to the whole data
        print(f'shap vals for {sgrna}')
        shap_values = get_shaply_values(models, whole_background, explainer_type, sg_x_selected, only_seq=only_seq)
        shap_vals.append(shap_values)
        np.save(os.path.join(output_path,f'{sgrna}.npy'), shap_values.values)
            
    # all explaination togther:
    whole_selected = [x[:100] for x in whole_selected] # get first 100 samples
    whole_selected = np.concatenate(whole_selected)
    shap_values = get_shaply_values(models, whole_background, explainer_type, whole_selected, only_seq=only_seq)
    shap_vals.append(shap_values)
    np.save(os.path.join(output_path,f'all_guides.npy'), shap_values.values)
    guides.append('All_guides')
    features = [get_feature_name(feature) for feature in features]
    #return shap_vals, features, output_path, guides
    plot_shap_only_epigenetics(shap_vals, output_path = output_path, 
                               feature_names = features, sgrna_otss= guides)

def plot_shap_only_epigenetics(shap_values, output_path, feature_names, sgrna_otss = None):
    '''
    Given a list of shap.explantions, extract the shap values for the epigenetic features and plot them.
    '''
    if isinstance(shap_values,list):
        shap_values = [convert_shap_to_shap_epi(shap_vals,feature_names=feature_names)for shap_vals in shap_values]
        if not sgrna_otss:
            sgrna_otss = [f'Guide {i+1}'for i in range(len(shap_values))]
        sub_plot_shap_beeswarn(shap_values,sgrna_otss,output_path)
        # Plot box plot, and absmean 
        
    # if isinstance(shap_values,shap.Explanation):
    #     shap_values = shap_values.values
    # num_of_features = len(feature_names)
    # if shap_values.ndim == 1:
    #     shap_values = shap_values.reshape(1, -1) 
    # epigenetic_values = shap_values[:,-num_of_features:]
    # np.save(os.path.join(output_path,'epigenetic_vals.npy'),epigenetic_values)

def convert_shap_to_shap_epi(shap_values, feature_names):
    """
    Create a shap explanantion object for the epigenetic features from a given shap explantation object.
    The feature_names should match the order of the feature in the shap object.

    Args:

    Returns:

"""
    feature_number = len(feature_names)
    if feature_number > shap_values.values.shape[1]:
        raise RuntimeError("number of features is bigger than shap values")
    subset_shap = shap.Explanation(
    values=shap_values.values[:, -feature_number:],
    base_values=shap_values.base_values,
    data=shap_values.data[:, -feature_number:],
    feature_names=feature_names  # custom names
)
    return subset_shap


    

def shap_for_epigenetic_only(model_path,data_path,explainer_type,output_path,
             num_of_points=None,specific_indices=None, specific_guides=None,
             only_seq=False, plot_all_guides = True, plot_single_guides = True):
    dis_file = pd.read_csv('/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/Change-seq/Epigenetic_disterbution.csv')
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    column_features = {col: get_feature_name(col) for col in dis_file.columns}
    epigenetic_disterbution_file = dis_file.rename(columns=column_features)
    features = [get_feature_name(feature) for feature in features]
    epi_vector = np.zeros(len(features))
    for i,feature in enumerate(features):
        epi_vector[i] = epigenetic_disterbution_file[feature].values[0]     

def plot_shap(shap_values, sgrna_otss, output_path,  number_of_epigenetic_features = 0, epigenetic_features = None,
              only_epigenetics = False):
    '''
    Plot the shap values for sgRNA and off-target sequences and the epignetics.
    If only epigenetics plot box plots for epigentic for the shap values.

'''
    if only_epigenetics:
        plot_shap_only_epigenetics(shap_values,output_path=output_path,feature_names=epigenetic_features)
        
        # plot only epigenetic shap values
    row_labels, x_ticks = nucleotides_for_heatmap()
    if number_of_epigenetic_features > 0 and len(epigenetic_features) == number_of_epigenetic_features:
        kwargs = {'additional_vector_y':"Epigenetic\nfeatures",'additional_vector_x':epigenetic_features}
    # # Plot first 10 samples
    # single_shap_values = shap_values[:10] 
    # sequence_shap_values, epigenetic_shap_values, min_shap,max_shap  = transform_to_heatmap(single_shap_values, 24,25,additional_features)
    # kwargs = {'vmin': min_shap, 'vmax': max_shap,'cbar': 'SHAP values','additional_vector': epigenetic_shap_values}
    # plot_subplots(sequence_shap_values,plot_types='heatmap',titles=None, x_label="Position", y_label="Nucleotides",x_ticks=x_ticks,
    #                 y_ticks=row_labels, output_path=output_path, general_title="10-SHAP values all_bg",sgrna_otss=sgrna_otss,**kwargs)
    # Summarized plot of all samples
    
    summarized_shap_values = np.sum(shap_values.values, axis=0)
    sequence_shap_values, epigenetic_shap_values, min_shap,max_shap  = transform_to_heatmap(summarized_shap_values, 24,25,number_of_epigenetic_features)
    data = [(sequence_shap_values,epigenetic_shap_values)]
    kwargs.update({'vmin': min_shap, 'vmax': max_shap,'cbar': 'SHAP values'})

    plot_subplots(data,plot_types='heatmap',titles=None, x_label="Position", y_label="Nucleotides",x_ticks=x_ticks,
                    y_ticks=row_labels, output_path=output_path, general_title="Summed-SHAP values all_bg",sgrna_otss=None,**kwargs)
    # Abs mean
    abs_mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
    sequence_shap_values, epigenetic_shap_values, min_shap,max_shap  = transform_to_heatmap(abs_mean_shap_values, 24,25,number_of_epigenetic_features)
    data = [(sequence_shap_values,epigenetic_shap_values)]
    kwargs.update({'vmin': min_shap, 'vmax': max_shap,'cbar': 'SHAP values'})
    plot_subplots(data,plot_types='heatmap',titles=None, x_label="Position", y_label="Nucleotides",x_ticks=x_ticks,
                        y_ticks=row_labels, output_path=output_path, general_title="AbsMean-SHAP values all_bg",sgrna_otss=None,**kwargs)

def main_shap():
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1"
    seq_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1/model_1.keras"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    background_data = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/vivo-silico-78_withEpigenetic.csv"
    explainer_type = ""
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 200
    run_shap(model_path=epi_model_path,background_data_path = background_data,explain_data_path=test_data_path,explainer_type=explainer_type,
                output_path=output_path,num_of_points=number_of_points,specific_guides=specific_guides,only_seq=False,plot_all_guides=False)
    # shap_for_epigenetic_only(model_path=epi_model_path,data_path=test_data_path,explainer_type=explainer_type,
    #          output_path=output_path,num_of_points=number_of_points,specific_guides=specific_guides,only_seq=False,plot_all_guides=False)
##################### Gradient asecnt #####################
def main_gradient_ascent():
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1/model_1.keras"
    seq_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1/model_1.keras"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 10
    run_gradient_asecnt(model_path=seq_model_path,data_path=test_data_path,output_path=output_path,
                        num_of_points=number_of_points,specific_guides=specific_guides,only_seq=True)

def run_gradient_asecnt(model_path, data_path, output_path,
             num_of_points=None,specific_indices=None, specific_guides=None,
             only_seq=False):
    from models import replace_argmax_layer
    models = get_model(model_path,"deep")
    model = models[0]
    model = replace_argmax_layer(model)
    print(model.summary())
    x = np.random.randint(2,size=(1,600),dtype=np.int8)
    alpha = 0.1
    
    # Convert input to trainable variable
    #input_var = tf.Variable(x, dtype=tf.float32)
    for i in range(10):
        grad = get_gradients(model, x)
        x += grad * alpha
    temp = x.numpy()
    seq = temp[0, :-11]
    seq = seq.reshape(32, 4)
    # Gradient ascent loop




# import logomaker
# import tensorflow as tf
# from matplotlib import pyplot as plt
# from CRISPRepi import *


def get_gradients(model, input_data):
    input_data = tf.convert_to_tensor(input_data,dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        preds = model(input_data,training=True)
    grads = tape.gradient(preds, input_data)
    
    return grads


# def saliency_map(model, epigenetics):
#     x1 = np.ones(shape=(32, 4)) * 0.25  # Initial input matrix
#     x1[21:23] = [0, 0, 1, 0]  # GG
#     x2 = np.ones(shape=(1, 11))
#     x2[:, 10] = 100  # CRISPROn
#     x2[:, 3] = avgEpi(epigenetics)  # methylation
#     x1 = x1.reshape(1, -1)
#     x = np.concatenate((x1, x2), axis=1)
#     lr = 0.1
#     # lr = 0.1, range(50000)
#     for i in range(50000):
#         grads = get_gradients(model,  x)
#         x += grads * lr
#     temp = x.numpy()
#     seq = temp[0, :-11]
#     seq = seq.reshape(32, 4)
#     seqDF = pd.DataFrame(seq, columns=['A', 'C', 'G', 'T'])
#     seqDF = seqDF.div(seqDF.sum(axis=0), axis=1)
#     save_logo(seqDF)
#     epi = x[:, -11:].numpy()
#     epi[:, 10] /= 100
#     #epi = np.array([[1.60, 1.14, 1.68, -1.65, 1.23, 1.31, -0.15, 1.44, -0.34, 0.93, 1.00]])
#     visualize_integrated_gradients(seq, 1)

#     visualize_integrated_gradients(epi, 0)


# def visualize_integrated_gradients(integrated_gradients, segFlag):
#     colors = ['white', 'white', 'white', 'black', 'white', 'white', 'black', 'white', 'black', 'white', 'white']
#     if segFlag:
#         integrated_gradients = np.flip(np.rot90(integrated_gradients, k=-1), axis=1)
#     else:
#         num_rows, num_cols = integrated_gradients.shape
#         for i in range(num_rows):
#             for j in range(num_cols):
#                 color = colors[j]
#                 plt.text(j, i, f'{integrated_gradients[i, j]:.2f}', ha='center', va='center', color=color, fontsize=10)
#     plt.imshow(integrated_gradients, cmap='Blues', interpolation='nearest', vmin=-2, vmax=2)
#     plt.colorbar(label='Feature importance')  # Add color bar
#     plt.axis('off')
#     plt.show()


# def save_logo(df):
#     IG_logo = logomaker.Logo(df)
#     IG_logo.ax.set_xticks(range(32))
#     IG_logo.ax.set_xticklabels(np.arange(1, 33), fontsize=12)
#     IG_logo.ax.set_ylabel('Importance score', fontsize=14)
#     plt.show()

# def format_titles(sgrna_otss):
#     """
#     Formats the titles for each sgRNA-OT pair to ensure alignment.
    
#     Parameters:
#         sgrna_otss (list of tuples): Each tuple contains (sgRNA, OT) sequences.
    
#     Returns:
#         list of str: Formatted titles with aligned labels.
#     """
#     labels = ["sgRNA:", "OT:"]
#     max_label_length = max(len(label) for label in labels)  # Ensures equal prefix length

#     # Generate aligned titles
#     titles = [
#         f"{'sgRNA:'.ljust(max_label_length)} {pair[0]}\n"
#         f"{'OT:'.ljust(max_label_length)} {pair[1]}"
#         for pair in sgrna_otss
#     ]

#     return titles 
    
##################### Epigenetics #####################   
def epigenetic_importance_for_offtargets_pert_05(sg_x, features, model):
    '''
    Calculate the epigenetic importance for all off-targets both pertubation and 05 analysis.
    Args:
        sg_x (np.array): All sgRNA-OT pairs.
        features (list): List of epigenetic features.
        model (tf.keras.Model): Model to interpret.
    Returns:
        2 important lists:
        mean_pertubation_importance_list (list): List of dictionaries of mean pertubation importance for each feature.
        importance_05_list (list): List of dictionaries of 0.5 importance for each feature.
    '''
    mean_pertubation_importance_list = []
    importance_05_list = []
    # Loop over off-target vectors
    for off_target_vector in sg_x:
        # Get perturbation importance and mean values for each feature
        pertubation_importance = epigenetic_pertubation_importance(features, off_target_vector, model)
        mean_pertubation_importance = {key: np.mean(value) for key, value in pertubation_importance.items()}
        
        # Get the 05 importance values for each feature
        importance_05 = epigenetic_05_vector(features, off_target_vector, model)
        
        # Append the dictionaries to their respective lists
        mean_pertubation_importance_list.append(mean_pertubation_importance)
        importance_05_list.append(importance_05)
    return mean_pertubation_importance_list,importance_05_list

def pertubation_and_05_importance(sg_x_selected, features, model, num_of_points):
    '''
    Calculate the epigenetic importance for all off-targets both pertubation and 05 analysis.
    Args:
        sg_x_selected (np.array): Selected sgRNA-OT pairs.
        features (list): List of epigenetic features.
        model (tf.keras.Model): Model to interpret.
        num_of_points (int): Number of points that sampled.
    Returns:
        epigenetic_importance_arrays (dict): Dictionary of 2D arrays for each feature.
        1 row: pretubation.
        2 row: 0.5 importance.
    '''
    epigenetic_importance_arrays = {feature: np.zeros((2, num_of_points)) for feature in features}

    mean_pertubation_importance_list, importance_05_list = epigenetic_importance_for_offtargets_pert_05(sg_x_selected, features, model)
    epigenetic_importance_arrays = convert_importance_dicts_to_2d_arrays(epigenetic_importance_arrays,
                                                                            mean_pertubation_importance_list, importance_05_list)
    return epigenetic_importance_arrays


def run_epigenetics(model_path, features, output_path = None, guide_list = None, mismatch_limit = 3,
                    save_model_scores = True, use_model_scores = False,
                    epigenetic_disterbution_path = None):
    '''
    This function will run epigenetic interpertation on the given model and guides.
    For each guide it will create syntethic off-targets with all optional mismatches.
    Than for each off-target it will add an epigenetic vector where all the epigenetic features are 1 and 0.
    The diffrenences in model prediction for all off targets i.e. the prediction with 1 in the epigenetic feature - the prediction with 0
    in the epigenetic feature will be box plotted.
    The model predictions can be saved for further extraction and avoiding running the model again.
    The delta in predictions can be saved aswell.
    
    The function will create 2 folders:
        by_mismatch: subplot each guide with the same number of mismatches
        by_guide: subplot each mismatch number with the same guide.
    
    Args:
        model_path (str): path to the model/folder of models
        features (list): List of epigenetic features.
            The feature list should match the feature assignmnet in the model!
        output_path (str): path to save the plots, model predictions, and delta in predictions.
        guide_list (list): List of guides to use.
        mismatch_limit (int): Maximum number of mismatches to create.
        save_model_scores (bool): If True, save the model scores for each guide and mismatch.
        use_model_scores (bool): If True, use the model scores from the model_scores_path.
        epigenetic_disterbution_path (str): Path to the epigenetic distribution file.
        
    Returns:
        None
        
    '''
    def get_model_outputs_from_synthesized_ots(guide_list):
        # Create off targets
        guides_dict = create_off_targets_for_guides(guide_list=guide_list,mismatch_limit=mismatch_limit)
        # Sample data - 3,4,5,6 to many options
        
        # Create epigenetic vector and add it to the off targets
        
        
        epigenetic_disterbution_file = pd.read_csv(epigenetic_disterbution_path)
        # for clarity creating a new dictionary
        guide_dict_with_epigentics = add_epigenetic_vector_to_offtargets(guides_dict,features,epigenetic_disterbution_file)
        
        del guides_dict
        
        # Get scores
        model_outputs = get_model_scores_for_features(model_path=model_path,
                                                        guide_dict_with_epi_genetics=guide_dict_with_epigentics,
                                                        save_model_scores=save_model_scores,
                                                        output_path=output_path)
        return model_outputs
    #NOTE: add the creation of off targets that not included in the model outputs already.
    guide_list = ['GAAGGCTGAGATCCTGGAGGCGG','GAGAATCAAAATCGGTGAATCGG','GCTGGTACACGGCAGGGTCAAGG','GGACTGAGGGCCATGGACACAGG',
                  'GTCAGGGTTCTGGATATCTGTGG','GTCCCTAGTGGCCCCACTGTTGG']
    # guide_list = ['GATGCAGAGACCCTGCTCAACGG','GGGACTCTACATCTGCAAGGAGG','GCTTGTCCGTCTGGTTGCTGCGG','GCTGGCGATGCCTCGGCTGCAGG',
    #               'GCCCTGCTCGTGGTGACCGATGG','GGTGAGGGAGGAGAGATGCCTGG']
    mismatch_limit = 3
    features = [get_feature_name(feature) for feature in features]

    # Open data if given
    if use_model_scores:
        model_scores_path = os.path.join(output_path,'Model_scores','Raw_scores')
        model_outputs,guides_not_checked = open_model_outputs_file(model_scores_path, guide_list, mismatch_limit)
        if guides_not_checked is not None:
            print(f'Guides not checked: {guides_not_checked}')
            model_outputs.update(get_model_outputs_from_synthesized_ots(guides_not_checked))
    
    else:
        model_outputs = get_model_outputs_from_synthesized_ots(guide_list)
        
    # model outputs {guide: {mismatch_num: ensemble_score}}
    for guide, mismatch_dict in model_outputs.items():
        for mismatch_num, ensemble_score in mismatch_dict.items():
            model_outputs[guide][mismatch_num] = epi_feature_importance_from_model_output(features,ensemble_score)
    #save_importance_values(model_outputs, output_path)
    # importance_scores = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability/Model_scores/Importance_scores'
    # from_file = load_importance_values(importance_scores,guide_list,mismatch_limit)
    # Sub plot all guides togther by mismatch number.
    # Create {guide : {feature : [importance]}} by the same mismatch number
    color_map = return_colormap(features)
    
    mismatch_path = create_folder(output_path,'By_mismatch')

    for mismatch_num in range(1,mismatch_limit+1):
        model_outputs_by_mismatch = [model_outputs[guide][mismatch_num] for guide in model_outputs]
        titles = [f'{guide}' for guide in model_outputs]
        plot_epigenetic_importance_by_pertubation(model_outputs_by_mismatch,mismatch_path,colormap=color_map,title_prefix=f'{mismatch_num}_mismatch',titles=titles)
        
    # Sub plot all mismatch number by the same guide.
    guide_path = create_folder(output_path,'By_guide')
    for guide,mismatch_dict in model_outputs.items():
        data = [mismatch_dict[mismatch_num] for mismatch_num in range(1,mismatch_limit+1)]
        titles = [f'Mismatch_{mismatch_num}' for mismatch_num in range(1,mismatch_limit+1)]
        plot_epigenetic_importance_by_pertubation(data,guide_path,colormap=color_map,title_prefix=f'{guide}',titles=titles)




def old_epigenetics_function(model_path,data_path,output_path = None,
                     num_of_points = 200, specific_guides = None , features = None,
                     plot_single_guides = True, plot_all_guides = True):
    '''
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 200
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    run_epigenetics(model_path=epi_model_path,data_path=test_data_path,output_path=output_path,
                    num_of_points=number_of_points,specific_guides=specific_guides,features=features,
                    plot_single_guides=False, plot_all_guides=True,save_model_scores=False,use_model_scores=True,
                    model_scores_path='/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability/Model_scores')

    '''
    
    models,models_path = get_model(model_path,"deep", sample=5)
    
    x_background,y,guides,otss_dict = get_data(data_path,only_seq=True)
    if specific_guides is None:
        specific_guides = guides
    guide_idx = keep_intersect_guides_indices(guides,specific_guides)
    
    color_map = return_colormap(features)
    titles = [f'{os.path.basename(i).split(".")[0]}' for i in models_path]
    models_results = []
    for model in models:
        whole_guides_interpertation = []
        for idx in guide_idx:
            sgrna = specific_guides[idx]
            sg_x_background = x_background[idx]
            sg_y = y[idx]
            sg_otss = otss_dict[sgrna]
            sg_x_selected, sgrna_otss, additional_features = filter_data_for_interpertation(sg_x_background, sg_y, sg_otss,
                                                                                            only_seq=True, number_of_points=num_of_points )
            #NOTE: pertubation is correlated with 0.5 therefor running only 0.5
            #epigenetic_importance_arrays = pertubation_and_05_importance(sg_x_selected, features, model, num_of_points) 
            epigenetic_importance_arrays = epigenetic_05_vector(features, sg_x_selected, model)
            whole_guides_interpertation.append(epigenetic_importance_arrays)
            if plot_single_guides:
                temp_output = create_folder(output_path,sgrna)
                plot_epigenetic_importance_by_pertubation(epigenetic_importance_arrays, temp_output, color_map)
        whole_dict = {}
        for key in whole_guides_interpertation[0]:
            arrays = [d[key] for d in whole_guides_interpertation]
            whole_dict[key] = np.hstack(arrays)
        models_results.append(whole_dict)
    if plot_all_guides:
        # whole_dict = {}
        # for key in whole_guides_interpertation[0]:
        #     arrays = [d[key] for d in whole_guides_interpertation]
        #     whole_dict[key] = np.hstack(arrays)
        temp_output = create_folder(output_path,"All_guides")
        plot_epigenetic_importance_by_pertubation(models_results, temp_output, color_map,title_prefix='Combined models', titles=titles)
        # plot_epigenetic_importance_by_pertubation(whole_guides_interpertation, temp_output, color_map,title_prefix='Separated')

        

def open_model_outputs_file(model_scores_path, guide_list, mismatch_limit,
                            model_suffix = None):
    """
    Open the model outputs from the given path.
    If some guides are not included in the model scores path, the user will be prompted to compute their scores.
    
    Args:
        model_scores_path (str): Path to the model scores - folder/guides_mismatches_scores.npy
        guide_list (list): List of guides to extract from the model scores.
        mismatch_limit (int): Maximum number of mismatches to consider.
    """
    model_outputs = {}
    if not os.path.exists(model_scores_path):
        raise ValueError("Model scores path does not exist")
    guides_in_path = [i.split("_")[0] for i in os.listdir(model_scores_path) if "scores" in i]
    difference_guides = set(guide_list).difference(set(guides_in_path))
    if len(difference_guides) > 0:
        print(f'The guides:\n{difference_guides}\nare not in the model scores path.\nWould you like compute their scores?\n1: Yes\n2: No')
        answer = input()
        if answer == "1":
            difference_guides = list(difference_guides)
        else:
            pass
    else: # Read all the model scores
        difference_guides = None
        error_file = f'{model_scores_path}/error_file.txt'
        for guide in guide_list:
            mismatch_dict = {}
            for mismatch_num in range(1,mismatch_limit+1):
                temp_guide_str = f'{guide}_{mismatch_num}_scores.npy' if not model_suffix else f'{guide}_{mismatch_num}_{model_suffix}_scores.npy'
                temp_path = os.path.join(model_scores_path,temp_guide_str) 
                if not os.path.exists(temp_path):
                    print(f'{temp_guide_str} does not exist')
                    with open(error_file,'a') as f:
                        f.write(f'{temp_guide_str}\n')
                model_scores = np.load(temp_path)
                mismatch_dict[mismatch_num] = model_scores
            model_outputs[guide] = mismatch_dict
    return model_outputs, difference_guides  

    
def plot_epigenetic_importance_by_pertubation(epigenetic_importance_arrays, output_path, colormap=None, title_prefix = "",
                                              titles = None):
    '''
    Plots the epigenetic importance pertubation calculation.
    If the epigenetic_importance_arrays is a dictionary of 2D arrays - there are 2 calculations for each feature.
        1. pretutabion. 2. 0.5 importance. 
        it will plot 2 box plots of mean pertubation values and 0.5 values and correlation between the 2 set of values.
    If the epigenetic_importance_arrays is a dictionary of 1D arrays - there is only 1 calculation for each feature.
        It will plot a box plot of the values.
    Args:
        epigenetic_importance_arrays (dict): Dictionary of 1/2D arrays for each feature.
        output_path (str): path to save the plots
        colormap (dict): Colormap for the features.
    '''
    if isinstance(epigenetic_importance_arrays,list):
        first_guide_dict = epigenetic_importance_arrays[0]
        first_key, first_value = next(iter(first_guide_dict.items()))
    elif isinstance(epigenetic_importance_arrays,dict):
        first_key, first_value = next(iter(epigenetic_importance_arrays.items()))
        if isinstance(first_value, dict): # Key: {epigenetic: [values]}
            data = [pd.DataFrame(guide_dict) for guide_dict in epigenetic_importance_arrays.values()]
            titles = [f'{guide}' for guide in epigenetic_importance_arrays.keys()]
    kwargs = {'colormap': colormap, 'showfliers': False,'showmeans':False,"order_by":"median"}
    
    if first_value.ndim == 2:
        epigenetic_importance_cor = {key: pearsonr(val[0],val[1]) for key, val in epigenetic_importance_arrays.items()}
        plot_subplots(data=epigenetic_importance_cor,plot_types='correlation', titles=None,
                    x_label='Mean pertubation importance',y_label='05 importance',
            output_path=output_path,general_title='Mean vs 0.5 importance correlation')
        pert_dict = {key: val[0] for key, val in epigenetic_importance_arrays.items()}
        dict_05 = {key: val[1] for key, val in epigenetic_importance_arrays.items()}
        data = [pd.DataFrame(pert_dict), pd.DataFrame(dict_05)]
        titles = ['Mean pertubation importance', '0.5 importance']
        plot_subplots(data=data,plot_types='boxplot',titles=titles, x_label='Epigenetic marks',y_label=f'{chr(916)} Prediction',
                output_path= output_path,general_title=f'{title_prefix} Mean vs 0.5 importance boxplot',**kwargs)   
    else: # 1D arrays
        if isinstance(epigenetic_importance_arrays,list):
            data = [pd.DataFrame(guide_dict) for guide_dict in epigenetic_importance_arrays]
        else:
            data = [pd.DataFrame(epigenetic_importance_arrays)]

        plot_subplots(data=data,plot_types='boxplot',titles=titles, x_label='Epigenetic marks',y_label=f'{chr(916)} Prediction',
                output_path= output_path,general_title=f'{title_prefix} importance boxplot',**kwargs)
 
def main_epigenetics():
    #seq_path='/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1'
    all_epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1"
    all_epi_features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    
    all_model_score_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability/Model_scores/Raw_scores'
    h3k27ac_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/H3K27ac/ensemble_1"
    h3k27ac_features = ["H3K27ac_peaks_binary"]
    epi_model_path = all_epi_model_path
    features = all_epi_features
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    model_name = epi_model_path.split('Binary_epigenetics')[1].split('/')[1] # get epigenetics used in the model
    output_path = create_folder(output_path,model_name)
    epi_dis_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/Change-seq/Epigenetic_disterbution.csv'
    run_epigenetics(model_path=epi_model_path,output_path=output_path,
                    features=features, save_model_scores=True,use_model_scores=True,
                    epigenetic_disterbution_path=epi_dis_path)
if __name__ == "__main__":
    main_shap()
    
