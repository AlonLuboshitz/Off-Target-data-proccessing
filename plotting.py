import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import pandas as pd
import seaborn as sns
import os
from file_utilities import create_paths
from plotting_utilities import *
import logomaker
import shap
#from file_management import File_management
#from features_engineering import get_epi_data_bw,get_epi_data_bed

def plot_n_rank(n_rank_values, n_tpr_arrays, titles, output_path, general_title):
    if len(n_rank_values) != len(n_tpr_arrays) != len(titles):
        raise ValueError('All input lists must have the same length.')
    #NOTE: check why tpr != 1
    n_tpr_arrays,n_rank_values,titles = argsort_by(n_rank_values, n_tpr_arrays,n_rank_values,titles,descending=True) 

    plt.figure(figsize=(8, 6))
    for i in range(len(n_rank_values)):
        x_values = np.arange(1, len(n_tpr_arrays[i]) + 1)
        plt.plot(x_values, n_tpr_arrays[i], lw=2, label=f'{titles[i]} (N-rank = {n_rank_values[i]:.2f})')
    plt.xlabel('Number of experiments', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve by N experiments')
    plt.legend(loc='lower right',fontsize=11)
    plt.grid(True)
    if not "N_rank" in general_title:
        general_title = general_title + "_N_rank"
    plt.tight_layout()  # Adjust layout to minimize whitespace
    plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory

def plot_last_tp(last_tp_index, last_tp_ratio, tpr_arrays, model_names,  information, general_title = None, 
                    ax=None, ax_title=None, output_path = None):
    one_pic = False
    if len(last_tp_index) != len(last_tp_ratio) != len(tpr_arrays) != len(model_names):
        raise ValueError('All input lists must have the same length.')
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        one_pic = True
        if ax_title is None:
            ax_title = "Last true positive index"
    # argsort in asecnding order by the last tp values
    last_tp_indices_sorted,tpr_arrays_sorted,titles_sorted = argsort_by(last_tp_index,last_tp_index, tpr_arrays,model_names )
    for i in range(len(last_tp_indices_sorted)):
        x_values = np.arange(1, len(tpr_arrays_sorted[i]) + 1)
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        ax.plot(x_values, tpr_arrays_sorted[i], lw=2, color=color, label=f'{titles_sorted[i]} (Last TP = {last_tp_indices_sorted[i]})')
        ax.axvline(x=last_tp_indices_sorted[i], color=color, lw=1, linestyle='--')
    
    ax.set_xlabel('Number of experiments', fontsize=14)
    ax.set_ylabel('True positive rate', fontsize=14)
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax.set_yticks(np.linspace(0, 1, num=6))  # Ensure tick customization if needed
    ax.set_title(ax_title)
    
    if information:
        label_text = '\n'.join([f'{key}: {value}' for key, value in information.items()])
        ax.plot([], [], ' ', label=label_text)  # Invisible line with empty style
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True)
    if one_pic:
        if not "Last_TP" in general_title:
            general_title = general_title + "_Last_TP"
        plt.tight_layout()  # Adjust layout to minimize whitespace
        plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
        plt.close()  # Close the figure to free memory
    
    

def plot_roc(fpr_list, tpr_list, aurocs, model_names,output_path,general_title,
              ax=None, ax_title=None, information=None, legend_title=None):
    """
    Plots the ROC curve for 1 or more models.
    
    Args:
        fpr_list (list): A list of false positive rates for each model.
        tpr_list (list): A list of true positive rates for each model.
        aurocs (list): A list of AUROC values for each model.
        model_names (list): A list of model names.
        output_path (str): The path to save the plot.
        general_title (str): The general title for the plot.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
        ax_title (str, optional): The title for the plot. If None, a default title is used.
        information (dict, optional): A dictionary of additional information to display on the plot.
    """
    if len(fpr_list) != len(tpr_list) != len(aurocs) != len(model_names):
        raise ValueError('All input lists must have the same length.')
    
    fpr_list, tpr_list, model_names, aurocs = argsort_by(aurocs, fpr_list, tpr_list, model_names, aurocs, descending=True)
    one_pic = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        one_pic = True
        if ax_title is None:
            ax_title = "Receiver Operating Characteristic (ROC) Curve"
    
    for i in range(len(fpr_list)):
        ax.plot(fpr_list[i], tpr_list[i], lw=2, label=f'{model_names[i]} (AUC = {aurocs[i]:.4f})')
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Baseline = 0.5')
    if information:
        label_text = '\n'.join([f'{key}: {value}' for key, value in information.items()])
        ax.plot([], [], ' ', label=label_text) # Invisible line with empty style
    ax.set_xlabel('False positive rate', fontsize=18)
    ax.set_ylabel('True positive rate', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_title(ax_title)
    
    ax.legend(loc='lower right', fontsize=14,title=legend_title, title_fontsize=15)
    
    ax.grid(True)
    if one_pic:
        if not "AUROC" in general_title:
            general_title = general_title + "_AUROC"
        plt.tight_layout()
        plt.savefig(output_path + f"/{general_title}.png", dpi=300)
        plt.close()
    


def plot_pr(recall_list, precision_list, auprcs, model_names, output_path, general_title,
            ax=None, ax_title=None,information=None, legend_title=None):
    """
    Plots the Precision-Recall curve for 1 or more models.
    
    Args:
        recall_list (list): A list of recall values for each model.
        precision_list (list): A list of precision values for each model.
        auprcs (list): A list of AUPRC values for each model.
        model_names (list): A list of model names.
        output_path (str): The path to save the plot.
        general_title (str): The general title for the plot.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
        ax_title (str, optional): The title for the plot. If None, a default title is used.
        information (dict, optional): A dictionary of additional information to display on the plot.
    
    ----------
    Show the figure and saves it.
    """
    if len(recall_list) != len(precision_list) != len(auprcs) != len(model_names):
        raise ValueError('All input lists must have the same length.')
    auprcs_ = [auprc[0] for auprc in auprcs]
    recall_list, precision_list, auprcs, model_names = argsort_by(auprcs_,  recall_list, precision_list, auprcs,model_names,descending=True)
    one_pic = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        one_pic = True
        if ax_title is None:
            ax_title = "Precision-Recall Curve"
    
    for i in range(len(recall_list)):
        ax.plot(recall_list[i], precision_list[i], lw=2,label=f'{model_names[i]} (AUC = {auprcs[i][0]:.3f})')
    baseline_val = auprcs[0][1]

    # Plot a horizontal dashed line at baseline
    ax.axhline(
        y=baseline_val,color='gray', linestyle='--', lw=2,
        label=f"Baseline = {baseline_val:.5f}"
    )
    
    if information:
        label_text = '\n'.join([f'{key}: {value}' for key, value in information.items()])
        ax.plot([], [], ' ', label=label_text)  # Invisible line with empty style
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_title(ax_title)
    ax.legend(loc='upper right', fontsize=14,title=legend_title, title_fontsize=15)

    ax.grid(True)
    if one_pic:
        if not "AUPRC" in general_title:
            general_title = general_title + "_AUPRC"
        plt.tight_layout()  # Adjust layout to minimize whitespace
        plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
        plt.close()  # Close the figure to free memory
def plot_correlation(x, y, x_axis_label, y_axis_label, r_coeff, p_value, title, output_path,ax = None,text_size = 16):
    '''This function plots a scatter plot with a linear regression line, and adds the correlation coefficient and p-value to the plot.
    Args:
    1. x: A numpy array representing the x values.
    2. y: A numpy array representing the y values.
    3. x_axis_label: A string representing the x-axis label.
    4. y_axis_label: A string representing the y-axis label.
    5. r_coeff: A float representing the correlation coefficient.
    6. p_value: A float representing the p-value.
    7. title: A string representing the title of the plot.
    8. output_path: A string representing the output path for saving the plot.
    
    ----------
    Show the figure and saves it.'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color='blue')
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    ax.set_title(title,fontsize=16)
    ax.grid(True)
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(y_axis_label, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    def sci_str(x, digits=2):
        m, e = f"{x:.{digits}e}".split("e")
        return rf"${float(m):.{digits}f} \cdot 10^{{{int(e)}}}$"
    num_of_points = len(x)
    # Adding text with correlation coefficient, p-value, and number of points
    ax.text(0.08, 0.75, f'r = {r_coeff:.2f}\np = {sci_str(p_value,2)}\nn = {num_of_points}', 
            fontsize=text_size, ha='left', va='center', transform=ax.transAxes)
    
def plot_logo(counts_df, ax=None, ax_title=None, output_path=None,y_label=None, x_label=None):
    """
    Plots a sequence logo using the provided counts DataFrame.
    """
    one_pic = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        one_pic = True
        if ax_title is None:
            ax_title = "Logo Plot"
    
    logo = logomaker.Logo(counts_df, ax=ax, color_scheme='classic')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel(y_label, fontsize=12)
    logo.ax.set_xlabel(x_label, fontsize=12)
    logo.ax.set_xticks(range(len(counts_df)))
    logo.ax.set_xticklabels(range(1,len(counts_df)+1), fontsize=12)
    ax.set_title(ax_title, fontsize=14)
    if one_pic:
        plt.tight_layout()
        plt.savefig(output_path + f"/{ax_title}.png", dpi=300)
        plt.close()

    


def plot_heatmap(data, ax=None, row_labels=None, col_labels=None, 
                 x_label=None, y_label=None, title=None, output_path=None, 
                 cbar = None, vmin = None, vmax = None, sgrna_ots = None,
                 additional_vector_y = None, additional_vector_x = None):
    """
    Plots a heatmap on a given subplot axis or creates a new figure if no axis is provided.

    Parameters:
        data (np.ndarray): 1D or 2D array to plot as a heatmap.
        ax (matplotlib.axes.Axes, optional): The subplot axis to plot on. If None, creates a new figure.
        row_labels (list, optional): Labels for rows.
        col_labels (list, optional): Labels for columns.
        x_label (str, optional): Label for the x-axis.
        y_label (str, optional): Label for the y-axis.
        title (str, optional): Title of the heatmap.
        output_path (str, optional): If provided, saves the plot to this path.
        cbar (str, optional): If provided, adds a label to the colorbar.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        if len(data) <=1:
            raise ValueError("Data list must contain more than one element for heatmap")
        additional_vector = data[1]
        data = data[0]
    
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a NumPy array")
    if data.ndim == 1: # Reshape 1D data to 2D
        data = data.reshape(1, -1)
    elif data.ndim ==3 and data.shape[0] == 1: # Reshape 3D data to 2D
        data = data[0]

    if data.ndim != 2:
        raise ValueError("Data shape not supported for heatmap")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    if additional_vector is not None:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [5, 1]})
        ax = axes[0]  # The first axis for the heatmap
        ax_vector = axes[1]  # The second axis for the additional vector
        vmin = min(vmin, np.min(additional_vector)) if vmin is not None else np.min(additional_vector)
        vmax = max(vmax, np.max(additional_vector)) if vmax is not None else np.max(additional_vector)
    # Plot the heatmap
    sns.heatmap(data.T, cmap='coolwarm', ax=ax, xticklabels=col_labels,
                yticklabels=row_labels, annot=False,  vmin=vmin, vmax=vmax, linewidths=0.1, linecolor='black')
    ymin, ymax = ax.get_ylim()
    if additional_vector is not None:
        # Reshape the vector to match the heatmap format (1 row, n columns)
        additional_vector = np.reshape(additional_vector, (1, -1))
        sns.heatmap(additional_vector, cmap='coolwarm', ax=ax_vector, cbar=True, annot=False, 
                    vmax=vmax, vmin=vmin,xticklabels=[], yticklabels=[])
        if additional_vector_x is not None:
            ax_vector.set_xticks(np.arange(len(additional_vector_x)))
            ax_vector.set_xticklabels(additional_vector_x, rotation=90)
        ax_vector.set_ylabel(additional_vector_y if additional_vector_y else "")
        
    if sgrna_ots is not None:
        positions, labels = render_sg_ot_to_positions(sgrna_ots[0],sgrna_ots[1])
        labels[0] = "sgRNA:\nOT: " + labels[0]
        for i, label in enumerate(labels):
            ax.text(i+0.5, ymax + 0.02, label, ha='center', va='bottom', fontsize=10)
    ax.set_xlabel(x_label if x_label else "")
    ax.set_ylabel(y_label if y_label else "")
    # ax.set_title(title, fontfamily="monospace", fontsize=12)
    if cbar:
        if additional_vector is not None:
            colorbar = ax_vector.collections[0].colorbar
        colorbar = ax.collections[0].colorbar
        colorbar.set_label(cbar)


    # Save the figure if output_path is provided
    if output_path and ax is None:  # Only save if no subplot (otherwise, user should save the full figure)
        plt.savefig(output_path, dpi=300)
        plt.close()

  




def box_plot(data, ax, x_label, y_label, title, output_path,  showmeans=True,
             x_in_data=None, y_in_data=None, colormap=None, showfliers=True, order_by=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    meanprops = mean_order = None
    if showmeans:
        meanprops = {"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
        mean_order = data.mean().sort_values(ascending=False).index
    if order_by =="median":
        mean_order = data.median().sort_values(ascending=False).index
        medians = data.median()
    elif order_by == "mean":
        mean_order = data.mean().sort_values(ascending=False).index
    # No need to create a new figure when using ax
    ax.set_title(title)
    sns.boxplot(data=data, x=x_in_data, y=y_in_data, order=mean_order,
                showmeans=showmeans, meanprops=meanprops, boxprops={"facecolor": "lightblue"},
                  ax=ax,palette=colormap, showfliers=showfliers)
    for i, category in enumerate(mean_order):
        median_val = medians[category]
        ax.text(i, median_val, f'{median_val:.2e}', ha='center', va='bottom', 
                fontsize=10, color='black', fontweight='bold',rotation = 90)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    if x_label:
        ax.set_xlabel(x_label, fontsize=16)
    if y_label:
        ax.set_ylabel(y_label, fontsize=16)
    if output_path and ax is None:  # Only save if no subplot (otherwise, user should save the full figure)
        plt.savefig(output_path, dpi=300)
        plt.close()

def plot_all_guides_pertubration(data, output_path,mismatch_numbers=None):
    feature_color_map = {
    
    "H3K9me3": "lavender",
    "H3K27me3": "sandybrown",
    "H3K36me3": "peachpuff",
    
    "H3K4me1": "lightcoral",
   
    "H3K9ac": "mediumpurple",
    "ATAC-seq": "thistle",
    "H3K27ac": "tan",
    "H3K4me3": "linen"
    
}
    guides = list(data.keys())
    order = ['H3K4me1','H3K27me3','H3K36me3','H3K4me3','ATAC-seq','H3K9ac','H3K9me3','H3K27ac']
    # Collect all mismatch numbers across guides/features
    
    
    nrows, ncols = len(guides), len(mismatch_numbers)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5*ncols, 3.5*nrows), sharey=True,sharex="col")
    
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)
    
    
    for i, guide in enumerate(guides):
        for j, mismatch in enumerate(mismatch_numbers):
            ax = axes[i, j]
            
            # Collect values into tidy format for seaborn
            records = []
            guide_data = data[guide]
            mismatch_data = guide_data[mismatch] 
            
            
            
            df = pd.DataFrame(mismatch_data)
            
            # Compute medians and order
            mean_order = df.median().sort_values(ascending=False).index
            
            #order = mean_order
            medians = df.median()
            
            # Draw seaborn boxplot
            sns.boxplot(
                data=df, order=order,
                 
                boxprops={"edgecolor": "black"},
                palette=feature_color_map,
                ax=ax,  showfliers=False
            )
            ax.axhline(0, linestyle="--", linewidth=1, color="k", alpha=0.8, zorder=0)
            # Annotate medians
            # for k, category in enumerate(order):
            #     median_val = medians[category]
            #     ax.text(
            #         k, median_val, f"{median_val:.2e}",
            #         ha="center", va="bottom", fontsize=11,
            #         color="black", fontweight="bold", rotation=90
            #     )
            
            # Titles & labels
            if i == 0:
                if j==0:
                    ax.set_title(f"{mismatch} mismatch",fontsize=18)
                else:
                    ax.set_title(f"{mismatch} mismatches",fontsize=18)

            if j == 0:
                ax.set_ylabel(guide,fontsize=13)
                ax.tick_params(axis="y", labelsize=12)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right",fontsize=14)
            ax.set_xticklabels(order, rotation=20, ha="right",fontsize=14)
    #fig.supylabel(f"{chr(916)} Prediction", fontsize=18)
    fig.supylabel(f"Difference in predicted off-target cleavage probability", fontsize=18)
    fig.supxlabel("Epigenetic features", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "All_guides_pertubration.png"), dpi=300)
    plt.close() 
def get_rows_cols(num_plots):
        """
        Returns a rows and cols number by trying to fill sqroot of num_plots.
        """
        rows = int(np.sqrt(num_plots))
        cols = int(np.ceil(num_plots / rows))
        return (rows, cols)
def plot_subplots(data, plot_types, titles,  additional_data=None,x_label=None, y_label=None,
                   x_ticks=None, y_ticks=None, output_path=None, general_title=None,
                   sgrna_otss =None,**kwargs):
    """
    Plots multiple subplots based on the provided data and plot types.

    Parameters:
        data (list,3D np.array, dict): 
            1. (list): of data arrays for each subplot.
            2. (3D np.array): 1d- amount of plots, 2-3d data for each plot.
            3. (Dict): keys -> plots and titles, values -> data for each plot.
        plot_types (list,str): List of plot types (e.g., 'line', 'scatter') for each subplot. 
            If 1 str is given all the subplots are from that type.
        titles (list): List of titles for each subplot.
        x_label (str, optional): Label for the x-axis.
        y_label (str, optional): Label for the y-axis.
        x_ticks (list, optional): List of x-tick values.
        y_ticks (list, optional): List of y-tick values.
        output_path (str, optional): If provided, saves the plot to this path.
        generall_title (str, optional): A string representing the general title for the plot.
        **kwargs: Additional keyword arguments for the sub plotting function."""
    if isinstance(data, list):
        num_plots = len(data)
    elif isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError("Data shape not supported for subplots")
        num_plots = data.shape[0]
        data = [data[i] for i in range(num_plots)]
    elif isinstance(data, dict):
        titles = list(data.keys())
        num_plots = len(titles)
        data = [data[key] for key in data.keys()]
    
    rows,cols = get_rows_cols(num_plots)
    fig, axes = plt.subplots(nrows=rows,ncols=cols,  figsize=(cols * 5, rows * 4))
    if num_plots >1:
        if axes.ndim > 1:
            axes = axes.flatten()
    if num_plots == 1:
        axes = [axes]
    if isinstance(plot_types, str):
        plot_types = [plot_types for i in range(num_plots)]
        general_title = general_title + " " + plot_types[0]
    if titles is None:
        titles = [f"Plot {i + 1}" for i in range(num_plots)]
    if sgrna_otss is None:
        sgrna_otss = [None for i in range(num_plots)]

    for ax_index,(plots_tuple) in enumerate(zip(axes, plot_types, titles, data,sgrna_otss)):
        ax, plot_type, title, data_,sgrna_ots = plots_tuple
        if plot_type == "heatmap":
            plot_heatmap(data_, ax=ax, row_labels=y_ticks, col_labels=x_ticks,
                          x_label=x_label, y_label=y_label, title=title,sgrna_ots=sgrna_ots, **kwargs)
        elif plot_type == "boxplot":
            box_plot(data_, x_label=x_label, y_label=y_label, title=title, ax=ax,output_path=output_path, **kwargs)
        elif plot_type == 'correlation':
            if type(data_).__name__ == 'PearsonRResult':
                plot_correlation(data_._x, data_._y, x_label, y_label, data_.statistic, data_.pvalue, title, output_path, ax=ax)
            else:
                plot_correlation(data_[0], data_[1], x_label, y_label, data_[2], data_[3], title, output_path, ax=ax)
        elif plot_type == 'last_tp':
            plot_last_tp(data_[0],data_[1],data_[2],information=sgrna_ots,ax=ax,ax_title=title,**kwargs)
        elif plot_type == 'roc':
            plot_roc(data_[0],data_[1],data_[2],ax=ax,ax_title=title,output_path=None,general_title=None,**kwargs)
        elif plot_type == 'pr':
            plot_pr(data_[0],data_[1],data_[2],ax=ax,ax_title=title,output_path=None,general_title=None,**kwargs)
        elif plot_type =='bigwig':
            plot_bigwig_enrichment_per_coords(data_,ax=ax,ax_title=title,**kwargs)
    for j in range(ax_index + 1, len(axes)): # Shut down unused axes
        axes[j].axis('off')
    plt.tight_layout()
    if output_path:
        output_path = os.path.join(output_path, general_title + ".png")
        plt.savefig(output_path,dpi=300)
    plt.close()
    

def plot_binary_feature_heatmap(df, axes=None, title=None, plots_path = None):
    
    enrichment_ratio = df.loc[['positive_enrichment','negative_enrichment']].copy()
    geo_fold_df = df.loc[['geo_fold_pos', 'geo_fold_negative']].copy()

    enrichment_ratio.index = ['Positive enrichment', 'Negative enrichment']
    geo_fold_df.index = ['Positive geo_fold', 'Negative geo_fold']

    columns = df.columns
    annotations = []
    for col in columns:
        p = df.loc['p_val', col]
        pos = int(df.loc['positive_peaks', col])
        neg = int(df.loc['negative_peaks', col])
        ann = f"p={p:.2e}\n+{pos}\n-{neg}"
        annotations.append(ann)
    one_pic = False
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2]})
        one_pic = True
    # First heatmap: enrichment
    sns.heatmap(
        enrichment_ratio,
        ax=axes[0],
        cmap='coolwarm',
        annot=True,
        fmt=".3f",
        cbar=True,
        xticklabels=annotations
    )
    axes[0].set_ylabel("")
    axes[0].set_title("Positive & Negative Enrichment")

    # Second heatmap: geo_fold
    sns.heatmap(
        geo_fold_df,
        ax=axes[1],
        cmap='vlag',
        annot=True,
        fmt=".3f",
        cbar=True
    )
    axes[1].set_ylabel("")
    axes[1].set_title("Geometric Fold Change (Positive and Negative)")
    if title:
        fig.suptitle(title, fontsize=14)
    
    if one_pic:
        fig.tight_layout()
        if plots_path:
            fig.savefig(os.path.join(plots_path, title + ".png"), dpi=300)
        else:
            fig.savefig(title + ".png", dpi=300)
        plt.close()





def plot_bigwig_enrichment_per_coords(bigwig_values_dict= None, window_size = 20000, ax=None, ax_title=None):
    
    #  set x coords for -10kb, center, +10 kb
    if bigwig_values_dict is None:
        raise ValueError("bigwig_values_dict is None")
    one_pic = False
    if ax is None:
        one_pic = True
        fig, ax = plt.subplots(figsize=(10, 6))
    left_lim = (int(-1*(window_size/2)))
    right_lim = (int(window_size/2))
    x_positions = np.linspace(left_lim, right_lim, window_size)
    y_values_example = next(iter(bigwig_values_dict.values()))  # Get y-values from the first entry
    if len(y_values_example) != len(x_positions):
        raise ValueError("Length of y-values does not match length of x_positions")
    colors = ['blue', 'green', 'red']
    for i,(data_type, big_wig_values) in enumerate(bigwig_values_dict.items()):
        ax.plot(x_positions, big_wig_values, label=data_type, color=colors[i % len(colors)])
    # Set x-axis limits
    ax.set_xlim(left_lim, right_lim)
    # Add vertical line in ther center
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Center')
    left_label = f"{left_lim / 10**3:.0f}kb" if -1*(left_lim/ 10**3) >=1 else f"{left_lim:.0f}bp"
    right_label = f"{right_lim / 10**3:.0f}kb" if right_lim/ 10**3 >=1 else f"{right_lim:.0f}bp"
    ax.axvline(x=left_lim, color='red', linestyle='-', linewidth=1, label=left_label)
    ax.axvline(x=right_lim, color='red', linestyle='-', linewidth=1, label=right_label)
    ax.legend()
    ax.set_xlabel('base pairs')
    ax.set_ylabel('average values')
    ax.set_title(f'{ax_title}')
    ax.grid(True)
    if one_pic:
        # Save
        plt.savefig(f'bigwig_enrichment_{ax_title}.png')
        plt.close()
   
'''function to draw some profiles of bw data for positive lables and negative labels'''
def draw_pos_neg_bw_profiles(pos_data_points, neg_data_points, epigenetic_name,window_size):
    # Find the maximum value in all datasets (positive and negative)
    max_value = max(np.max(np.concatenate(pos_data_points)), np.max(np.concatenate(neg_data_points))) + 0.2
    # Determine the number of sets in positive and negative data
    x_coords = np.arange(start=0,stop=window_size,step=1)

    # Create subplots based on the number of sets
    # Determine the total number of data point sets
    total_sets = len(pos_data_points)

    # Create subplots with a layout determined by the total number of data point sets
    fig, axs = plt.subplots(nrows=total_sets, ncols=2, figsize=(12, 4 * 8))

    # Plot positive data sets in the first column
    for i in range(total_sets):
        axs[i, 0].plot(x_coords, pos_data_points[i], label=f'Positive Set {i + 1}', color='blue')
        axs[i, 0].set_title(f'Positive Set {i + 1} Profile')
        axs[i, 0].set_xlabel('BP')
        axs[i, 0].set_ylabel('Values')
        axs[i, 0].legend()
        axs[i, 0].set_ylim([0, max_value])  # Set y-axis limits

    # Plot negative data sets in the second column
    for j in range(total_sets):
        axs[j, 1].plot(x_coords, neg_data_points[j], label=f'Negative Set {j + 1}', color='red')
        axs[j, 1].set_title(f'Negative Set {j + 1} Profile')
        axs[j, 1].set_xlabel('BP')
        axs[j, 1].set_ylabel('Values')
        axs[j, 1].legend()
        axs[j, 1].set_ylim([0, max_value])  # Set y-axis limits

    # Add a common title for the entire figure
    fig.suptitle(f'Epigenetic Profiles - {epigenetic_name}')

# Adjust layout to prevent overlap
    plt.tight_layout()
    # Add any other details or customization as needed
    # For example, saving the figure or showing it
    plt.savefig(f'{epigenetic_name}_{window_size}_profiles.jpg')
def extract_data_points_bw(data, epigenetic_file, chrom_column, label_column, center_value_column, data_amount,window_size):
    pos_data_sampling = data[data[label_column]==1].sample(data_amount)
    neg_data_sampling = data[data[label_column]==0].sample(data_amount)
    print(f'pos:\n{pos_data_sampling[label_column]}\nneg:\n{neg_data_sampling[label_column]}')
    pos_coords = []
    neg_coords = []
    for center_loc,chrom in zip(pos_data_sampling[center_value_column], pos_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bw(epigenetic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        pos_coords.append(y_values)
    for center_loc,chrom in zip(neg_data_sampling[center_value_column], neg_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bw(epigenetic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        neg_coords.append(y_values)
    return (pos_coords,neg_coords)
def extract_data_points_bed(data, epigenetic_file, chrom_column, label_column, center_value_column, data_amount,window_size):
    pos_data_sampling = data[data[label_column]==1].sample(data_amount)
    neg_data_sampling = data[data[label_column]==0].sample(data_amount)
    print(f'pos:\n{pos_data_sampling[label_column]}\nneg:\n{neg_data_sampling[label_column]}')
    pos_coords = []
    neg_coords = []
    for center_loc,chrom in zip(pos_data_sampling[center_value_column], pos_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bed(epigenetic_bed_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        pos_coords.append(y_values)
    for center_loc,chrom in zip(neg_data_sampling[center_value_column], neg_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bed(epigenetic_bed_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        neg_coords.append(y_values)
    return (pos_coords,neg_coords)
def run_pos_neg_profiles(data,file_manager):
    data = pd.read_csv(data)
    window_size = 10000
    #for epi_name,epi_file in file_manager.get_bigwig_files():
       # pos_coords,neg_coords = extract_data_points_bw(data=data,epigenetic_file=epi_file,chrom_column="chrom",label_column="Label",center_value_column="chromStart",data_amount=10,window_size=window_size)
        #draw_pos_neg_bw_profiles(pos_coords, neg_coords,epigenetic_name=epi_name,window_size=window_size)
    for epi_name,epi_file in file_manager.get_bed_files():
        pos_coords,neg_coords = extract_data_points_bed(data=data,epigenetic_file=epi_file,chrom_column="chrom",label_column="Label",center_value_column="chromStart",data_amount=10,window_size=window_size)
        draw_pos_neg_bw_profiles(pos_coords, neg_coords,epigenetic_name=epi_name,window_size=window_size)



def draw_histogram_bigwig(file_manager):
    epigenetics_object = file_manager.get_bigwig_files()
    
    fig, axs = plt.subplots(nrows=len(epigenetics_object), ncols=1, figsize=(8, 4 * len(epigenetics_object)))

    for i, (epigenetic_mark, epigenetic_file) in enumerate(epigenetics_object):
        chr_len = epigenetic_file.chroms("chr7")
        big_wig_values = epigenetic_file.values("chr7",0,chr_len)
        big_wig_values = np.array(big_wig_values)
        big_wig_values[np.isnan(big_wig_values)] = 0.0
        counts, bins = np.histogram(big_wig_values)
        # Plot the histogram in the corresponding subplot
        axs[i].stairs(counts, bins)
        # Customize the subplot if needed (e.g., labels, title, etc.)
        axs[i].set_xticks(bins)
        axs[i].set_xticklabels([f'{bin_val:.2f}' for bin_val in bins])

        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Count')
        axs[i].set_title(f'Histogram for {epigenetic_mark}')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the entire figure to a file
    plt.savefig('epigenetics_histograms.png')



def sub_plot_shap_bar_plot(shap_explanations, guide_rnas, output_path, linkage = None, suffix = None):
    
    if len(shap_explanations)!= len(guide_rnas):
        raise RuntimeError("number of guides not equal to number of shap objects")
    num_plots = len(shap_explanations)
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()
    for i in range(num_plots):
        plt.sca(axes[i])
        shap.plots.bar(shap_explanations[i], show=False, clustering=linkage,clustering_cutoff=2)
        
        for text in axes[i].texts:
            if text.get_text():
                text.set_visible(False)
        for tick in axes[i].get_yticklabels():
            tick.set_fontsize(8)
        axes[i].set_title(f"{guide_rnas[i]}", fontsize=8)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    fig_name = f'shap_bar_{suffix}.pdf' if suffix else 'shap_bar.pdf'
    output_path = os.path.join(output_path,fig_name)
    plt.savefig(output_path, format="pdf")
    
    

def sub_plot_shap_beeswarn(shap_explanations, guide_rnas, output_path, suffix = None):
    """
    Plot beeswarn explanations togther for given guide rnas and their shap explanations objects.
    """
    if len(shap_explanations)!= len(guide_rnas):
        raise RuntimeError("number of guides not equal to number of shap objects")

    num_plots = len(shap_explanations)
    cols = 3
    rows = int(np.ceil(num_plots/cols))
    fig, axes = plt.subplots(rows, cols,sharex=True ,figsize=(cols * 6, rows * 5))

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for i, shap_exp in enumerate(shap_explanations):
        plt.sca(axes[i])  # set current axis
        shap.plots.beeswarm(shap_exp, show=False,color_bar_label="",color_bar=False)
        axes[i].set_title(f"{guide_rnas[i]}", fontsize=8)  # or any smaller size
        for tick in axes[i].get_yticklabels():
            tick.set_fontsize(8)
        #axes[i].set_xticklabels([])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    fig.supxlabel("SHAP value", fontsize=12)
    fig.supylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    red_circle = mlines.Line2D([0], [0], marker='o', color='red', label='1', markersize=5, markerfacecolor='red', markeredgewidth=0)
    blue_circle = mlines.Line2D([0], [0], marker='o', color='blue', label='0', markersize=5, markerfacecolor='blue', markeredgewidth=0)

    # Add the legend with circular markers
    fig.legend(
        handles=[red_circle, blue_circle],
        loc='lower center',
        bbox_to_anchor=(0.35, -0.02),
        ncol=2,
        title="Feature value",
        frameon=False
    )

    fig_name = f'shap_beeswarn_{suffix}.pdf' if suffix else 'shap_beeswarn.pdf'
    output_path = os.path.join(output_path,fig_name)
    plt.savefig(output_path, format="pdf")

'''Draw a bar plot. y- metric\premonace, x - num of models in the ensemble'''
def plot_ensemeble_preformance(y_values, x_values, title, y_label,x_label,stds,output_path,if_scaling = True, if_ticks = False):
    '''This is a scatter plot function that plots '''
    plt.clf()
    # clear underscores from the x_values
    x_positions = np.arange(len(x_values))
    plt.scatter(x_positions, y_values)
    plt.errorbar(x_positions, y_values, yerr=stds, fmt='none', capsize=5, elinewidth=2, markeredgewidth=2, color='blue')
    
    plt.title(title)
    if if_scaling:
        x_values = [int(x/100) for x in x_values]
        x_label = x_label + " (× 10²)"
    if if_ticks:
        plt.xticks(ticks=x_positions,labels=x_values, fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    output_path = output_path + f"/{title}.png"
    plt.savefig(output_path)


def plot_ensemble_performance_on_ax(ax, y_values, x_values, stds,
                              if_scaling=False, if_ticks=False,small_ticks=False):
    # clear underscores from the x_values
    x_positions = np.arange(len(x_values))
    ax.scatter(x_positions, y_values, color="blue")
    ax.errorbar(
        x_positions, y_values, yerr=stds,
        fmt='none', capsize=5, elinewidth=2,
        markeredgewidth=2, color='blue'
    )
    #ax.set_title(title, fontsize=14)
    if if_scaling:
        x_values = [int(x / 100) for x in x_values]
    if if_ticks:
        ax.set_xticks(x_positions)
        if small_ticks:
            ax.set_xticklabels(x_values, fontsize=14, rotation=45, ha='right')
        else:
            ax.set_xticklabels(x_values, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)


def plot_ensemble_performance_mean_std(mean_values, std_values, x_values,p_values, 
                                       title, y_label, path,partition_information= None ,asecnding = False, fmt='.3f', 
                                       only_seq = 'Only-seq'):
    """
    Plots a horizontal bar plot with mean and standard deviation values.
    """
    plt.clf()
    # Sort indices based on mean values
    sorted_indices = np.argsort(mean_values)
    if asecnding:
        sorted_indices = sorted_indices[::-1]
    mean_values_sorted = [mean_values[i] for i in sorted_indices]
    x_values_sorted = [x_values[i] for i in sorted_indices]
    std_sorted = [std_values[i] for i in sorted_indices]
    # get amount of models and set widgth of bars
    num_models = len(mean_values_sorted)
    ind = np.arange(num_models)  # the y locations for the groups
    width = 0.8  # the width of the bars
    # Get the longest label to determine the figure size
    x_singel_labels = [x for x in x_values_sorted if "_" not in x]
    longest_label = max(x_singel_labels, key=len) # Get longest label without considering the subset labels
    label_width = len(longest_label) * 0.2  # Adjust the multiplier as needed for proper spacing

    # Set the figure size based on the width required for the longest label
    fig_width = 8 + label_width  # Adjust the initial figure width as needed
    fig_height = 6  # Adjust the initial figure height as needed

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # create plt
    fig.tight_layout(pad=5)
    bars = ax.barh(ind, mean_values_sorted, width, xerr=std_sorted)
    multi = False
    std_gap = max(std_values) if max(std_values) > 0 else 0.02
    min_x = min(mean_values_sorted) - 3 * std_gap if min(mean_values_sorted) > 0 else 0
    max_x = max(mean_values_sorted) + 2 * std_gap
    # Add p-value annotations
    if len(p_values) > 0: # not empty
        for i, bar in enumerate(bars):
            model = x_values_sorted[i]
            if model == only_seq:
                plt.text((bar.get_width() - 2*std_sorted[i])  , bar.get_y() + (width/2), f'{mean_values_sorted[i]:{fmt}}', va='center', fontsize=10, color='white')
                continue
            else :
                p_val = p_values[model]
                annotation = p_val_annotation(p_val)
            plt.text(bar.get_width() + std_sorted[i] + 0.0005 , bar.get_y() + (width/2), annotation, va='center', fontsize=8)
            plt.text((bar.get_width() - 2*std_sorted[i]) , bar.get_y() + (width/2), f'{mean_values_sorted[i]:{fmt}}', va='center', fontsize=10,color='white')

            if "epigenetics" in model:
                multi = True
                bar.set_color('red')
            elif "All" in model:
                bar.set_color('green')
          
       

    ax.set_ylabel('Features', fontsize=12)
    ax.set_xlabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(ind)
    
    ### THIS NOT NEEDED??????
    # Initialize variables for subset labels
    subset_count = 1
    y_labels = []
    subset_mapping = {}
    # Iterate through each model in x_values_sorted
    for model in x_values_sorted:
        if "_" in model:
            subset_label = f'Subset {subset_count}'
            y_labels.append(subset_label)
            subset_mapping[subset_label] = model.split("_")
            subset_count += 1
        else:
            y_labels.append(model)
    ###############################################
    ax.set_yticklabels( y_labels,fontsize = 12)
    
    fig.subplots_adjust(left=label_width/fig_width)
    if multi:
        ax.plot([], label='Epigenetic subsets', color='red')
        for subset_label, subset_models in subset_mapping.items():
            ax.plot([], label=f'{subset_label}: {", ".join(subset_models)}', color='none')
    if partition_information:
        for key,info_ in partition_information.items():
            ax.plot([], label=f'{key}: {info_}', color='none')
    ax.legend(loc='lower right',bbox_to_anchor=(1.05, 0.0),fontsize = 'small',borderaxespad=0.05,ncol=1)
    
    ax.set_xlim(min_x, max_x)
    # Remove right and upper spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    path = path + f"/{title}.png"
    plt.savefig(path,dpi=300)
    plt.close()

