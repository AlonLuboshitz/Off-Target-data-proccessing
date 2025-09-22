import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
from matplotlib.gridspec import GridSpec
from plotting import plot_correlation, plot_ensemble_performance_on_ax
from evaluation import get_group_dict, get_roc_pr_values, get_x_vals, plot_roc_pr_for_ensmble_by_paths
from features_and_model_utilities import transform_labels
def linear_fit(x, y):
    if x is None:
        x = np.arange(len(y))
    if y is None:
        raise RuntimeError("No y values given")
    if len(x) != len(y):
        raise RuntimeError("x and y must be the same length")
    x = np.asarray(x, float); y = np.asarray(y, float)
    fit = linregress(x, y)              # slope, intercept, rvalue, pvalue, ...
    r_from_fit = fit.rvalue             # == Pearson r in simple linear reg with intercept

    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
                 
        "R^2": r_from_fit**2,
        "pvalue": fit.pvalue
       
    }
    

def plot_hendel_changeseq_increasing_training_data():
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex="col", sharey="row")
    #axes = axes.flatten()
    cs_base_path_sg = 'Thesis_data/localdata/alon/ML_results/Change-seq/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence'
    n_models_in_ensmbel = 50
    group_dict = get_group_dict(cs_base_path_sg, n_models_in_ensmbel)
    group_dict_ = {5-key: value for key, value in group_dict.items() if key!=5} # Group by 5
    group_dict_[5] = group_dict[5] # Add group 5
    group_dict = group_dict_
    group_dict = dict(sorted(group_dict.items()))
    y_values_list,stds_list = get_roc_pr_values(group_dict)
    x_vals = range(1,len(group_dict)+1)
    linear_fits = {}
    plot_ensemble_performance_on_ax(
        axes[0,0],
        y_values_list[0],
        x_vals,
        stds_list[0],
        if_scaling=False,
        if_ticks=True
    )
    plot_ensemble_performance_on_ax(
        axes[1,0],
        y_values_list[1],
        x_vals,
        stds_list[1],
        if_scaling=False,
        if_ticks=True
    )
    linear_fits['Change-seq AUPRC'] = linear_fit(x_vals,y_values_list[0])
    linear_fits['Change-seq AUROC'] = linear_fit(x_vals,y_values_list[1])
    # Hendel sgRNA
    
    h_base_path_sg = "Thesis_data/localdata/alon/ML_results/Hendel/vivo-silico/Classification/Performance-increasing-sgRNAs"
    group_dict = get_group_dict(h_base_path_sg, 50)
    y_values_list,stds_list = get_roc_pr_values(group_dict)
    x_vals = range(1,len(group_dict)+1)
    plot_ensemble_performance_on_ax(
        axes[0,1],
        y_values_list[0],
        x_vals,
        stds_list[0],
        if_scaling=False,
        if_ticks=True,
        small_ticks=False
    )
    plot_ensemble_performance_on_ax(
        axes[1,1],
        y_values_list[1],
        x_vals,
        stds_list[1],
        if_scaling=False,
        if_ticks=True,
        small_ticks=False
    )
    linear_fits['Hendel AUPRC'] = linear_fit(x_vals,y_values_list[0])
    linear_fits['Hendel AUROC'] = linear_fit(x_vals,y_values_list[1])
    # Hendel OTSs
    
    base_path = 'Thesis_data/localdata/alon/ML_results/Hendel/vivo-silico/Classification/Performance-increasing-OTss/by_positives'
    group_dict = get_group_dict(base_path, 50)
    data = 'Data/Hendel_lab/Hendel-Partition_1.csv'
    x_vals = get_x_vals(data,False,True)
    y_values_list,stds_list = get_roc_pr_values(group_dict)
    plot_ensemble_performance_on_ax(
        axes[0,2],
        y_values_list[0],
        x_vals,
        stds_list[0],
        if_scaling=True,
        if_ticks=True,
        small_ticks=False
    )
    plot_ensemble_performance_on_ax(
        axes[1,2],
        y_values_list[1],
        x_vals,
        stds_list[1],
        if_scaling=True,
        if_ticks=True,
        small_ticks=False
    )
    linear_fits['Hendel OTSs AUPRC'] = linear_fit(x_vals,y_values_list[0])
    linear_fits['Hendel OTSs AUROC'] = linear_fit(x_vals,y_values_list[1])

    for key,value in linear_fits.items():
        print(key,value)
    axes[1, 0].set_xlabel("Number of training subsets", fontsize=18)
    axes[1,1].set_xlabel("Number of training subsets", fontsize=18)
    #axes[1, 2].set_xlabel("Number of OTSs", fontsize=18)
    axes[0,0].set_ylabel("AUPRC", fontsize=18)
    axes[1,0].set_ylabel("AUROC", fontsize=18)
    axes[0,0].set_title("CHANGE-seq", fontsize=20)
    axes[0,1].set_title("Hendel", fontsize=20)
    axes[0,2].set_title("Hendel", fontsize=20)
    axes[1,2].set_xlabel("Number of OTSs (× 10²)", fontsize=18)
    letters = ['A', 'B', 'C', 'D','E','F']
    selected_axes = axes[:, :].flatten()
    for ax,letter in zip(selected_axes,letters):
        ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    first_row_axes = axes[0, :]
    for ax in first_row_axes:
        ax.set_yticks(np.arange(0, 0.49, 0.07))  # from 0.00 to 0.45 in steps of 0.05
    plt.tight_layout()
    plt.savefig("Plots/Thesis/Performance_by_parts.png", dpi=300)


def plot_H_C_HC_models():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", sharey="row")
    titles = ["CHANGE-seq","Hendel","Hendel + CHANGE-seq"]
    scores_path = ["Thesis_data/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/test_on_hendel/6_intersect/all_6/Scores/ensemble_1.csv",
    "Thesis_data/localdata/alon/ML_results/Hendel/vivo-silico/Classification/Performance-increasing-sgRNAs/11_group/1-2-3-4-5-6-7-8-9-10-11_partition/1-2-3-4-5-6-7-8-9-10-11_partition_50/Scores/ensemble_1.csv",
    "Thesis_data/localdata/alon/ML_results/Hendel_Changeseq/vivo-silico/test_on_hendel/6_intersecting/all_6/Scores/ensemble_1.csv"]
    hendel_axes = [axes[0,0],axes[0,1]]
    plot_roc_pr_for_ensmble_by_paths(scores_path,titles,"Plots/Thesis","Test_on_hendel",
                                     legend_title='Training data',axes=hendel_axes)
    scores_path = ["Thesis_data/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/7_partition/7_partition_50/Scores/ensemble_1.csv",
                    "Thesis_data/localdata/alon/ML_results/Hendel/vivo-silico/Classification/test_on_changeseq/6_intersect/all_6/Scores/ensemble_1.csv",
                    "Thesis_data/localdata/alon/ML_results/Hendel_Changeseq/vivo-silico/test_on_changeseq/6_intersecting/all_6/Scores/ensemble_1.csv"]
    
    cs_axes = [axes[1,0],axes[1,1]]
    plot_roc_pr_for_ensmble_by_paths(scores_path,titles,"Plots/Thesis","Test_on_changeseq",
                                     legend_title='Training data',axes=cs_axes)
    axes[0,0].set_ylabel("Test on CHANGE-seq\nPrecision", fontsize=18)
    axes[1,0].set_ylabel("Test on Hendel\nPercision", fontsize=18)
    letters = ['A', 'B', 'C', 'D']
    for ax,letter in zip(axes.flatten(),letters):
        ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    plt.tight_layout()
    plt.savefig("Plots/Thesis/H_C_HC_models.png",dpi=300)


def hendel_change_corr_per_guide():
    data = pd.read_csv("Thesis_data/Hendel_vs_CHANGE-seq_regression.csv")
    grouped = data.groupby('target')
    #num_plots = len(grouped)
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    # Outer 2x2 grid
    outer = GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=[1, 0.8])
    # Top row (spans both columns) → split into 2x2 small axes
    top = outer[0, :].subgridspec(nrows=2, ncols=2, wspace=0.3, hspace=0.4)
    ax_s11 = fig.add_subplot(top[0, 0])
    ax_s12 = fig.add_subplot(top[0, 1])
    ax_s21 = fig.add_subplot(top[1, 0])
    ax_s22 = fig.add_subplot(top[1, 1])
    axes = [ax_s11, ax_s12, ax_s21, ax_s22]
    # Bottom row → two larger axes
    grouped = [(name,group) for name, group in grouped if len(group) >= 3]
    for (name, group),ax in zip(grouped,axes):
        
        hendel  = group["Hendel"].values
        change = group["CHANGE-seq"].values
        r, p = pearsonr(hendel,change)
        plot_correlation(hendel,change,None,None,r_coeff=r,p_value=p,title=name,output_path=None,ax=ax,text_size=14)
        #r_log, p_log = pearsonr(transform_labels(hendel,'log'),transform_labels(change,'log'))
        # normal_r.append(r)
        # log_r.append(r_log)
    
    ax_big1 = fig.add_subplot(outer[1, 0])
    ax_big2 = fig.add_subplot(outer[1, 1])
    hendel  = data["Hendel"].values
    change = data["CHANGE-seq"].values
    r, p = pearsonr(hendel,change)
    r_log, p_log = pearsonr(transform_labels(hendel,'log'),transform_labels(change,'log'))
    plot_correlation(hendel,change,"Hendel - read count","CHANGE-seq - read count",r_coeff=r,p_value=p,title=None,output_path=None,ax=ax_big1)
    plot_correlation(transform_labels(hendel,'log'),transform_labels(change,'log'),"Hendel - read count (log)","CHANGE-seq - read count (log)",r_coeff=r_log,p_value=p_log,title=None,output_path=None,ax=ax_big2)
    ax_top_labels = fig.add_subplot(outer[0, :], frameon=False)
    ax_top_labels.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_top_labels.set_xlabel('Hendel - read count',fontsize=16)
    ax_top_labels.set_ylabel('CHANGE-seq - read count',fontsize=16,labelpad=30)
    #ax_top_labels.set_in_layout(False)
    ax_s11.text(-0.05, 1.02, 'A', transform=ax_s11.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    for ax,letter in zip([ax_big1,ax_big2],['B','C']):
        ax.text(-0.05, 1.02, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    plt.tight_layout()
    plt.savefig("Plots/Thesis/Hendel_vs_CHANGE-seq_correlation.png",dpi=300)
    plt.close()
    

def read_coverage(path):
    data = pd.read_csv(path)
    groups = data.groupby('target').agg('Read_count').sum()
    return groups


def p_values_bar_plot():
    one_vs_two = pd.read_csv("Thesis_data/1_vs_2_table.csv")
    two_vs_three = pd.read_csv("Thesis_data/2_vs_3_table.csv")
    direction_test = {'H3K27ac': 'less','H3K9me3': 'less','H3K36me3': 'greater','H3K4me1': 'greater','H3K4me3': 'greater',
                      'ATAC-seq': 'greater', 'H3K9ac': 'greater','H3K27me3':'greater'}
    colors = {'greater': 'lightcoral', 'less': 'cornflowerblue'}
    fig, ax = plt.subplots(1, 2, figsize=(12, 6),constrained_layout=False)
    def plot_on_ax(df,ax, title):
        counts = df.iloc[6, 1:].astype(float)   # row "# of p values ≤ 0.05"
        max_pvals = df.iloc[7, 1:].astype(float) # row "Maximum p value among significant sgRNAs"
        log_max_pvals = -np.log10(max_pvals)
        marks = log_max_pvals.index
        bar_colors = [colors[direction_test[m]] for m in marks]

        bars = ax.bar(marks, log_max_pvals, color=bar_colors)

        # Annotate counts (out of 6)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{int(val)}/6', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=18)
        ax.set_xticklabels(marks, rotation=45, ha='right', fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=14)
    plot_on_ax(one_vs_two,ax[0],"2 vs. 1 mismatches")
    plot_on_ax(two_vs_three,ax[1],"3 vs. 2 mismatches")
    ax[0].set_ylabel(r'$-\log_{10}(\mathrm{max\ p\ value})$', fontsize=16)
    letters = ['A', 'B']
    for axis,letter in zip(ax,letters):
        axis.text(-0.05, 1.02, letter, transform=axis.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    fig.legend(handles=[plt.Line2D([0],[0], color='lightcoral', lw=4, label='Greater'),
                    plt.Line2D([0],[0], color='cornflowerblue', lw=4, label='Less')],
           loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=14, frameon=False)

    plt.tight_layout(rect=[0, 0.10, 1, 1]) 
    plt.savefig("Plots/Thesis/p_values_bar_plot.png",dpi=300,bbox_inches="tight")
    plt.close()

plot_hendel_changeseq_increasing_training_data()
plot_H_C_HC_models()
hendel_change_corr_per_guide()
p_values_bar_plot()
