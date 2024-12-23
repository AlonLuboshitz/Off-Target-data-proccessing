import argparse
import os
import json
def features_method_dict():
    '''A dictionary for the feature incorporate in the model.'''
    return {
            1: "Only_sequence",
            2: "With_features_by_columns",
            3: "Base_pair_epigenetics_in_Sequence",
            4: "Spatial_epigenetics"
            
        }

def cross_val_dict():
    '''A dictionary for the cross validation methods.'''
    return {
            1: "Leave_one_out",
            2: "K_cross",
            3: "Ensemble"
        }

def model_dict():
    '''A dictionary for the models to use.'''
    return {
            1: "LOGREG",
            2: "XGBOOST",
            3: "XGBOOST_CW",
            4: "CNN",
            5: "RNN",
            6: "GRU-EMB"
        }
def encoding_dict():
    '''
    A dictionary for the sequence encoding types.
    '''
    return {
        1: "PiCrispr_encoding",
        2: "Full_encoding"
    }

def off_target_constrians_dict():
    '''
    A dictionary for the off-target constraints.
    '''
    return {
        1: "No_constraints",
        2: "Mismatch_only",
        3: "Bulges_only"
    }
def class_weights_dict():
    '''
    A dictionary for the class weights.
    '''
    return {
        1: "CW",
        2: "No_CW"
    }
def early_stoping_dict():
    '''
    A dictionary for the early stoping.
    '''
    return {
        1: "Early_stop",
        2: "No_early_stop"
    }
def main_argparser():
    parser = argparse.ArgumentParser(description='''Python script to init a model and train it on off-target dataset.
                                     Different models, feature types, cross_validations and tasks can be created.
                                     ''')
    parser.add_argument('--model','-m', type=int, 
                        help='''Model number: 1 - LogReg, 2 - XGBoost,
                          3 - XGBoost with class weights, 4 - CNN, 5 - RNN, 6- GRU-EMB''',
                         required=True, default=4)
    parser.add_argument('--cross_val','-cv', type=int,
                         help='''Cross validation type: 1 - Leave one out, 
                         2 - K cross validation, 3 - Ensmbel, 4 - K cross with ensemble''',
                         required=True, default=1)
    parser.add_argument('--features_method','-fm', type=int,
                         help='''Features method: 1 - Only_sequence, 2 - With_features_by_columns, 
                         3 - Base_pair_epigenetics_in_Sequence, 4 - Spatial_epigenetics''', 
                        required=True, default = 1)
    parser.add_argument('--features_columns', '-fc', type=str,
                     help='Features columns - path to a dict with keys as feature type and values are the columns names', required=False)
    parser.add_argument('--epigenetic_window_size','-ew', type=int, 
                        help='Epigenetic window size - 100,200,500,2000', required=False, default=2000)
    parser.add_argument('--epigenetic_bigwig','-eb', type=str,
                         help='Path for epigenetic folder with bigwig files for each mark.', required=False)
    parser.add_argument('--task','-t', type=str, help='Task: Classification/Regression/T_Regression - T = Transformed', required=True, default='Classification')
    parser.add_argument('--transformation','-tr', type=str, help='Transformation type: Log/MinMax/Z_score', required=False, default=None)
    parser.add_argument('--job','-j', type=str, help='Job type: Train/Test/Evaluation/Process', required=True)
    parser.add_argument('--over_sampling','-os', type=str, help='Over sampling: y/n', required=False)
    parser.add_argument('--seed','-s', type=int, help='Seed for reproducibility', required=False)
    parser.add_argument('--data_reproducibility','-dr', type=str, help='Data reproducibility: y/n', required=False, default='n')
    parser.add_argument('--model_reproducibility','-mr', type=str, help='Model reproducibility: y/n', required=False, default='n')
    parser.add_argument('--config_file','-cfg', type=str, 
                        help='''Path to a json config file with the next dictionaries:
                        1. Data columns:
                        target_column, offtarget_column, chrom_column, start_column, end_column, binary_label_column,regression_label_column
                        2. Data paths:
                        Train_guides, Test_guides, Vivo-silico, Vivo-vitro, Model_path, ML_results, Data_name
                        ''',
                         required=True)
    parser.add_argument('--data_name','-jd', type=str,
                         help='''Dictionary names: 1 - Change_seq, 2 - Hendel, 3 - Hendel_Changeseq
                        The name of the data dict need to parse from the json file''', required=False)
    parser.add_argument('--data_type','-dt', type=str, help='''Data type: silico/vitro''', required=True)
    parser.add_argument('--partition','-p', type=int, nargs='+',help='Partition number given via list', required=False)
    parser.add_argument('--n_models','-nm', type=int, help='Number of models in each ensmbel', required=False)
    parser.add_argument('--n_ensmbels','-ne', type=int, help='Number of ensmbels', required=False)
    parser.add_argument('--encoding_type','-et', type=int, help='Sequence encoding type: 1 - PiCrispr, 2 - Full', default=1)
    parser.add_argument('--off_target_constriants','-otc', type=int, help='Off-target constraints: 1 - No_constraints, 2 - Mismatch_only, 3 - Bulges_only', default=1)
    parser.add_argument('--class_weights','-cw', type=int, help='Class weights: 1 - CW, 2 - No_CW', default=1)
    parser.add_argument('--deep_params','-dp', nargs='+',type=int, help='Deep learning parameters - epochs, batch', default=None)
    parser.add_argument('--early_stoping','-es', nargs='+',type=int, help='''Early stoping[0]: 1 - Early_stop, 2 - No_early_stop
                        Early stoping[1]: paitence size''', default=None)
    parser.add_argument('--guides_constraints','-gc', type=str,nargs='+', help='(guides_description , path to guides to exclude from the data, target_column)', default=None)
    return parser

def parse_args(argv,parser):
    
    if '--argfile' in argv:
        argfile_index = argv.index('--argfile') + 1
        argfile_path = argv[argfile_index]
     # Read the arguments from the file
        with open(argfile_path, 'r') as f:
            file_args = f.read().split()
        
        # Parse args with the file arguments included
            args = parser.parse_args(file_args)
    else:
    # Parse normally if no config file is provided
        args = parser.parse_args()    
    # Read the JSON file and load it as a dictionary
    return args


 
def validate_main_args(args):
    if not os.path.exists(args.config_file):
        raise ValueError("Data columns config file does not exist") 
    if args.data_type not in ['vivo-silico','vivo-vitro','vitro-silico']:
        raise ValueError("Data type must be either vivo-silico/vivo-vitro/vitro-silico")
    if args.task.lower() not in ['classification','regression','t_regression','reg_classification']:
        raise ValueError("Task must be either Classification, Regression or T_Regression")
    if args.task.lower() == 't_regression' and (args.transformation is None or args.transformation.lower() not in ["log","minmax","z_score"]):
        raise ValueError("Transformation must be given for transformed regression or must be either Log, MinMax or Z_score")
    if args.job.lower() not in ['train','test','evaluation','process']:
        raise ValueError("Job must be either Train\Test\Evaluation")
    if args.deep_params is not None and len(args.deep_params) != 2:
        raise ValueError("Deep learning parameters must be given as a list of 2 integers - epochs and batch size")
    if args.early_stoping is not None and len(args.early_stoping) != 2:
        raise ValueError("Early stoping parameters must be given as a list of 2 integers - early stoping and patience")
    args.guides_constraints =  validate_exclude_guides(args.guides_constraints)
    ## Print all args:
    print("Arguments are:")
    for arg, value in vars(args).items():
        print(f'{arg}: {value}') 
    with open(args.config_file, 'r') as f:
        print("Parsing config file")
        configs = json.load(f)
        data_columns = configs["Columns_dict"]
        data_configs = configs[args.data_name]
        return args, data_configs, data_columns
    

def validate_exclude_guides(exclude_guides = None):
    '''
    Validate the exclude_guides- Tuple of (guides_description , path to guides to exclude from the data, target_column)
    '''
    if exclude_guides is not None:
        if len(exclude_guides) != 3:
            raise ValueError("Exclude guides must be a tuple of 3 elements")
        if not os.path.exists(exclude_guides[1]):
            raise ValueError("Path to exclude guides does not exist")
        return exclude_guides
    return None
        