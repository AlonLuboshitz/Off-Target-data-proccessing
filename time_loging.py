import time
import os  
from parsing import model_dict, cross_val_dict
global TIME_LIST, KEEP_TIME, TIME_LOGS_PATHS
TIME_LIST = []
def set_time_log(keep_time = False, time_logs_paths = None):
    global KEEP_TIME, TIME_LOGS_PATHS
    if keep_time:
        KEEP_TIME = keep_time
    if os.path.exists(time_logs_paths):
        TIME_LOGS_PATHS = time_logs_paths
    else:
        TIME_LOGS_PATHS = os.getcwd()
def log_time(event_name):
    '''This function log the time of the event for further evalautions.'''
    global TIME_LIST
    TIME_LIST.append((event_name, time.time()))
    pass
    

def save_log_time(args):
    global TIME_LIST, TIME_LOGS_PATHS, KEEP_TIME
    model = model_dict()[args.model]
    cross_val = cross_val_dict()[args.cross_val]
    
    if KEEP_TIME:
        with open(os.path.join(TIME_LOGS_PATHS, f"time_logs_{model}_{cross_val}_{args.data_type}_ens:{args.n_ensmbels}_N:{args.n_models}.txt"), "w") as f:
            n = len(TIME_LIST)
            for i in range(n // 2):  # Loop only through half of the list
                first_event, first_time = TIME_LIST[i]
                last_event, last_time = TIME_LIST[n - 1 - i]
                time_diff = last_time - first_time
                f.write(f"{first_event} to {last_event}: {time_diff}\n")
            
            # If the list has an odd number of events, handle the middle event
            if n % 2 == 1:
                middle_event, middle_time = TIME_LIST[n // 2]
                f.write(f"{middle_event}: N/A\n")
