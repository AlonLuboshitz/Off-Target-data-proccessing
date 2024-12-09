import time
import os  
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
    return
    if isinstance(time_list, list):
        time_list.append((event_name, time.time()))
    else:
        raise ValueError("The time_list must be a list")
    return time_list

def save_log_time(args):
    global TIME_LIST, TIME_LOGS_PATHS, KEEP_TIME
    params = f"model: {args.model} data: {args.data_type} models: {args.n_models} ensembles: {args.n_ensmbels} task: {args.task}"
    if KEEP_TIME:
        with open(os.path.join(TIME_LOGS_PATHS, f"time_logs_{args.task}_{args.data_type}.txt"), "w") as f:
            f.write(f"Time logs for {params}\n")
            for event, time in TIME_LIST:
                f.write(f"{event}: {time}\n")
