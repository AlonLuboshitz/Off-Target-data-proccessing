# script to run cas-offinder
# given a folder with input file, run each one with cas-offinder app, and create 
# correspoding output file.
'''this function uses sub-proccess(sp) for each input.
after sp is done continue to the next input file, etc.'''
import sys
import subprocess
import re
import os
def casoffinder_ruuner(casoffinder_path,fileinputs_path):
    #valid cas-offinder and get CPU/GPU type 
    answer = valid_offinder(casoffinder_path)
    cg_type = answer[0]
    args = answer[1]
    # create an output folder in the dir of the inputfiles.
    parent_path = os.path.dirname(fileinputs_path)
    output_folder = os.path.join(parent_path,"casoffinder_outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # after validation run the cas-offinder on each input txt file.
    #iterate on each inputfile in the folder
    print("start running cas-offinder on files, this might take a while...")
    i=1
    for filename in os.listdir(fileinputs_path):
        
        if filename.endswith('_input.txt'):
            print("running " + str(i) + " file: " + filename)
            input_txt = os.path.join(fileinputs_path,filename)
            
            output = filename.split('_input.txt')[0] + '_output.txt'
            output_txt = os.path.join(output_folder,output)
            
            temp_args = args + " " + input_txt + " " + cg_type + " " + output_txt
            try:
                p = subprocess.run(temp_args,capture_output=True,text=True,shell=True,check=True)
                
            except subprocess.CalledProcessError as e:
                print("CalledProcessError:", e)
                print("Running file: {} faild".format(filename))
        i += 1
    
    
'''function valids cas-offinder, if so returns tuple with C\G and args for cas-offinder.
if not prints the error and exit the script'''
def valid_offinder(casoffinder_path):
    args = casoffinder_path + '/./cas-offinder'
    try:
        p = subprocess.run(args,capture_output=True,text=True,shell=True,check=True)
        answer = p.stdout
        #get CPU or GPU
        cg_type = re.search( r'Type:\s+(.*?),', answer)
        cg_type = cg_type.group(1)[0]
        return (cg_type,args)
    except subprocess.CalledProcessError as e:
        print("CalledProcessError:", e)
        sys.exit(0)
'''' first arg is casoffidner path.
second arg is file input path.'''
if __name__ == '__main__':
    casoffinder_ruuner(sys.argv[1],sys.argv[2])
    
   