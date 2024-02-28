import os
import pyBigWig
#import pybedtools

class File_management:
    # Positive and negative are files paths, pigenetics_bed and bigwig are folders path
    def __init__(self, positives, negatives, epigenetics_bed , bigwig, merged_data, data_folder_path) -> None: 
        self.positive_path = positives
        self.negative_path = negatives
        self.epigenetics_folder_path = epigenetics_bed
        self.bigwig_folder_path = bigwig
        self.merged_data_path = merged_data
        self.data_folder_path = data_folder_path
        self.create_bigwig_files_objects()
        self.set_global_bw_max()
        #self.create_bed_files_objects()
    ## Getters:
    def get_positive_path(self):
        return self.positive_path
    def get_negative_path(self):
        return self.negative_path
    def get_merged_data_path(self):
        return self.merged_data_path
    def get_epigenetics_folder(self):
        return self.epigenetics_folder_path
    def get_bigwig_folder(self):
        return self.bigwig_folder_path
    def get_number_of_bigiwig(self):
        return self.bigwig_amount
    def get_number_of_bed_files(self):
        return self.bed_files_amount
    def get_bigwig_files(self):
        if self.bigwig_amount > 0 :
            return self.bigwig_files.copy() # keep original list
        else: raise RuntimeError("No bigwig files")
    def get_bed_files(self):
        if self.bed_files_amount > 0 :
            return self.bed_files.copy()
        else: raise RuntimeError("No bedfiles setted")
    def get_global_max_bw(self):
        if len(self.glb_max_dict) > 0 :
            return self.glb_max_dict
        else : raise RuntimeError("No max values setted for bigwig files")
    
    ## Setters:

    def set_model_results_output_path(self, output_path):
        self.model_results_output_path = output_path
    
    def set_bigwig_files(self,bw_list):
        flag = False
        if bw_list: #bw list isnt empty
            flag = True
            # check if bw/bedgraph
            for file_name,file_object in bw_list:
                if not file_object.isBigWig():
                    # not bigwig throw error
                    flag = False
                    raise Exception(f'trying to set bigwig files with other type of file\n{file_name}, is not bw file')
                else : 
                    continue # check next file
        if flag: # not empty list + all files are big wig
           #self.close_big_wig(only_bw_object)
           self.bigwig_files = bw_list
           self.bigwig_amount = len(self.bigwig_files)
        else : # flag is false list is empty
            raise Exception('Trying to set bigwig files with empty list, try agian.') 


    # Functions to create paths from folders
    '''Create paths list from folder'''
    def create_paths(self,folder):
        paths = []
        for path in os.listdir(folder):
            paths.append(os.path.join(folder,path))
        return paths

    '''Create pyBigWig objects list of the bigwig files'''
    def create_bigwig_files_objects(self):
        self.bigwig_files = []
        for path in self.create_paths(self.bigwig_folder_path):
            name = path.split("/")[-1].split(".")[0] # retain the name of the file (includes the marker)
            try:
                name_object_tpl = (name,pyBigWig.open(path))
                self.bigwig_files.append(name_object_tpl)
            except Exception as e:
                print(e)
        self.bigwig_amount = len(self.bigwig_files) # set amount
        
    # def create_bed_files_objects(self):
    #     self.bed_files = []
    #     for path in self.create_paths(self.epigenetics_folder_path):
    #         name = path.split("/")[-1].split(".")[0] # retain the name of the file (includes the marker)
    #         try:
    #             name_object_tpl = (name,pybedtools.BedTool(path))
    #             self.bed_files.append(name_object_tpl)
    #         except Exception as e:
    #             print(e)
    #     self.bed_files_amount = len(self.bed_files) # set amount
    '''Function to close all bigwig objects'''
    def close_big_wig(self,new_bw_object_list):
        if self.bigwig_files: # not empty
            for file_name,file_object in self.bigwig_files:
                if file_object in new_bw_object_list: # setting new list with objects from old one - dont close the file
                    continue
                else:
                    try:
                        file_object.close()
                    except Exception as e:
                        print(e)
    # def close_bed_files(self):
    #     if self.bed_files:
    #         pybedtools.cleanup(remove_all=True)
    def set_global_bw_max(self):
        self.glb_max_dict = {}
        for bw_name,bw_file in self.bigwig_files:
            # get chroms
            chroms = bw_file.chroms()
            max_list = []
            for chrom,length in chroms.items():
                # get max
                max_val = bw_file.stats(chrom,0,length,type='max')[0]
                max_list.append(max_val)
            self.glb_max_dict[bw_name] = max(max_list)

    '''Function to save machine learning results to file
    '''
    def save_ml_results(self, results_table, model_name): 
        # concatenate self.model_results_output_path with model_name
        output_path = os.path.join(self.model_results_output_path,f'{model_name}.csv')
        # save results to file
        results_table.to_csv(output_path)
        
    '''dtor'''
    def __del__(self):
        self.close_big_wig([])
        #self.close_bed_files()
        # call more closing