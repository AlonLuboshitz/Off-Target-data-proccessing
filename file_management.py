import os
#import pyBigWig
from utilities import create_paths, validate_path
#import pybedtools

class File_management:
    # Positive and negative are files paths, pigenetics_bed and bigwig are folders path
    def __init__(self, models_path = None , ml_results_path = None, train_guides_path = None,
                  test_guides_path = None,  epigenetics_bed = None , epigenetic_bigwig = None, vivo_silico_path = None, vivo_vitro_path = None) -> None: 
        self.set_paths = False
        self.set_all_paths(models = models_path, ml_results = ml_results_path, train_guides = train_guides_path,
                           test_guides = test_guides_path, vivo_silico = vivo_silico_path, vivo_vitro = vivo_vitro_path, epi_folder = epigenetics_bed, bigiw_folder = epigenetic_bigwig)
        
        self.test_data_path = False
        
    ## Getters:
    def get_positive_path(self):
        return self.positive_path
    def get_negative_path(self):
        return self.negative_path
    def get_merged_data_path(self):
        if self.merged_data_path:
            return self.merged_data_path
        elif self.vivo_silico_path is None and self.vivo_vitro_path is None:
            raise RuntimeError("No merged data path set")
        else:
            raise RuntimeError("No silico/vitro bools data path set")
        

    def get_epigenetics_folder(self):
        return self.epigenetics_folder_path
    def get_bigwig_folder(self):
        return self.bigwig_folder_path
    def get_number_of_bigiwig(self):
        return 0
        return self.bigwig_amount
    def get_model_path(self):
        return self.models_path
    def get_ml_results_path(self):
        return self.ml_results_path
    def get_number_of_bed_files(self):
        return self.bed_files_amount
    def get_ensmbel_path(self):
        return self.ensmbel_train_path
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
    def get_seperate_test_data(self):
        if self.test_data_path:
            return self.test_data_path
    
    ## Setters:
    def set_all_paths(self, models, ml_results, train_guides, test_guides, vivo_silico, vivo_vitro, epi_folder, bigiw_folder):
        self.set_models_path(models)
        self.set_ml_results_path(ml_results)
        self.set_train_guides_path(train_guides)
        self.set_test_guides_path(test_guides)
        self.set_epigenetic_paths(epi_folder, bigiw_folder)
        self.set_vivo_silico_data_path(vivo_silico)
        self.set_vivo_vitro_data_path(vivo_vitro)
        self.set_paths = True

    def set_models_path(self, models_path):
        self.validate_path_exsits(models_path)
        self.models_path = models_path
    def set_ml_results_path(self, ml_results_path):
        self.validate_path_exsits(ml_results_path)
        self.ml_results_path = ml_results_path
    def set_train_guides_path(self, train_guides_path):
        self.validate_path_exsits(train_guides_path)
        self.train_guides_path = train_guides_path
    def set_test_guides_path(self, test_guides_path):
        self.validate_path_exsits(test_guides_path)
        self.test_guides_path = test_guides_path
    def set_epigenetic_paths(self, epigenetics_bed, bigwig):
        self.validate_path_exsits(epigenetics_bed)
        self.validate_path_exsits(bigwig)
        self.epigenetics_folder_path = epigenetics_bed
        self.bigwig_folder_path = bigwig
        #self.create_bigwig_files_objects()
        #self.set_global_bw_max()
        #self.create_bed_files_objects()
    
    ## Data paths - silico/vitro negative OTSs ##
    '''These functions dont validate the path, a dataset can have only one of the paths.
    When choosing to use one of the datasets then the path will be validated.'''
    def set_vivo_silico_data_path(self, vivo_silico_path):
        self.vivo_silico_path = vivo_silico_path
    def set_vivo_vitro_data_path(self, vivo_vitro_path):
        self.vivo_vitro_path = vivo_vitro_path

    def set_model_parameters(self,data_type, model_task, cross_validation, model_name, features, transformation = None):
        '''This function sets the model parameters to save the model in the corresponding path.
        
        Path: .../Models/{data_type}/{model_task}/{cross_validation}/{model_name}/{features}
              .../Ml_results/{data_type}/{model_task}/{cross_validation}/{model_name}/{features}
        '''
        if data_type == "silico":
            self.set_silico_vitro_bools(silico_bool = True)
        elif data_type == "vitro":
            self.set_silico_vitro_bools(vitro_bool = True)
        else:
            raise RuntimeError("Data type not set")
        if transformation:
            full_path = os.path.join(model_task,transformation,model_name,cross_validation,features)
        else:
            full_path = os.path.join(model_task,model_name,cross_validation,features)
        self.add_type_to_models_paths(full_path)
        self.task = model_task
        
        


    def set_silico_vitro_bools(self, silico_bool = False, vitro_bool = False):
        '''This function sets the merged data path to the vivo_silico_path or vivo_vitro_path.
        As well adds to model and model results path the vivo-silico/vivo-vitro suffix.
        Args:
        1. silico_bool - bool, default False
        2. vitro_bool - bool, default False
        -----------
        Returns: Error if both bools are false'''
        suffix_str = ""
        if silico_bool:
            self.validate_path_exsits(self.vivo_silico_path)
            self.merged_data_path = self.vivo_silico_path
            suffix_str = "vivo-silico"
        elif vitro_bool:
            self.validate_path_exsits(self.vivo_vitro_path)
            self.merged_data_path = self.vivo_vitro_path
            suffix_str = "vivo-vitro"
        else:
            raise RuntimeError("No silico vitro bools were given data path set")
        if (suffix_str not in self.models_path) and (suffix_str not in self.ml_results_path):
            self.add_type_to_models_paths(suffix_str)
        else:
            raise Exception(f"Suffix {suffix_str} already in model or results paths:\n {self.models_path}\n{self.ml_results_path}")
    
    def set_seperate_test_data(self, test_data_path,guides_path):
        '''This function sets the test data path and the guides path for a different test data then the initiated one.
        Args:
        1. test_data_path - str, path to the test data
        2. guides_path - str, path to the guides for the test data
        '''
        self.validate_path_exsits(test_data_path)
        self.test_data_path = True
        self.merged_data_path = test_data_path
        self.test_guides_paths = guides_path
    def add_type_to_models_paths(self, type):
        '''Given a type create folders in ML_results and Models with the type
        type will be anything to add:
        1. model type - cnn,rnn...
        2. cross val type -  k_fold, leave_one_out,ensmbel,
        3. features - only_seq, epigenetics, epigenetics_in_seq, spatial_epigenetics'''
        self.validate_ml_results_and_model()
       # create folders
        self.ml_results_path = self.add_to_path(self.ml_results_path,type)
        self.models_path = self.add_to_path(self.models_path,type)
  
    ## ENSMBELS:    

    def set_n_models(self, n_models):
        if n_models < 0 :
            raise RuntimeError('Number of models cannot be negative')
        self.n_models = n_models
        self.add_n_models_ensmbel_path()
    def add_partition_path(self):
        self.validate_ml_results_and_model()
        partition_str = "-".join(map(str,self.partition)) # Join the partition numbers into str seperated by '-'
        self.ml_results_path = self.add_to_path(self.ml_results_path,f'{partition_str}_partition')
        self.models_path = self.add_to_path(self.models_path,f'{partition_str}_partition')
    
    def add_n_models_ensmbel_path(self):
        self.validate_ml_results_and_model()
        partition_str = "-".join(map(str,self.partition)) # Join the partition numbers into str seperated by '-'
        self.ml_results_path = self.add_to_path(self.ml_results_path,f'{partition_str}_partition_{self.n_models}')
        self.models_path = self.add_to_path(self.models_path,f'{partition_str}_partition_{self.n_models}')
        
    def set_ensmbel_result_path(self, ensmbel_path):
        self.validate_path_exsits(ensmbel_path)
        self.ensmbel_result_path = ensmbel_path
    def set_ensmbel_guides_path(self, guides_path):
        self.validate_path_exsits(guides_path)
        if os.listdir(guides_path) == 0:
            raise Exception('Guides path is empty')
        self.train_guides_path = guides_path
    def add_to_path(self, path, path_to_add):
        '''Function take two paths, concatenate togther, create a folder and return the path'''
        temp_path = os.path.join(path, path_to_add)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        return temp_path
    def add_path_to_train_ensmbel(self, path_to_add):
        temp_path = os.path.join(self.ensmbel_train_path,path_to_add)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.ensmbel_train_path = temp_path
    def add_path_to_result_ensmbel(self, path_to_add):
        temp_path = os.path.join(self.ensmbel_result_path,path_to_add)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.ensmbel_result_path = temp_path    
    
    def set_partition(self, partition_list, train = False, test = False):
        '''Function to set the partition number
        The partition number argument is a list of numbers.
        Each number is a partition number.
        The fuction check if the partition number is in the range of the number of partitions
        If so it set the partition number list
        Other wise it raise an exception
        
        Args: 
        1. partition_list - list of ints, partitions numbers
        2. train - bool, default False, if True set the train partition
        3. test - bool, default False, if True set the test partition'''
        if train:
            self.guide_prefix = "Train"
            self.guides_partition_path = self.train_guides_path
        elif test:
            self.guide_prefix = "Test"
            self.guides_partition_path = self.test_guides_path
        else:
            raise ValueError('eather test or train must be set to True')
        
        # Check for number of partitions
        if os.path.exists(self.guides_partition_path):
            self.partition = []
            for partition in partition_list:
                    
                # check for partition
                if partition > 0 and partition <= len(os.listdir(self.guides_partition_path)):
                    self.partition.append(partition)
                else:
                    self.partition = [] # clear the list
                    raise Exception(f'Partition number {partition} is out of range')
        else: 
            raise Exception('Guides path not set')
        self.add_partition_path()

    def get_partition(self):
        if self.partition:
            return self.partition
        else: raise Exception('Partition not set')
    
    
    
    
    
    def get_guides_partition(self):
        '''Function to get the guides for the partition setted
        If partition/path not setted, raise exception
        Else sort the list of guides by partition number and 
        return the path of the partition guides
        '''
        if self.guides_partition_path:
            guides_list = os.listdir(self.guides_partition_path)
            if self.partition:
                guides_path = []
                for partition in self.partition:
                    guides_txt = f'{self.guide_prefix}_guides_{partition}_partition.txt'
                    if guides_txt not in guides_list:
                        raise RuntimeError(f'Guides for partition {partition} not found')
                    guides_path.append(os.path.join(self.guides_partition_path,guides_txt))
                return guides_path                
               
                
            else : raise Exception('Partition not set')
        else : raise Exception('Guides path not set')
        if self.train_guides_path:
            if self.test_data_path:
                return [self.test_guides_paths]
            

    
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
    

    '''Create pyBigWig objects list of the bigwig files'''
    def create_bigwig_files_objects(self):
        self.bigwig_files = []
        for path in create_paths(self.bigwig_folder_path):
            name = path.split("/")[-1].split(".")[0] # retain the name of the file (includes the marker)
            try:
                name_object_tpl = (name,pyBigWig.open(path))
                self.bigwig_files.append(name_object_tpl)
            except Exception as e:
                print(e)
        self.bigwig_amount = len(self.bigwig_files) # set amount
        
    # def create_bed_files_objects(self):
    #     self.bed_files = []
    #     for path in create_paths(self.epigenetics_folder_path):
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
    def create_ensemble_train_folder(self, i_ensmbel):
        output_path = os.path.join(self.models_path,f'ensemble_{i_ensmbel}')
        # create dir output_path if not exsits
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path
    
    def create_ensemble_result_folder(self, i_ensmbel):
        output_path = os.path.join(self.ensmbel_result_path,f'ensemble_{i_ensmbel}')
        # create dir output_path if not exsits
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path
    
    def create_ensemble_score_nd_combi_folder(self):
        if not self.ml_results_path:
            raise Exception("Ensemble result path not set")
        score_path = os.path.join(self.ml_results_path,"Scores")
        combi_path = os.path.join(self.ml_results_path,"Combi")
        
        if not os.path.exists(score_path):
            os.makedirs(score_path)
        if not os.path.exists(combi_path):
            os.makedirs(combi_path)
        return score_path,combi_path
        
    ## Validations
    '''Function to validate the paths'''
    def validate_path_exsits(self,path):
        assert os.path.exists(path), f"{path}Path does not exist"
    def validate_ml_results_and_model(self):
        if not self.ml_results_path:
            raise Exception("ML results path not set")
        if not self.models_path:
            raise Exception("Models path not set")
    '''dtor'''
    def __del__(self):
        #self.close_big_wig([])
        #self.close_bed_files()
        # call more closing
        pass