

## PATHS:
DATA_PATH = "/home/dsi/lubosha/Off-Target-data-proccessing/Data"
ML_RESULTS = "/home/dsi/lubosha/Off-Target-data-proccessing/ML_results"
MODELS_PATH = "/home/dsi/lubosha/Off-Target-data-proccessing/Models"


## CHANGESEQ
# Preprocessed data
CHANGESEQ_VIVOSILICO_DATA = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/vivosilico_nobulges_withEpigenetic_indexed.csv"
CHANGESEQ_VIVOVITRO_DATA = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/vivovitro_nobulges_withEpigenetic_indexed.csv"
# MODELS:
CHANGESEQ_MODEL_PATH = "/localdata/alon/Models/Change-seq"
CHANGESEQ_RESULTS_PATH = "/localdata/alon/ML_results/Change-seq"
CHANGESEQ_GUIDES_FOLDER = "/home/dsi/lubosha/Off-Target-data-proccessing/Models/Change_seq_gs_cs/CNN/Ensemble/Test_guides"

    # Epigenetics
EPIGENETIC_FOLDER = "/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics"
BED_FILES_FOLDER = "/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/Chromstate"
BIG_WIG_FOLDER = "/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/bigwig"
CHANGESEQ_DICT = {"Test_guides": CHANGESEQ_GUIDES_FOLDER, "Vivo-silico": CHANGESEQ_VIVOSILICO_DATA, "Model_path": CHANGESEQ_MODEL_PATH, "ML_results": CHANGESEQ_RESULTS_PATH,
               "Vivo-vitro": CHANGESEQ_VIVOVITRO_DATA}
#### HENDEL LAB
TEST_GUIDES_HENDEL = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Partitions_guides"
HENDEL_GS_SILICO = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv"
HENDEL_MODEL_PATH = "/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data"
HENDEL_ML_RESULTS_PATH = "/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data"
HENDEL_DICT = {"Test_guides": TEST_GUIDES_HENDEL, "Vivo-silico": HENDEL_GS_SILICO, "Model_path": HENDEL_MODEL_PATH, "ML_results": HENDEL_ML_RESULTS_PATH,
               "Vivo-vitro": None}