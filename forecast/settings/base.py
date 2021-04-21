import os
import sys

sys.path.append("/Users/corentinvasseur/Desktop/TestDecathlon/decathlon-test")

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(REPO_DIR, 'data')
LOGGING_CONFIGURATION_FILE = os.path.join(os.path.dirname(__file__), 'logging.yaml')

# RAW DATA CONFIG
DATA_DIR_RAW = os.path.join(DATA_DIR, 'raw')
DATA_DIR_PREPROCESSED = os.path.join(DATA_DIR, 'preprocessed')
DATA_DIR_OUTPUT = os.path.join(DATA_DIR, 'results')
DATA_DIR_NEW = os.path.join(DATA_DIR, 'new')

# MODEL SETTINGS
FEATURES = ['weekofyear_cos_1', 'weekofyear_sin_1', 'x', 'y', 'z', 'dpt_num_department_88', 
                'dpt_num_department_117', 'dpt_num_department_127', 'zod_idr_zone_dgr_3', 
                'zod_idr_zone_dgr_4', 'zod_idr_zone_dgr_6', 'zod_idr_zone_dgr_10', 'zod_idr_zone_dgr_35', 
                'zod_idr_zone_dgr_59', 'zod_idr_zone_dgr_72']

LIST_FEATURES_TO_DUMMY = ['dpt_num_department',"zod_idr_zone_dgr"]