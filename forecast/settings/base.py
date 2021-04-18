import os
import sys

sys.path.append("/Users/corentinvasseur/Desktop/TestDecathlon/decathlon-test")

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(REPO_DIR, 'data')
LOGGING_CONFIGURATION_FILE = os.path.join(os.path.dirname(__file__), 'logging.yaml')

# RAW DATA CONFIG
DATA_DIR_RAW = os.path.join(DATA_DIR, 'raw')
DATA_DIR_OUTPUT = os.path.join(DATA_DIR, 'results')