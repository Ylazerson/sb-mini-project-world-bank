# B''H #


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This will allow the module to be import-able from other scripts and callable from arbitrary places in the system.
MODULE_DIR = os.path.dirname(__file__)

PROJ_ROOT = os.path.join(MODULE_DIR, os.pardir)

DATA_DIR = os.path.join(PROJ_ROOT, 'data')

def print_dir_constants():
    print('MODULE_DIR               :', MODULE_DIR)    
    print('PROJ_ROOT                :', PROJ_ROOT)    
    print('DATA_DIR                 :', DATA_DIR)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

