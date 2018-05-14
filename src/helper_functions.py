# B''H #


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import sys


from collections import namedtuple

import glob

import numpy as np
import pandas as pd

import tweepy

import json

from IPython.display import display, Markdown
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This will allow the module to be import-able from other scripts and callable from arbitrary places in the system.
MODULE_DIR = os.path.dirname(__file__)

PROJ_ROOT = os.path.join(MODULE_DIR, os.pardir)

DATA_DIR = os.path.join(PROJ_ROOT, 'data')

DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')

DATA_CLEAN_DIR = os.path.join(DATA_DIR, 'clean')

DATA_TWEETS_DIR = os.path.join(DATA_DIR, 'tweets')

NOTEBOOKS_DIR = os.path.join(PROJ_ROOT, 'notebooks')

NOTEBOOK_FILES_DIR = os.path.join(NOTEBOOKS_DIR, 'files')

SEC_13_DIR = os.path.join(NOTEBOOK_FILES_DIR, '13-df-etl')
SEC_14_DIR = os.path.join(NOTEBOOK_FILES_DIR, '14-advanced-indexing')
SEC_15_DIR = os.path.join(NOTEBOOK_FILES_DIR, '15-reshaping-data')
SEC_16_DIR = os.path.join(NOTEBOOK_FILES_DIR, '16-grouping-data')
SEC_17_DIR = os.path.join(NOTEBOOK_FILES_DIR, '17-summer-olympics')
SEC_18_DIR = os.path.join(NOTEBOOK_FILES_DIR, '18-preparing-data')
SEC_19_DIR = os.path.join(NOTEBOOK_FILES_DIR, '19-concatenating-data')
SEC_20_DIR = os.path.join(NOTEBOOK_FILES_DIR, '20-merging-data')
SEC_21_DIR = os.path.join(NOTEBOOK_FILES_DIR, '21-cleaning-data')
SEC_22_DIR = os.path.join(NOTEBOOK_FILES_DIR, '22-life-expectancy-case-study')
SEC_23_DIR = os.path.join(NOTEBOOK_FILES_DIR, '23-python-for-ds-part-2')
SEC_24_DIR = os.path.join(NOTEBOOK_FILES_DIR, '24-world-bank-case-study')
SEC_25_DIR = os.path.join(NOTEBOOK_FILES_DIR, '25-importing-data-part-1')
SEC_26_DIR = os.path.join(NOTEBOOK_FILES_DIR, '26-importing-data-part-2')

def print_dir_constants():
    print('MODULE_DIR               :', MODULE_DIR)    
    print('PROJ_ROOT                :', PROJ_ROOT)    
    print('DATA_DIR                 :', DATA_DIR)
    print('DATA_RAW_DIR             :', DATA_RAW_DIR)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_csv(
    p_dir, 
    p_file_name, 
    p_sep         = ',',
    p_header      = 'infer',
    p_names       = None,
    p_index_col   = None,
    p_compression = None, 
    p_dtype       = None,
    p_parse_dates = False,
    p_skiprows    = None,
    p_chunksize   = None
):

    v_file = os.path.join(p_dir, p_file_name)

    df = pd.read_csv(
        v_file,
        sep         = p_sep,
        header      = p_header,
        names       = p_names,
        index_col   = p_index_col,
        compression = p_compression,
        dtype       = p_dtype,
        parse_dates = p_parse_dates,
        skiprows    = p_skiprows,
        chunksize   = p_chunksize
    )

    return df
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def display_images(p_dir):

    file_list = glob.glob(os.path.join(p_dir, "*.png"))

    for file_path in sorted(file_list):

        file_name = file_path.replace(p_dir, '')[1:]
    
        file_relative_path = file_path[file_path.find('/files/')+1:]
    
        content = '### '+file_name+'\n'+'<img src="'+file_relative_path+'">\n\n---'
        display(Markdown(content))
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MyStreamListener(tweepy.StreamListener):    
    """
    This Tweet listener 
        - creates a file called 'tweets.txt' 
        - collects streaming tweets as .jsons and writes them to the file 'tweets.txt' 
        - once 100 tweets have been streamed, the listener closes the file and stops listening.
    """
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open(
            os.path.join(DATA_TWEETS_DIR, "tweets.txt"),
            "w"
            )

    def on_status(self, status):
        tweet = status._json
        self.file.write(json.dumps(tweet) + '\n')
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



