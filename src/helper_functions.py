# B''H #


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import numpy as np
import pandas as pd
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This will allow the module to be import-able from other scripts and callable from arbitrary places in the system.
MODULE_DIR = os.path.dirname(__file__)

PROJ_ROOT = os.path.join(MODULE_DIR, os.pardir)

DATA_DIR = os.path.join(PROJ_ROOT, 'data')
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_csv(
    p_dir,
    p_file_name,
    p_sep        = ',',
    p_header     = 'infer',
    p_names      = None,
    p_index_col  = None,
    p_compression= None,
    p_dtype      = None,
    p_parse_dates= False,
    p_skiprows   = None,
    p_chunksize  = None
):

    v_file = os.path.join(p_dir, p_file_name)

    df = pd.read_csv(
        v_file,
        sep        = p_sep,
        header     = p_header,
        names      = p_names,
        index_col  = p_index_col,
        compression= p_compression,
        dtype      = p_dtype,
        parse_dates= p_parse_dates,
        skiprows   = p_skiprows,
        chunksize  = p_chunksize
    )

    return df
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                    x-data for the ECDF: x
    #
    # The x-values are the sorted data.
    x = np.sort(data)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                    y-data for the ECDF: y
    #
    # The y data of the ECDF go from 1/n to 1 in equally spaced increments.
    # You can construct this using np.arange().
    # Remember, however, that the end value in np.arange() is not inclusive.
    # Therefore, np.arange() will need to go from 1 to n+1.
    # Be sure to divide this by n.
    y = np.arange(1, n+1) / n

    return x, y
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_x_y_for_norm_plot(p_data):
    """Generates data for a normal probability plot.
   
    Returns:
        x: random values from the standard normal distribution.
        y: the sorted values from the data         
    """

    # -- -------------------------------------------
    # From a standard normal distribution (µ = 0 and σ = 1)
    #     - generate a random sample with the same size as the data
    #     - sort it
    mu    = 0
    sigma = 1
    sample_size = len(p_data)

    x = np.random.normal(mu, sigma, sample_size)
    x.sort()
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Sort the values in the data
    y = np.array(p_data)
    y.sort()
    # -- -------------------------------------------

    return x, y
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_x_y_for_line(bounds, y_intercept, slope):
    """
    Get x and y for plotting a straight line.
    """    

    x = np.sort(bounds)

    y = y_intercept + (slope * x)

    return x, y
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_srr_bins(p_data):
    """
    Get number of bins using the "square root rule"
    """
    
    n_data = len(p_data)
    
    n_bins = np.sqrt(n_data)
    
    return int(n_bins)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bootstrap_replicate_1d(data, func):
    """
    Generate bootstrap replicate of 1-dimensional data
    """
    bs_sample = np.random.choice(data, len(data))

    return func(bs_sample)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_bootstrap_replicates(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
