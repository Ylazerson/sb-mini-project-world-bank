# B''H #


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

from collections import namedtuple

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
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


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sem(p_provided_std, p_sample_size):
    """
    Return the Standard Error of the Mean
    """

    return p_provided_std / np.sqrt(p_sample_size)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def z_t_test_single_sample(p_sample_mean, p_hypothesized_mean, p_provided_std, p_sample_size):
    """
    Peform a single sample z or t test.
    """

    # -- -------------------------------------------
    sem = p_provided_std / np.sqrt(p_sample_size)

    z_t_stat = (p_sample_mean - p_hypothesized_mean) / sem
    # -- -------------------------------------------

    # -- -------------------------------------------
    Result = namedtuple('Result', 'z_t_stat sem')

    result = Result(
        z_t_stat,
        sem
    )

    return result
    # -- -------------------------------------------

# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_val_at_standard_score(p_standard_score, p_hypothesized_mean, p_sem):
    """
    Get the value that corresponds to the z or t score   
    """

    return (p_standard_score * p_sem) + p_hypothesized_mean
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_two_tailed_critical_values(p_alpha):
    """
    Get the lower and upper critical values that indicate the end ot the nonrejection area.  
    """

    Result = namedtuple('Result', 'lower_critical_value upper_critical_value')

    result = Result(
        p_alpha/2,
        1 - (p_alpha/2)
    )

    return result
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def print_namedtuple(p_namedtuple):

    for name, value in p_namedtuple._asdict().items():
        print(name.ljust(25), ':', value)
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_ci_sigma_unknown(p_data, p_alpha):
    """
    Get the confidence interval where sigma is unknown 
    """

    # -- -------------------------------------------
    # Gather the core statistic values
    sample_mean = np.mean(p_data)
    sample_std  = np.std(p_data, ddof=1)  # using ddof=1 for sample std
    sample_size = len(p_data)

    df = sample_size - 1
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Get the standard error of the mean 
    sem = get_sem(
        p_provided_std      = sample_std,
        p_sample_size       = sample_size
    )
    # -- -------------------------------------------


    # -- -------------------------------------------    
    # Calculate the Margin of Error and Confidence Interval
    _, upper_critical_value = get_two_tailed_critical_values(p_alpha = p_alpha)

    upper_critical_t = stats.t.ppf(upper_critical_value, df)

    # Get the margin of error:
    moe = upper_critical_t * sem

    # Calculate the confidence interval:
    ci = np.array([sample_mean - moe, sample_mean + moe])
    # -- -------------------------------------------


    # -- -------------------------------------------
    Result = namedtuple('Result', 'sample_mean sample_std sample_size alpha sem confidence_level_pct critical_t_statistic margin_of_error confidence_interval')

    result = Result(
        sample_mean,
        sample_std,
        sample_size,
        p_alpha,
        sem,
        upper_critical_value * 100,
        upper_critical_t,
        moe,
        ci
    )

    return result
    # -- -------------------------------------------

# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



  

# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_two_tailed_t_test(p_hypothesized_mean, p_data, p_alpha, p_data_content_desc):
    '''
    Plot a single sample t-test.
    '''

    # -- -------------------------------------------
    # Gather the core statistic values
    sample_mean = np.mean(p_data)
    sample_std  = np.std(p_data, ddof=1)  # using ddof=1 for sample std
    sample_size = len(p_data)

    df = sample_size - 1
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Get the t score and sem
    t, sem = z_t_test_single_sample(
        p_sample_mean       = sample_mean,
        p_hypothesized_mean = p_hypothesized_mean,
        p_provided_std      = sample_std,
        p_sample_size       = sample_size
    )
    # -- -------------------------------------------    


    # -- -------------------------------------------
    # Get the lower and upper critical boundary values
    lower_critical_value, upper_critical_value = get_two_tailed_critical_values(p_alpha = p_alpha)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Get the lower and upper critical boundary t scores
    lower_critical_t = stats.t.ppf(lower_critical_value, df)

    upper_critical_t = stats.t.ppf(upper_critical_value, df)
    # -- -------------------------------------------
   

    # -- -------------------------------------------
    # Get the p-value
    p_value = stats.t.cdf(t, df)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Create the x ticks
    xticks = list(range(-6, 7, 2))
    
    xticks.extend([t, lower_critical_t, upper_critical_t])
    
    xticks = sorted(xticks)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Create the x tick labels
    xticklabels = []
    for x in xticks:
        xticklabels.append(
            round(
                get_val_at_standard_score(
                    p_standard_score   = x,
                    p_hypothesized_mean= p_hypothesized_mean,
                    p_sem              = sem
                ), 1
            )
        )
    # -- -------------------------------------------
    

    # -- -------------------------------------------
    # Setup up the plot axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Plot t-distribution on ax1
    rv = stats.t(
        df    = df,
        loc   = 0,
        scale = 1
    )

    x = np.linspace(
        rv.ppf(0.0001),
        rv.ppf(0.9999),
        100
    )

    y = rv.pdf(x)

    _ = ax1.plot(x, y)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Plot the t values on ax1
    _ = ax1.axvline(
        x     = lower_critical_t,
        color = 'green'
    )

    _ = ax1.axvline(
        x     = upper_critical_t,
        color = 'green',
        label = 'critical values (\u03B1 = '+str(p_alpha)+')'
    )

    _ = ax1.axvline(
        x     = t,
        color = 'red',
        label = 't (p-value = '+str(round(p_value, 3))+')'
    )
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Set ax1 lables and ticks
    _ = ax1.set_xlabel('t scores (df = '+str(df)+')')
    _ = ax1.set_ylabel('PDF')
    _ = ax1.legend(loc='upper right')

    _ = ax1.set_xlim(-7, 7)
    _ = ax1.set_xticks(xticks)
    _ = ax1.tick_params(axis = 'x', labelrotation = 70)

    _ = ax1.set_ylim(0, .7)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Set ax2 labels and ticks
    _ = ax2.set_xlim(ax1.get_xlim())
    
    _ = ax2.set_xticks(xticks)
    _ = ax2.set_xticklabels(xticklabels)
    _ = ax2.tick_params(axis = 'x', labelrotation = 70)

    _ = ax2.set_xlabel(p_data_content_desc+' (SEM = '+str(round(sem, 2))+')')

    plt.show()
    # -- -------------------------------------------

# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_two_tailed_z_test(p_hypothesized_mean, p_data, p_alpha, p_data_content_desc):
    '''
    Plot a single sample z-test.
    '''

    # -- -------------------------------------------
    # Gather the core statistic values
    sample_mean  = np.mean(p_data)
    provided_std = np.std(p_data, ddof = 0) # ddof=0 because we're using (or appropriating) the population std 
    sample_size  = len(p_data)

    df = sample_size - 1
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Get the z score and sem
    z, sem = z_t_test_single_sample(
        p_sample_mean       = sample_mean,
        p_hypothesized_mean = p_hypothesized_mean,
        p_provided_std      = provided_std,
        p_sample_size       = sample_size
    )
    # -- -------------------------------------------    


    # -- -------------------------------------------
    # Get the lower and upper critical boundary values
    lower_critical_value, upper_critical_value = get_two_tailed_critical_values(p_alpha = p_alpha)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Get the lower and upper critical boundary z scores
    lower_critical_z = stats.norm.ppf(lower_critical_value)

    upper_critical_z = stats.norm.ppf(upper_critical_value)
    # -- -------------------------------------------
   

    # -- -------------------------------------------
    # Get the p-value
    p_value = stats.norm.cdf(z)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Create the x ticks
    xticks = list(range(-6, 7, 2))
    
    if .03 <= p_alpha <= .07:
        # don't add lower/upper critical z values because overlap with std of 2
        xticks.append(z)
    else:
        xticks.extend([z, lower_critical_z, upper_critical_z])
    
    xticks = sorted(xticks)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Create the x tick labels
    xticklabels = []
    for x in xticks:
        xticklabels.append(
            round(
                get_val_at_standard_score(
                    p_standard_score   = x,
                    p_hypothesized_mean= p_hypothesized_mean,
                    p_sem              = sem
                ), 1
            )
        )
    # -- -------------------------------------------
    

    # -- -------------------------------------------
    # Setup up the plot axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Plot z-distribution on ax1
    mu       = 0
    variance = 1
    sigma    = np.sqrt(variance)

    x = np.linspace(
        mu - 3*sigma, 
        mu + 3*sigma, 
        100
    )

    y = stats.norm.pdf(
        x, 
        mu, 
        sigma    
    )

    _ = ax1.plot(x, y)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Plot the z values on ax1
    _ = ax1.axvline(
        x     = lower_critical_z,
        color = 'green'
    )

    _ = ax1.axvline(
        x     = upper_critical_z,
        color = 'green',
        label = 'critical values (\u03B1 = '+str(p_alpha)+')'
    )

    _ = ax1.axvline(
        x     = z,
        color = 'red',
        label = 'z (p-value = '+str(round(p_value, 3))+')'
    )
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Set ax1 lables and ticks
    _ = ax1.set_xlabel('z scores')
    _ = ax1.set_ylabel('PDF')
    _ = ax1.legend(loc='upper right')

    _ = ax1.set_xlim(-7, 7)
    _ = ax1.set_xticks(xticks)
    _ = ax1.tick_params(axis = 'x', labelrotation = 70)

    _ = ax1.set_ylim(0, .7)
    # -- -------------------------------------------


    # -- -------------------------------------------
    # Set ax2 labels and ticks
    _ = ax2.set_xlim(ax1.get_xlim())
    _ = ax2.set_xticks(xticks)
    _ = ax2.set_xticklabels(xticklabels)
    _ = ax2.tick_params(axis = 'x', labelrotation = 70)
    
    _ = ax2.set_xlabel(p_data_content_desc+' (SEM = '+str(round(sem, 2))+')')

    plt.show()
    # -- -------------------------------------------

# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_permutation_replicates(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def diff_frac(data_1, data_2):
    """Difference in rate of two arrays."""

    frac_1 = np.sum(data_1) / len(data_1)
    frac_2 = np.sum(data_2) / len(data_2)

    return frac_1 - frac_2
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def two_proportion_standard_error(hits1, attempts1, hits2, attempts2):
    """Return the standard error of two proportions."""
    
    # Calculate proportions:
    proportion1 = hits1/attempts1
    proportion2 = hits2/attempts2
    
    # Calculate standard error:
    SE = np.sqrt(proportion1*(1 - proportion1)/attempts1  +  proportion2*(1 - proportion2)/attempts2)
    return SE
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     


# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def two_proportion_confidence_interval(hits1, attempts1, hits2, attempts2, alpha=0.05):
    """Return the confidence interval for a two-proportion test."""
    
    # Calculate proportions:
    proportion1 = hits1/attempts1
    proportion2 = hits2/attempts2
    difference_of_proportions = proportion1 - proportion2
    
    # Calculate standard error:
    SE = two_proportion_standard_error(hits1, attempts1, hits2, attempts2)
    
    # Save the critical value at the specified confidence:
    z_critical = stats.norm.ppf(1 - 0.5*alpha)
    
    # Calculate margin of error:
    moe = z_critical * SE
    
    # Calculate confidence intervals:
    confidence_lower = proportion2 - proportion1 - moe
    confidence_higher = proportion2 - proportion1 + moe
    
    return difference_of_proportions, moe, confidence_lower, confidence_higher
# -- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    