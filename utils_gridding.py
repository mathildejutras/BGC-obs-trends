"""
utils_gridding.py
2025

Author: Mathilde Jutras
Contact: mathilde_jutras@uqar.ca

This script accompanies script gridded_trends_from_obs.py
and contains functions used within that script

Description:

Requirements:
    - Python >= 3.9.23
    Packages:
    - numpy=1.26.4
    - scipy=1.13.1
    - seawater=3.3.5
    - scikit-learn=1.6.1

License:
    MIT
"""

import seawater as sw
import numpy as np
import math
import scipy.stats as stats
import time as time_package
from sklearn.utils import resample 
from sklearn.metrics import mean_squared_error


# --- Function to apply Argo QC flags

def apply_QC_flags_argo(ds, svar):

    sf = svar+'_QC'
    flags = ds[sf].astype('float')

    valid_mask = (flags != 3) & (flags != 4) & (flags != 9) & (flags != 0) & (flags != 2)

    ds[svar] = ds[svar].where(valid_mask, np.nan)

    return ds


# --- Function to calculate mixed layer

def mld_dbm_v3(temp_cast, salin_cast, press_cast, sig_theta_threshold):

    # Sort cast by increasing depth
    depth_cast = sw.dpth(press_cast, -40)
    dep_sort = np.sort(depth_cast)
    ix = np.argsort(depth_cast)
    temp = temp_cast[ix]
    salin = salin_cast[ix]

    # Compute potential temperature and potential density
    theta = sw.ptmp(salin, temp, dep_sort, 0)
    sig_theta = sw.dens0(salin, theta)

    # Find reference depth
    ref_ind = np.argmax(dep_sort > 9)

    # Reference depth is too deep - no shallow depths in cast
    if dep_sort[ref_ind] > 25:
        return np.nan

    # Choose the reference sigma depending on the threshold chosen
    ref_sig_theta = sw.dens0(salin[ref_ind], theta[ref_ind]) + sig_theta_threshold

    # Search for MLD
    if np.sum(~np.isnan(sig_theta)) > 1:  # Not a one-point or all-NaN cast
        # Find mixed layer depth
        not_found = True
        start_ind = ref_ind
        iter_count = 1
        while not_found:
            # Begin search at reference (10 m) index
            # Find next point below reference depth that exceeds criterion
            ml_ind = np.argmax(sig_theta[start_ind:] > ref_sig_theta)
            if len(sig_theta) >= ml_ind + start_ind:  # ml_ind is within the interior of the cast
                if sig_theta[ml_ind + start_ind] > ref_sig_theta:  # Next point also meets criterion, therefore likely not a spike
                    not_found = False
                    ml_ind += start_ind - 1  # Final index
            else:  # Last point in cast
                not_found = False
                ml_ind += start_ind - 1
            # If a spike, start search again at first point after spike
            start_ind = ml_ind + start_ind
            iter_count += 1
            # Break loop if cast is all spikes/no MLD found
            if iter_count > len(sig_theta):
                break
        # If an MLD is found, interpolate to find depth at which density = ref_sig_theta
        # added that not first depth
        if not not_found:
            if not np.any(np.isnan(sig_theta[ml_ind - 1:ml_ind])) and ml_ind > 0:
                mld_out = np.interp(ref_sig_theta, sig_theta[ml_ind - 1:ml_ind], dep_sort[ml_ind - 1:ml_ind])
            else:
                mld_out = np.nan
        else:
            mld_out = np.nan

    else:
        mld_out = np.nan

    return mld_out


# --- Detect linear trend in grid cell

def fit_trend(x, y, plot_local=False, verbose=False):

    """
    Parameters:
    x: x data
    y: y data
    plot_local: to plot time series 
    verbose: print the statistical information

    Returns: fit, coeffs_linear, coeffs_poly, se_slope, [x_pred_boot, lower, upper]
    fit: type of fit:   0 if no significant trend, 
                        1 if statistically significant linear fit, 
                        2 if statistically significant polynomial fit (and also significant linear fit)
    coeffs_linear: slope of linear fit
    coeffs_poly: coefficients of the polynomial fit
    se_slope: slope error
    x_pred_boot: x values for the bootstrap estimates
    lower: y values for the lower bootstrap estimates
    upper: y values for the upper bootstrap estimates
    """

    # do not compute trend if only two datapoints
    if len(x) <= 2:
        return None, [np.nan, np.nan], [np.nan, np.nan, np.nan], np.nan, [[], [], []]

    # use a loosser criteria for p_value if very little datapoints
    if len(x) < 5:
        p_crit = 0.4 # p_value criteria for significance
    else:
        p_crit = 0.1

    if verbose:
        print(' ')

    # Fit models
    try:
        coeffs_linear = np.polyfit(x, y, 1)
    except:
        #plt.scatter(x,y) ; plt.show()
        print('polyfit did not converge')
        return  None, [np.nan, np.nan], [np.nan, np.nan, np.nan], np.nan, [[], [], []]
    if len(x) > 6:
        coeffs_poly = np.polyfit(x, y, 2)
    else:
        coeffs_poly = [np.nan, np.nan, np.nan]

    # Predicted values
    y_pred_linear = np.polyval(coeffs_linear, x)
    y_pred_poly = np.polyval(coeffs_poly, x)

    # Residual sum of squares
    rss_total = np.sum((y - np.mean(y))**2)
    rss_linear = np.sum((y - y_pred_linear)**2)
    rss_poly = np.sum((y - y_pred_poly)**2)
    #mse = mean_squared_error(y, y_pred_linear)

    # Number of predictors
    pred_linear = len(coeffs_linear) - 1  # Number of predictors in the linear model
    pred_poly = len(coeffs_poly) - 1  # Number of predictors in the polynomial model
    # Degrees of freedom
    n = len(x)
    df_linear = n - len(coeffs_linear)  # Linear model degrees of freedom
    df_poly = n - len(coeffs_poly)    # Polynomial model degrees of freedom

    # Calculate uncertainty on slope of linear fit
    rse = np.sqrt( rss_linear / (n-2) ) # residual standard error
    se_slope = rse / np.sqrt( np.sum(( x - np.mean(x))**2) ) # standard error


    if df_poly > 1: # if enough data to calculate polynomial fits
        # F-statistic for comparison
        F = ((rss_linear - rss_poly) / (df_linear - df_poly)) / (rss_poly / df_poly)
        p_value = 1 - stats.f.cdf(F, df_linear - df_poly, df_poly)

        if verbose:
            print('Test variability')
            print(f"F-statistic: {F}")
            print(f"p-value: {p_value}")
    else:
        p_value = np.nan

    # bootstrap for confidence intervals
    if (len(x) > 5) & plot_external: # only calculate to plot them. Don't calculate if don't need since slow
        n_inter = 1000
        predictions = [] ; slopes = []
        for _ in range(n_inter):
            x_boot, y_boot = resample(x, y)
            with warnings.catch_warnings():
                warnings.simplefilter('error', np.lib.polynomial.PolyfitRankWarning)
                coefs = np.polyfit(x_boot, y_boot, 1)
            x_pred_boot = np.linspace(min(x), max(x), 100)
            predictions.append( np.polyval(coefs, x_pred_boot) )
            slopes.append( coefs[0] )

        # Confidence intervals
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)

        # Not consider if the CI are way too large because too little data
        if max(upper-lower) > 1000:
            lower = [] ; upper = [] ; x_pred_boot = []
    else:
        lower = [] ; upper = [] ; x_pred_boot = []

    fit = None
    # If polynomial fit explains the data better
    if p_value < 0.01:

        # test the statistical significance of the poly fit
        F_poly = ((rss_total - rss_poly) / pred_poly) / (rss_poly / (n - pred_poly - 1))
        p_value_poly = 1 - stats.f.cdf(F_poly, pred_poly, n - pred_poly - 1)

        if p_value_poly < 0.05:
            title = 'Polynomial fit is significantly better than the linear fit\nand is statistically significant.'
            if verbose:
                print(title)
            fit = 'Poly'
        else:
            if verbose:
                print('Polynomial fit was not statistically significant')

    # If polynomial fit is not significant, evaluate linear fit
    if (p_value >= 0.01) | (fit==None) | np.isnan(p_value):

        # F-statistic for Linear Model
        F_linear = ((rss_total - rss_linear) / pred_linear) / (rss_linear / (n - pred_linear - 1))
        p_value_linear = 1 - stats.f.cdf(F_linear, pred_linear, n - pred_linear - 1)

        if verbose:
            print('Test linear fit')
            print(f"F-statistic: {F_linear}")
            print(f"p-value: {p_value_linear}")

        if p_value_linear < p_crit:
            title = 'Linear fit is statistically significant'
            if verbose:
                print(title)
            fit = 'Linear'

        else:
            title = 'No fit'
            if verbose:
                print(title)

    # --- plot and save
    if plot_local:
        pathsave = '../figures/on_grid/tests/'
        ct = time_package.time()
        cs=[0]
        for filename in [pathsave+each for each in os.listdir(pathsave) if 'fit_' in each]:
            file_creation_time = os.path.getctime(filename)
            # Check if the file was created within the last three minute
            if ct - file_creation_time > 180:
                os.remove(filename)
            else:
                cs.append( int(filename.replace(pathsave+'fit_','').replace('.png','')) )

        plt.clf() ; plt.close()
        plt.figure(figsize=(6,3.5))
        plt.plot(x, y, 'o', label='data')
        plt.plot(x, y_pred_linear, '-k', label='linear fit')
        plt.fill_between(x_pred_boot, lower, upper, color='gray', alpha=0.3)
        plt.plot(x, y_pred_poly, '--k', label='poly fit')
        plt.title(title)
        plt.legend()
        #plt.savefig('../figures/on_grid/tests/fit_%i.png'%(max(cs)+1), dpi=300)
        plt.show()

    return fit, coeffs_linear, coeffs_poly, se_slope, [x_pred_boot, lower, upper]
