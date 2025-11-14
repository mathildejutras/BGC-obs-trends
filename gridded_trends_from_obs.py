"""
gridded_trends_from_obs.py
2025

Author: Mathilde Jutras
Contact: mathilde_jutras@uqar.ca

This script was used to generate the results of manuscript XXXX

Description:
This script
1) Reads different observational datasets (for now Argo data, GLODAP, and WOD)
2) Puts this data on a lat-lon-neutral density grid
3) Calculates long-term trends within each grid cell using a linear fit, 
   testing for the statistical validity of the trend

Requirements:
    - Python >= 3.9.23
    Packages:
    - numpy=1.26.4
    - pandas=2.3.1
    - xarray=2023.6.0
    - scipy=1.13.1
    - dask=2024.5.0
    - statsmodels=0.14.5
    - pygamman
    - joblib=1.4.2

License:
    MIT
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime
import pandas as pd
import pygamman.gamman as nds
from joblib import Parallel, delayed
import matplotlib.colors as mcolors

# Functions in separate files
import utils_gridding import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import faulthandler
faulthandler.enable()

# ------------------------------------------------------------------------------
# SET-UP
# ------------------------------------------------------------------------------
dataset_name = 'all' # all datasets, or glodap, or argo, or wod only

get_data = False        # load and bin the data, then save it
calc_trends = False  # calculate the trends
plot_trends = True  # if make maps of results

# select grid size
lon_grid = 2.5
lat_grid = 2.5
# Select if calculate trends using all data or focusing on specific period
period = '' # '_since2000': including data since 2000; '_sincepriorXXXX': must include data from before XXXX

n_jobs = 8  # processors used by parallel processing

# ------------------------------------------------------------------------------
# ---------
# load data
# ---------
data_dir_glo = 'PATH_TO_GLODAP_DATA/'
data_dir_argo = 'PATH_TO_ARGO_DATA/'
output_dir_figs = 'PATH_TO_WHERE_SAVE_FIGURE/'

lon_vec = list(np.arange(-180, 180 + 1, lon_grid))
lat_vec = list(np.arange(-80, 80 + 1, lat_grid))
lon_meshgrid, lat_meshgrid = np.meshgrid(lon_vec, lat_vec)

gamma_range = np.arange(25.7,28.1,0.1)
time_bins = np.arange(1960,2026,1)

# limit range for figures
lims = {'TEMP_ADJUSTED':0.05, 'PSAL_ADJUSTED':0.015, 'DOXY_ADJUSTED':1, 'NITRATE_ADJUSTED':0.2, 'DIC':2, 'PH_IN_SITU_TOTAL_ADJUSTED':0.005, 'PRES_ADJUSTED':10, 'spiciness0':0.02, 'PHOSPHATE_ADJUSTED':0.02}
# variables to analyze
vars = ['TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'DOXY_ADJUSTED', 'NITRATE_ADJUSTED', 'DIC', 'PH_IN_SITU_TOTAL_ADJUSTED', 'PRES_ADJUSTED', 'PHOSPHATE_ADJUSTED']

# this will be used to load glodap data
dtype={'G2aou': 'float64', 'G2c13': 'float64', 'G2c14': 'float64', 'G2c14err': 'float64','G2ccl4': 'float64','G2cfc11': 'float64','G2cfc113': 'float64','G2cfc12': 'float64','G2chla': 'float64', 'G2doc': 'float64', 'G2fco2': 'float64','G2fco2temp': 'float64','G2gamma': 'float64', 'G2h3': 'float64','G2h3err': 'float64','G2he': 'float64','G2he3': 'float64','G2he3err': 'float64', 'G2neon': 'float64','G2nitrate': 'float64','G2nitrite': 'float64','G2o18': 'float64','G2oxygen': 'float64','G2pccl4': 'float64','G2pcfc11': 'float64','G2pcfc113': 'float64','G2pcfc12': 'float64','G2phosphate': 'float64','G2phts25p0': 'float64','G2phtsinsitutp': 'float64','G2psf6': 'float64','G2sf6': 'float64','G2silicate': 'float64','G2talk': 'float64','G2tco2': 'float64'}

# ------
def extract_data(lonbin, latbin):
    """
    Find all the observational data falling within one grid cell.

    Parameters:
    lonbin: longitude of the grid cell
    latbin: latitude of the grid cell

    Returns:
    no output. Saves a csv file with all the data falling within this grid cell.
    """

    print('(', lonbin, latbin, ')')

    # Read the matching Argo files for this grid cell
    if (dataset_name == 'all') | (dataset_name == 'argo'):

        # create an empty local dataframe that we will fill
        df_argo = pd.DataFrame(columns=['G2cruise', 'G2station', 'G2cast', 'LATITUDE', 'LONGITUDE', 'PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'sigma0',
                            'DOXY_ADJUSTED', 'NITRATE_ADJUSTED', 'PHOSPHATE_ADJUSTED', 'DIC', 'PH_IN_SITU_TOTAL_ADJUSTED', 'datetime', 'gamma', 'Depth'])

        argo_list = argo_info[ (lonbin >= np.round(argo_info['MINLON'],1)) & (lonbin+lon_grid <= np.round(argo_info['MAXLON'])) & (latbin >= np.round(argo_info['MINLAT'])) & (latbin+lat_grid <= np.round(argo_info['MAXLAT'])) ].PLATFORM_NUMBER.values
        if len(argo_list) > 0:
            argo_list = [each for each in argo_files if any(str(pn) in each for pn in argo_list)]

            # Open each float and select the profiles
            for file in argo_list:

                with xr.open_dataset(file) as ds:
                    varsl = [var for var in vars if var in list(ds.keys())]

                    # apply flags
                    for var in varsl:
                        ds = apply_QC_flags_argo(ds, var)

                    date = ds.JULD.values
                    date = date.astype('datetime64[Y]').astype(float)+1970. + (date - date.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) / 365.
                    ds['DATE'] = xr.DataArray(date, dims=['N_PROF'], coords=ds.LATITUDE.coords)

                    bytes_vars = [var for var in ds.data_vars if ds[var].dtype == 'O']
                    ds = ds.drop_vars(bytes_vars)
                    mask = (ds.LATITUDE < latbin+lat_grid/2) & (ds.LATITUDE > latbin-lat_grid/2) & (ds.LONGITUDE < lonbin+lon_grid/2) & (ds.LONGITUDE > lonbin-lon_grid/2) 
                    mask = mask & (ds.PRES_ADJUSTED > ds.MLD)

                    if mask.any():  # need this otherwise it crashed when the mask is all False
                        ds = ds.where(mask, drop=True)

                        # add the data to the dataframe
                        for iprof in range(len(ds.N_PROF)):
                            dsl = ds.isel(N_PROF=iprof)
                            toappend = {}
                            lats = dsl['LATITUDE'].broadcast_like(dsl['PRES_ADJUSTED']).values
                            toappend['LATITUDE'] = lats
                            lons = dsl['LONGITUDE'].broadcast_like(dsl['PRES_ADJUSTED']).values
                            toappend['LONGITUDE'] = lons
                            for var in varsl:
                                if var == 'DIC':
                                    data = dsl[var+'_ESPER_MX'].values.flatten()
                                else:
                                    data = dsl[var].values.flatten()
                                data = dsl[var].values.flatten()
                                toappend[var] = data
                            sig0 = dsl.sigma0.values.flatten()
                            gamma = dsl.gamma.values.flatten()
                            depth = dsl.depth.values.flatten()
                            toappend['sigma0'] = sig0
                            toappend['gamma'] = gamma
                            time = dsl.DATE.values.flatten()
                            toappend['datetime'] = time
                            toappend['Depth'] = depth

                            df_argo = pd.concat([df_argo, pd.DataFrame(toappend)])

    # Get GLODAP data
    if (dataset_name == 'all') | (dataset_name == 'glodap') :
        match = (np.logical_and(np.logical_and(gdap.LATITUDE.values>latbin,
                                                gdap.LATITUDE.values<=latbin+lat_grid),
                                np.logical_and(gdap.LONGITUDE.values>lonbin,
                                                gdap.LONGITUDE.values<=lonbin+lon_grid)) )

        gdap_match = gdap[match]
        gdap_match = gdap_match.sort_values(by='datetime', ascending=True)

        # Split per station
        gdap_keep = []
        gdap_match_cruise = np.unique(gdap_match.G2cruise)
        for cruise_index, cr in enumerate(gdap_match_cruise):
            gdap_cruise = gdap_match[gdap_match.G2cruise==gdap_match_cruise[cruise_index]]
            station_list = np.unique(gdap_cruise.G2station)
            for stn_idx, st in enumerate(station_list):
                gdap_stn = gdap_cruise[gdap_cruise.G2station==station_list[stn_idx]]
                cast_list = np.unique(gdap_stn.G2cast)
                for cast_idx, cast in enumerate(cast_list):
                    gdap_cast = gdap_stn[gdap_stn.G2cast==cast_list[cast_idx]]

                    if (len(gdap_cast)>2):

                        # remove data above MLD
                        idx_ores_order = np.argsort(gdap_cast.PRES_ADJUSTED.values)
                        mld = mld_dbm_v3(gdap_cast.TEMP_ADJUSTED.values[idx_ores_order], gdap_cast.PSAL_ADJUSTED.values[idx_ores_order], gdap_cast.PRES_ADJUSTED.values[idx_ores_order], 0.03)

                        if np.isnan(mld):
                            mld = 20

                        mask_mld = gdap_cast.PRES_ADJUSTED > mld
                        gdap_masked = gdap_cast.copy()
                        gdap_masked = gdap_masked[mask_mld]
                        if len(gdap_masked) > 0:
                            gdap_keep.append(gdap_masked)
        if len(gdap_keep) > 0: 
            gdap_keep = pd.concat(gdap_keep, ignore_index=True)
        else:
            gdap_keep = gdap_match.iloc[0:0].copy()

    # get the WOD data that was not in GLODAP and Argo
    if (dataset_name == 'all') | (dataset_name == 'wod') :
        match = (np.logical_and(np.logical_and(df_wod.LATITUDE.values>latbin,
                                                df_wod.LATITUDE.values<=latbin+lat_grid),
                                np.logical_and(df_wod.LONGITUDE.values>lonbin,
                                                df_wod.LONGITUDE.values<=lonbin+lon_grid)) )
        wod_match = df_wod[match]
        wod_match = wod_match.sort_values(by='datetime', ascending=True)

        # remove data above MLD
        wod_keep = []
        wod_match['profile_id'] = wod_match.groupby(['LATITUDE', 'LONGITUDE', 'datetime']).ngroup()
        for prof_id in wod_match['profile_id'].unique() :
            profile = wod_match[wod_match['profile_id'] == prof_id].sort_values(by='Depth')

            mld = mld_dbm_v3(profile.TEMP_ADJUSTED.values, profile.PSAL_ADJUSTED.values, profile.PRES_ADJUSTED.values, 0.03)

            mask_mld = profile.Depth > mld
            wod_keep.append( profile[mask_mld] )

        if len(wod_keep)>0:
            wod_keep = pd.concat(wod_keep, ignore_index=True)
        else:
            wod_keep = wod_match.iloc[0:0].copy()

        # targeting density not implemented

    # ------ SAVE ------
    # Save intermediate product: all data at this grid point
    if dataset_name == 'all':
        gdap_keep['DATASET'] = 'GLODAP'
        df_argo['DATASET'] = 'Argo'
        wod_keep['DATASET'] = 'WOD'
        df_combined = pd.concat([gdap_keep, df_argo, wod_keep], axis=0)
    elif dataset_name == 'glodap':
        gdap_keep['DATASET'] = 'GLODAP'
        df_combined = gdap_keep
    elif dataset_name == 'argo':
        df_argo['DATASET'] = 'Argo'
        df_combined = df_argo
    elif dataset_name == 'wod':
        wod_keep['DATASET'] = 'WOD'
        df_combined = wod_keep

    # remove anything above the seasonal MLD
    if (lonbin > mldmax.lon[0]) & (lonbin < mldmax.lon[-1]) & (latbin > mldmax.lat[0]) & (latbin < mldmax.lat[-1]):
        mask_mld = df_combined.PRES_ADJUSTED < float(mldmax.sel(lon=lonbin, lat=latbin))
        df_combined.loc[mask_mld, ['TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'DOXY_ADJUSTED', 'NITRATE_ADJUSTED', 'PHOSPHATE_ADJUSTED', 'DIC', 'PH_IN_SITU_TOTAL_ADJUSTED']] = np.nan

    folder = 'outputs/temporary/%.1f_%.1f/'%(lon_grid,lat_grid)
    if not os.path.exists(folder):
        os.makedirs(folder)
    df_combined.to_csv(folder+'dataset_%s_%.1f_%.1f.csv'%(dataset_name,lonbin,latbin))

# -----------

if get_data:

    # load the seasonal MLD
    mldmax = xr.open_dataset('PATH_TO_FILE/Maximum_seasonal_MLD.nc').mld_max
    lon_vec = np.array(lon_vec) ; lat_vec = np.array(lat_vec)
    mldmax = mldmax.interp(lon=lon_vec[(lon_vec >= float(mldmax.lon[0])) & (lon_vec <= float(mldmax.lon[-1]))], lat=lat_vec[(lat_vec >= float(mldmax.lat[0])) & (lat_vec <= float(mldmax.lat[-1]))])

    if (dataset_name == 'all') | (dataset_name == 'glodap') :
        print('Load GLODAP')

        if 'gdap' not in locals() : # if gdap data not loaded yet
            gdap = pd.read_csv(data_dir_glo+'GLODAPv2.2023_Merged_Master_File.csv', dtype=dtype)
            df = pd.DataFrame({'year': gdap['G2year'],
                               'month': gdap['G2month'],
                               'day': gdap['G2day'],
                               'hour': gdap['G2hour'],
                               'minute': gdap['G2minute']})
            dates = pd.to_datetime(df)
            deci_year = [date.year + ( (date - datetime(date.year, 1, 1)).total_seconds() / (datetime(date.year + 1, 1, 1) - datetime(date.year, 1, 1)).total_seconds() ) for date in dates]
            gdap['datetime'] = deci_year
            # drop unused data
            gdap = gdap.drop(columns=['G2expocode', 'G2region', 'G2year', 'G2month', 'G2day', 'G2hour', 'G2minute', 'G2bottomdepth', 'G2maxsampdepth', 'G2bottle', 'G2sigma1', 'G2sigma2', 'G2sigma3', 'G2sigma4', 'G2aou', 'G2aouf', 'G2nitrite', 'G2nitritef', 'G2silicate', 'G2silicatef', 'G2silicateqc', 'G2fco2', 'G2fco2f', 'G2fco2temp', 'G2cfc11', 'G2pcfc11', 'G2cfc11f', 'G2cfc11qc', 'G2cfc12', 'G2pcfc12', 'G2cfc12f', 'G2cfc12qc', 'G2cfc113', 'G2pcfc113', 'G2cfc113f', 'G2cfc113qc', 'G2ccl4', 'G2pccl4', 'G2ccl4f', 'G2ccl4qc', 'G2sf6', 'G2psf6', 'G2sf6f', 'G2sf6qc', 'G2c13', 'G2c13f', 'G2c13qc', 'G2c14', 'G2c14f', 'G2c14err', 'G2h3', 'G2h3f', 'G2h3err', 'G2he3', 'G2he3f', 'G2he3err', 'G2he', 'G2hef', 'G2heerr', 'G2neon', 'G2neonf', 'G2neonerr', 'G2o18', 'G2o18f', 'G2toc', 'G2tocf', 'G2doc', 'G2docf', 'G2don', 'G2donf', 'G2tdn', 'G2tdnf', 'G2chla', 'G2chlaf', 'G2doi',
            'G2talk', 'G2talkf', 'G2talkqc', 'G2phts25p0', 'G2phts25p0f', 'G2theta'])
            del df

            # apply flags
            flagvars = ['G2salinity', 'G2oxygen', 'G2nitrate', 'G2tco2', 'G2phtsinsitutp', 'G2phosphate']
            for v in flagvars:
                flag = v+'f'
                naninds = gdap[flag]!=2
                gdap[v][naninds] = np.nan
            # drop flags
            gdap = gdap.drop(columns=['G2salinityqc', 'G2oxygenqc', 'G2nitrateqc', 'G2tco2qc', 'G2phtsqc', 'G2salinityf', 'G2oxygenf', 'G2nitratef', 'G2tco2f', 'G2phtsinsitutpf', 'G2phosphatef', 'G2phosphateqc'])

            #rename GLODAP comparison variables to match argo
            gdap = gdap.rename(columns={'G2longitude':'LONGITUDE', 'G2latitude':'LATITUDE', 'G2pressure':'PRES_ADJUSTED',
                                        'G2temperature':'TEMP_ADJUSTED','G2salinity':'PSAL_ADJUSTED',
                                        'G2oxygen':'DOXY_ADJUSTED','G2nitrate':'NITRATE_ADJUSTED', 'G2tco2':'DIC',
                                        'G2phosphate':'PHOSPHATE_ADJUSTED',
                                        'G2phtsinsitutp':'PH_IN_SITU_TOTAL_ADJUSTED','G2sigma0':'sigma0', 'G2gamma':'gamma',
                                        'G2depth':'Depth'})

            print('GLODAP KEYS:', list(gdap.keys()))

    # Prepare Argo
    argo_files =  os.listdir(data_dir_argo)
    argo_files = [data_dir_argo+each for each in argo_files]

    # Get the location info for each argo file
    if os.path.exists('outputs/float_info.csv'):
        files = np.sort([each for each in os.listdir(data_dir_argo) if 'Sprof.nc' in each])

        df_ai = {'PLATFORM_NUMBER':[], 'DATE_0':[], 'DATE_1':[], 'MAXLAT':[], 'MINLAT':[], 'MAXLON':[], 'MINLON':[]}
        for file in files:
            print(file)
            ds = xr.open_dataset(data_dir_argo+file)
            df_ai['PLATFORM_NUMBER'].append( int(ds.PLATFORM_NUMBER[0].values) )
            date = ds.JULD.values
            date = np.array([date[0], date[-1]])
            date = date.astype('datetime64[Y]').astype(float)+1970. + (date - date.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) / 365.
            df_ai['DATE_0'].append(date[0])
            df_ai['DATE_1'].append(date[1])
            lats = ds.LATITUDE
            lats[lats < -80] = np.nan # replace bad values with nans
            maxlat = np.nanmax(lats)
            minlat = np.nanmin(lats)
            lons = ds.LONGITUDE
            lons[lons < -200] = np.nan
            maxlon = np.nanmax(lons)
            minlon = np.nanmin(lons)

            df_ai['MAXLAT'].append(maxlat)
            df_ai['MINLAT'].append(minlat)
            df_ai['MAXLON'].append(maxlon)
            df_ai['MINLON'].append(minlon)

        df_ai = pd.DataFrame.from_dict(df_ai)
        df_ai.to_csv('outputs/float_info.csv', index=False)
        del df_ai
    argo_info = pd.read_csv('outputs/float_info.csv')

    if (dataset_name == 'all') | (dataset_name == 'wod'):
        # load
        print('Load WOD')
        path = "../../../../Datasets/Data_Products/WOD/Query_oxygen_2025/NotInGLODAP/"
        files = os.listdir(path)
        df_wod = []
        for file in files:
            if '.nc' not in file:
                df_wod.append(pd.read_csv(path+file))
        df_wod = pd.concat(df_wod, ignore_index=True)

        # calculate missing pressures
        df_wod.loc[df_wod['press'].isna(), 'press'] = sw.pres( df_wod.loc[df_wod['press'].isna()].depths, df_wod[df_wod['press'].isna()].lats )

        # calculate gamman
        gamman = []
        for i in range(len(df_wod)):
            if ~df_wod.loc[i, ['sals', 'temps', 'press', 'lons', 'lats']].isna().any():
                gamman.append( nds.gamma_n(df_wod['sals'][i], df_wod['temps'][i], df_wod['press'][i], 1, df_wod['lons'][i], df_wod['lats'][i])[0][0] )
            else:
                gamman.append( np.nan )
        df_wod['gamma'] = gamman

        # convert time
        dates = pd.to_datetime(df_wod['datetimes'])
        df_wod['datetimes'] = [date.year + ( (date - datetime(date.year, 1, 1)).total_seconds() / (datetime(date.year + 1, 1, 1) - datetime(date.year, 1, 1)).total_seconds() ) for date in dates]

        df_wod = df_wod.rename(columns={'lons':'LONGITUDE', 'lats':'LATITUDE', 'press':'PRES_ADJUSTED',
                                    'temps':'TEMP_ADJUSTED','sals':'PSAL_ADJUSTED',
                                    'oxys':'DOXY_ADJUSTED','nitrates':'NITRATE_ADJUSTED', 'phosphates':'PHOSPHATE_ADJUSTED',
                                    'phs':'PH_IN_SITU_TOTAL_ADJUSTED', 'depths':'Depth', 'datetimes':'datetime'})
        print('WOD KEYS:', list(df_wod.keys()))

    # ------
    # GET THE DATA IN EACH GRID POINTS AND DENSITY RANGE
    # ------

    Parallel(n_jobs=n_jobs)(delayed(extract_data)(lonbin, latbin) for lonbin in lon_vec for latbin in lat_vec)


# ------
# CALCULATE TRENDS
# ------

def compute_bin(lonbin, latbin):
    """
    Function to compute the trends in each grid cell

    Parameters:
    lonbin: longitude of the grid cell
    latbin: latitude of the grid cell

    Returns: ilon, ilat, trendsl, values_refl
    ilon: index of longitude grid cell
    ilat: index of latitude grid cell
    trendsl: dictionary of trend for all variables
    values_refl: reference value at that grid cell, from which the anomalies are computed
    Also save netcdf file with anomalies at that grid cell. Used to plot time series
    """

    ilon = lon_vec.index(lonbin)
    ilat = lat_vec.index(latbin)
    print(lonbin, latbin)

    trendsl = {var: np.zeros(len(gamma_range))*np.nan
        for var in [
            'TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'DOXY_ADJUSTED', 'NITRATE_ADJUSTED', 'PHOSPHATE_ADJUSTED', 'DIC', 'Mean_pres', 'Min_pres', 'Max_pres', 'PRES_ADJUSTED', 'PH_IN_SITU_TOTAL_ADJUSTED',
            'TEMP_ADJUSTED_SIGNIFICANCE', 'PSAL_ADJUSTED_SIGNIFICANCE', 'DOXY_ADJUSTED_SIGNIFICANCE', 'NITRATE_ADJUSTED_SIGNIFICANCE', 'PHOSPHATE_ADJUSTED_SIGNIFICANCE',
            'DIC_SIGNIFICANCE', 'PRES_ADJUSTED_SIGNIFICANCE', 'PH_IN_SITU_TOTAL_ADJUSTED_SIGNIFICANCE',
            'TEMP_ADJUSTED_std_error', 'PSAL_ADJUSTED_std_error', 'DOXY_ADJUSTED_std_error', 'NITRATE_ADJUSTED_std_error', 'DIC_std_error', 'PHOSPHATE_ADJUSTED_std_error',
            'PRES_ADJUSTED_std_error', 'PH_IN_SITU_TOTAL_ADJUSTED_std_error'
        ]}
#    values_1990l = {var: np.zeros(len(gamma_range))*np.nan for var in ['TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'DOXY_ADJUSTED', 'NITRATE_ADJUSTED', 'DIC', 'Mean_pres', 'PRES_ADJUSTED', 'PH_IN_SITU_TOTAL_ADJUSTED']}
    values_refl = {var: np.zeros(len(gamma_range))*np.nan for var in ['TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'DOXY_ADJUSTED', 'NITRATE_ADJUSTED', 'DIC', 'PHOSPHATE_ADJUSTED', 'Mean_pres', 'PRES_ADJUSTED', 'PH_IN_SITU_TOTAL_ADJUSTED']}

    # Load
    df_combined = pd.read_csv('outputs/temporary/%.1f_%.1f/dataset_%s_%.1f_%.1f.csv'%(lon_grid,lat_grid,dataset_name,lonbin,latbin))
    if len(df_combined):

        # remove negative dates
        df_combined.loc[df_combined['datetime'] < 0, 'datetime'] = np.nan
        df_combined.loc[df_combined['sigma0'] < 0, 'sigma0'] = np.nan

        # focus on the wanted period
        if period == '_since2000':
            df_combined = df_combined[df_combined['datetime'] >= 2000]

        # for pressure, get a fit to give value on center of density bin
        mask = ~np.isnan(df_combined.PRES_ADJUSTED) & ~np.isnan(df_combined.sigma0)
        if np.sum(mask)>5:
            pres_fit = np.poly1d( np.polyfit(df_combined.sigma0[mask], df_combined.PRES_ADJUSTED[mask], deg=2) )
            target_sigma = (gamma_range[1:]+gamma_range[:-1])/2
            closest_sigma = np.array([target_sigma[np.argmin(np.abs(target_sigma-s))] for s in df_combined.gamma])
            fitted_pres_at_target = pres_fit(closest_sigma)
            correction = np.where(mask, df_combined.PRES_ADJUSTED - pres_fit(df_combined.gamma), np.nan)
            df_combined['PRES_ADJUSTED'] = np.where(mask, fitted_pres_at_target + correction, np.nan)

        # initiate dataarray to save time series of anomalies
        dfanom = { var: pd.DataFrame(np.nan, index=gamma_range, columns=time_bins[:-1]) for var in vars }
        dfanom = dfanom | { var+'_std': pd.DataFrame(np.nan, index=gamma_range, columns=time_bins[:-1]) for var in vars }
        # for each density layer...
        for i in range(len(gamma_range)):
            sig = gamma_range[i]

            data_sig = df_combined[(df_combined.gamma>sig-0.05) & (df_combined.gamma<=sig+0.05)]

            if len(data_sig) > 0:
                time = data_sig.datetime
                for var in varlist:
                    datal = data_sig[var]
                    timel = time[~np.isnan(datal)]
                    dataset = data_sig['DATASET'][~np.isnan(datal)]
                    datal = datal[~np.isnan(datal)]
                    if (len(timel) > 0) :
                        # Apply 2 conditions:
                        # 1) if period _sincepriorXXXX, time series need to include data < XXXX
                        period_condition_met = False
                        if (period == '') | (period == '_since2000') :
                            period_condition_met = True
                        elif '_sinceprior' in period:
                            yrcond = int(period.replace('_sinceprior',''))
                            if min(timel) < yrcond:
                                period_condition_met = True
                        # 2) minimum of X years of data coverage to calculate a trend
                        if (max(timel) - min(timel) > 15) & period_condition_met :

                            # Store GLODAP & WOD data
                            glodap = data_sig[data_sig['DATASET'] == 'GLODAP']
                            datal_glodap = glodap[var].values
                            timel_glodap = glodap.datetime.values
                            timel_glodap = timel_glodap[~np.isnan(datal_glodap)]
                            datal_glodap = datal_glodap[~np.isnan(datal_glodap)]
                            wod = data_sig[data_sig['DATASET'] == 'WOD']
                            datal_wod = wod[var].values
                            timel_wod = wod.datetime.values
                            timel_wod = timel_wod[~np.isnan(datal_wod)]
                            datal_wod = datal_wod[~np.isnan(datal_wod)]

                            # Identify Argo data
                            argo = data_sig[data_sig['DATASET'] == 'Argo']
                            argo = argo[argo[var].notna()] # filter out nans

                            # Group per month
                            datetime_time = pd.to_datetime((argo.datetime - 1970) * 365.25, origin=f'{1970}-01-01', unit='D')
                            argo.loc[:,'period'] = datetime_time.dt.to_period('M')
                            df_mean = argo.groupby('period')[var].mean().reset_index()
                            timel_argo = [each.year+(each.month-1+0.5)/12 for each in df_mean.period.values]

                            datal_argo = df_mean[var].values

                            # Combine
                            timel = np.array( list(timel_glodap)+list(timel_wod)+list(timel_argo) )
                            datal = np.array( list(datal_glodap)+list(datal_wod)+list(datal_argo) )

                            # remove outliers
                            if var != 'PRES_ADJUSTED':
                                datal[datal > 3000] = np.nan
                                datal[datal < -3000] = np.nan

                            # calculate trend
                            fittype, coefs, coefs2, slope_error, [x_line, y_line_low, y_line_high] = fit_trend(timel, datal, plot_local=False, verbose=False)
                            if fittype == None:
                                signi = 0
                            elif fittype == 'Poly':
                                signi = 2
                            else:
                                signi = 1

                            y_est = coefs[1]+ np.array(timel)*coefs[0]

                            # save trend
                            trendsl[var][i] = coefs[0]
                            trendsl[var+'_SIGNIFICANCE'][i] = signi
                            trendsl[var+'_std_error'][i] = slope_error

                            # --- Time series of yearly anomalies relative to 1990
                            value_1990 = coefs[0]*1990 + coefs[1]

                            try:
                                value_ref = clim[var].sel(lon=lonbin, lat=latbin, gamma=sig, method='nearest').values
                            except:
                                value_ref = value_1990 # for pressure
                            values_refl[var] = value_ref

                            # remove outliers before computing the mean
                            dum = datal - value_ref##value_1990
                            # bin yearly
                            sum, _ = np.histogram(timel, bins=time_bins, weights=datal)
                            count, _ = np.histogram(timel, bins=time_bins)
                            yearly_mean = np.divide( sum, count )
                            sum2, _ = np.histogram(timel, bins=time_bins, weights=np.array(datal)**2)
                            std = np.sqrt( np.divide(sum2, count) - yearly_mean**2 )
                            std[std<1e-5] = np.nan # replace with nan where only one value per year

                            dfanom[var].iloc[i,:] = yearly_mean - value_ref#value_1990
                            dfanom[var+'_std'].iloc[i,:] = std

            # calculate mean pressure - for section plots
            trendsl['Mean_pres'][i] = np.nanmean(data_sig.PRES_ADJUSTED)
            trendsl['Min_pres'][i] = data_sig.PRES_ADJUSTED.min()
            trendsl['Max_pres'][i] = data_sig.PRES_ADJUSTED.max()

        # save anomaly time series
        dsanom = xr.Dataset({var: (['gamma', 'year'], df.values) for var, df in dfanom.items()}, coords={'gamma':gamma_range, 'year':time_bins[:-1]})
        dsanom.to_netcdf('outputs/temporary/%.1f_%.1f/anom/anom_time_series_%.1f_%.1f.nc'%(lon_grid,lat_grid,lonbin,latbin))

    return ilon, ilat, trendsl, values_refl

# ---------------------------------

# Calculate trends per grid cell

varlist = ['TEMP_ADJUSTED', 'DOXY_ADJUSTED', 'PSAL_ADJUSTED', 'NITRATE_ADJUSTED', 'DIC', 'PRES_ADJUSTED', 'PH_IN_SITU_TOTAL_ADJUSTED', 'PHOSPHATE_ADJUSTED']
if calc_trends:

    # load climatologies of density
    clim = xr.open_dataset('PATH_TO_FILE/Climatologies_gamma_2.5.nc')

    s = ((len(lon_vec), len(lat_vec), len(gamma_range)))
    trends = {'TEMP_ADJUSTED':np.zeros(s)*np.nan, 'PSAL_ADJUSTED':np.zeros(s)*np.nan, 'DOXY_ADJUSTED':np.zeros(s)*np.nan, 'NITRATE_ADJUSTED':np.zeros(s)*np.nan, 'PHOSPHATE_ADJUSTED':np.zeros(s)*np.nan, 'DIC':np.zeros(s)*np.nan, 'Mean_pres':np.zeros(s)*np.nan, 'Min_pres':np.zeros(s)*np.nan, 'Max_pres':np.zeros(s)*np.nan, 'PRES_ADJUSTED':np.zeros(s)*np.nan, 'PH_IN_SITU_TOTAL_ADJUSTED':np.zeros(s)*np.nan,
                'TEMP_ADJUSTED_SIGNIFICANCE':np.zeros(s), 'PSAL_ADJUSTED_SIGNIFICANCE':np.zeros(s), 'DOXY_ADJUSTED_SIGNIFICANCE':np.zeros(s), 'NITRATE_ADJUSTED_SIGNIFICANCE':np.zeros(s), 'PHOSPHATE_ADJUSTED_SIGNIFICANCE':np.zeros(s), 'DIC_SIGNIFICANCE':np.zeros(s), 'PRES_ADJUSTED_SIGNIFICANCE':np.zeros(s), 'PH_IN_SITU_TOTAL_ADJUSTED_SIGNIFICANCE':np.zeros(s),
                'TEMP_ADJUSTED_std_error':np.zeros(s), 'PSAL_ADJUSTED_std_error':np.zeros(s), 'DOXY_ADJUSTED_std_error':np.zeros(s), 'NITRATE_ADJUSTED_std_error':np.zeros(s), 'PHOSPHATE_ADJUSTED_std_error':np.zeros(s), 'DIC_std_error':np.zeros(s), 'PRES_ADJUSTED_std_error':np.zeros(s), 'PH_IN_SITU_TOTAL_ADJUSTED_std_error':np.zeros(s)}

    values_1990 = {'TEMP_ADJUSTED':np.zeros(s)*np.nan, 'PSAL_ADJUSTED':np.zeros(s)*np.nan, 'DOXY_ADJUSTED':np.zeros(s)*np.nan, 'NITRATE_ADJUSTED':np.zeros(s)*np.nan, 'PHOSPHATE_ADJUSTED':np.zeros(s)*np.nan, 'DIC':np.zeros(s)*np.nan, 'Mean_pres':np.zeros(s)*np.nan, 'PRES_ADJUSTED':np.zeros(s)*np.nan, 'PH_IN_SITU_TOTAL_ADJUSTED':np.zeros(s)*np.nan}

    results = Parallel(n_jobs=n_jobs)(delayed(compute_bin)(lonbin, latbin) for lonbin in lon_vec for latbin in lat_vec)
    for each in results:
        for key in trends:
            trends[key][each[0], each[1], :] = each[2][key]
            if key in list(values_1990.keys()):
                values_1990[key][each[0], each[1], :] = each[3][key]

    # Save to a netcdf
    # remove outliers
    trends['Mean_pres'] = np.where( (trends['Mean_pres'] >= 0) & (trends['Mean_pres'] <= 5000), trends['Mean_pres'], np.nan)
    trends['Min_pres'] = np.where( (trends['Min_pres'] >= 0) & (trends['Min_pres'] <= 5000), trends['Min_pres'], np.nan)
    trends['Max_pres'] = np.where( (trends['Max_pres'] >= 0) & (trends['Max_pres'] <= 5000), trends['Max_pres'], np.nan)

    # Save trends
    print('Saving netcdf')
    dims = ('lon', 'lat', 'gamma')
    coords = {'lon':lon_vec, 'lat':lat_vec, 'gamma':gamma_range}
    dstrends = xr.Dataset({key: (dims, value) for key, value in trends.items()}, coords=coords)
    file = 'outputs/trends_on_grid_%s%s.nc'%(dataset_name,period)
    if os.path.exists(file):
        os.remove(file)
    dstrends.to_netcdf(file)

    # Save reference value
    ds1990 = xr.Dataset({key: (dims, value) for key, value in values_1990.items()}, coords=coords)
    file = 'outputs/value1990_on_grid_%s%s.nc'%(dataset_name,period)
    if os.path.exists(file):
        os.remove(file)
    ds1990.to_netcdf(file)


# ------------------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------------------

if plot_trends:
    print(' ')
    print('Plot maps')

    varname = {'DOXY_ADJUSTED':'Oxygen', 'NITRATE_ADJUSTED':'Nitrate', 'PHOSPHATE_ADJUSTED':'Phosphate', 'DIC':'DIC', 'PSAL_ADJUSTED':'Salinity', 'TEMP_ADJUSTED':'Temperature', 'PH_IN_SITU_TOTAL_ADJUSTED':'pH', 'PRES_ADJUSTED':'Pressure', 'spiciness0':'Spiciness'}
    units = {'DOXY_ADJUSTED':'$\mu$mol kg$^{-1}$ yr$^{-1}$', 'NITRATE_ADJUSTED':'$\mu$mol kg$^{-1}$ yr$^{-1}$', 'PHOSPHATE_ADJUSTED':'$\mu$mol kg$^{-1}$ yr$^{-1}$', 'DIC':'$\mu$mol kg$^{-1}$ yr$^{-1}$', 'TEMP_ADJUSTED':'$^\circ$C yr$^{-1}$', 'PSAL_ADJUSTED':'yr$^{-1}$', 'PH_IN_SITU_TOTAL_ADJUSTED':'yr$^{-1}$', 'PRES_ADJUSTED':'dbar yr$^{-1}$', 'spiciness0':''}

    # prepare modified cmap
    cmap = plt.get_cmap('RdYlBu', 256)
    colors = cmap(np.linspace(0,1,256))

    # --- COMPLETE PLOT ---

    n = 5 # number of subplots in each direction
    size = (35,20)
    file = 'outputs/trends_on_grid_%s%s.nc'%(dataset_name,period)
    s = 5

    dstrends = xr.open_dataset(file)

    # plot from 25.0 to 28.0
    for var in vars:
        print(var)

        # custom cmap
        values = np.linspace(-lims[var],lims[var],len(colors))
        colors_custom = colors.copy()
        colors_custom[ (values<=lims[var]/10) & (values>=-lims[var]/10) ] = np.array([0.7, 0.7, 0.7, 1.0])
        custom_cmap = mcolors.ListedColormap(colors_custom)

        fig = plt.figure(figsize=size)
        c=0
        for dens in gamma_range:

            dsl = dstrends.sel(gamma=dens, method='nearest')

            ax = plt.subplot(n,n,c+1, projection=ccrs.Robinson(central_longitude=-160))
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.set_global()
            gl = ax.gridlines(draw_labels=True)
            gl.right_labels = False
            gl.bottom_labels = False

            X, Y = np.meshgrid(dsl.lon, dsl.lat)

            sct = ax.scatter(X, Y, s=s, c=dsl[var].T, transform=ccrs.PlateCarree(), vmin=-lims[var] , vmax=lims[var], cmap=custom_cmap)

            # where trend significant
            signi = dsl[var+'_SIGNIFICANCE'].where(dsl[var+'_SIGNIFICANCE']==1).T.values.flatten()
            Xflat = X.flatten()
            Yflat = Y.flatten()
            ax.scatter(Xflat[~np.isnan(signi)], Yflat[~np.isnan(signi)], s=s+5, marker='o', color='k', facecolor='none', transform=ccrs.PlateCarree(), linewidth=0.5)

            # where variability in the trend
            signi = dsl[var+'_SIGNIFICANCE'].where(dsl[var+'_SIGNIFICANCE']==2).T.values.flatten()
            Xflat = X.flatten()
            Yflat = Y.flatten()
            ax.scatter(Xflat[~np.isnan(signi)], Yflat[~np.isnan(signi)], s=s+5, marker='o', color='dimgrey', facecolor='none', transform=ccrs.PlateCarree(), linewidth=0.5)

            ax.set_title(np.round(dens,1))

            c+=1

        fig.subplots_adjust(right=0.9, hspace=0.15, wspace=0.1, left=0.02, top=0.95, bottom=0.01)
        cbar_ax = fig.add_axes([0.93, 0.25, 0.01, 0.5])
        fig.colorbar(sct, cax=cbar_ax, label='%s $\mu$mol kg$^{-1}$ yr$^{-1}$'%var)

        plt.savefig('PATH_TO_FIGS/trends_on_grid_%s%s_%s.png'%(dataset_name,period,var), dpi=300)
        plt.clf()
