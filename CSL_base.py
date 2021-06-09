#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------#
#                                                                                    #
#          SCRIPT for basic functions to calculate statistics and draw plots         #
#                                                                                    #
#                                                               BY   : KM.Noh        #
#                                                               DATE : 2021.06.09    #
#                                                                                    #
#------------------------------------------------------------------------------------#

## Modules for Calculate netCDF 
import numpy    as np
import xarray   as xr
import netCDF4 
import pandas as pd

## Modules for caculating statistics
from scipy   import stats, signal
from sklearn import linear_model

## Modules for plottings
import matplotlib.pyplot as plt 
import matplotlib.colorbar as cb
from mpl_toolkits.basemap import Basemap, shiftgrid



## Function for reading latlon 3D data [time, lat, lon]
def read_var_latlon_loc(path_name,var_name,yr_str,yr_end,latS,latN,lonL,lonR):
    ## Open the netCDF4 file
    f    = xr.open_dataset(path_name)
    
    ## Read the Coordinates
    time       = f["time"]
    lat        = f["lat"] 
    lon        = f["lon"]
    
    ## Load 'var_name' data
    var_model  = f[var_name]
        
    ## Find indexes for time, lat, lon ranges
    time_year = f['time'].dt.year

    ind_year  = np.where((time_year>=yr_str) * (time_year<=yr_end))
    ind_lat   = np.where((lat>=latS) * (lat<=latN))
    ind_lon   = np.where((lon>=lonL) * (lon<=lonR))

    var_model_loc = var_model[ind_year[0],ind_lat[0],ind_lon[0]]
    
    return var_model_loc



## Function for reading station 3D data [time, lat, lon]
def read_var_station_loc(path_name,var_name,yr_str,yr_end):
    ## Open the netCDF4 file
    f    = xr.open_dataset(path_name)
    
    ## Read the Coordinates
    time       = f["time"]
    
    ## Load 'var_name' data
    var_model  = f[var_name]
        
    ## Find indexes for time
    time_year = f['time'].dt.year

    ind_year  = np.where((time_year>=yr_str) * (time_year<=yr_end))
    var_model_loc = var_model[ind_year[0],:]
    
    return var_model_loc



### Function for calculating correaltion & p-value with nan values
def calc_corr_pval_nan(x_array,y_array):
    ma_x_array = np.ma.masked_invalid(x_array)
    ma_y_array = np.ma.masked_invalid(y_array)
    msk_x_y    = (~ma_x_array.mask & ~ma_y_array.mask)

    corr, pval  = stats.pearsonr(ma_x_array[msk_x_y],ma_y_array[msk_x_y])

    return corr, pval



### Function for calculating linear regression line
def calc_lin_regline(x_array,y_array):
    N_TS      = x_array.shape[0]
    N_nan     = np.where(np.isnan(x_array))[0].shape[0]
    ind_nnan  = np.where(~np.isnan(x_array))[0]

    reg = linear_model.LinearRegression()
    reg.fit(x_array[ind_nnan].values.reshape(-1,1),y_array[ind_nnan].values.reshape(-1,1))

    xp = np.linspace(np.min(x_array[ind_nnan]),np.max(x_array[ind_nnan]),100)
    yp = reg.coef_*xp + reg.intercept_

    return xp,yp



## Function for calculating anomalies
def calc_anomaly(var):
    clim_var = var.groupby("time.month").mean()
    anom_var = var.groupby("time.month") - clim_var
    
    return clim_var, anom_var



## Function for extracting climate indexes in specified regions
def extract_index(var,latS,latN,lonL,lonR):

    lat = var.lat 
    if (var.lat[0].values < 0): tmp_latN = latN ; tmp_latS = latS
    else:                       tmp_latN = latS ; tmp_latS = latN
        
    var_sel = var.sel(lat=slice(tmp_latS,tmp_latN),lon=slice(lonL,lonR))

    ## Calcuate the weighted mean of variable
    weights_lat      = np.cos(np.deg2rad(lat))
    weights_lat.name = "weights"  

    index_var = var_sel.weighted(weights_lat).mean(("lat","lon"))
    
    return index_var



## Functions for detrending 
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    da_detrended = detrend_dim(da_detrended, dims, deg=deg)
    return da_detrended



## Functions for calculating annual time series from month [DJF]
def mon_to_ann_DJF(ds):
    
    ds_DJF_all = ds.where(ds['time.season'] == 'DJF')
    # Rolling mean -> only Jan is not nan
    # However, we loose Jan/ Feb in the first year and Dec in the last
    ds_DJF = ds_DJF_all.rolling(min_periods=3, center=True, time=3).mean()
    # Exceptional Case for first year
    ds_DJF[0] = ds_DJF_all[0:2].mean()

    # make annual mean
    ds_DJF = ds_DJF.groupby('time.year').mean('time')
    
    return ds_DJF



## Functions for calculating annual time series in each season
def mon_to_ann_season(var,season):
    
    var_season_all = var.where(var['time.season'] == season)
    if (season=="DJF"):
        # Rolling mean -> only Jan is not nan
        # However, we loose Jan/ Feb in the first year and Dec in the last
        var_season    = var_season_all.rolling(min_periods=3, center=True, time=3).mean()
        # Exceptional Case for first year
        var_season[0] = var_season_all[0:2].mean()

        # make annual mean
        var_season = var_season.groupby('time.year').mean('time')
    else:
        var_season = var_season_all.groupby('time.year').mean('time')
        
    return var_season



## Functions for calculating correlation, regression, composite
def calc_corr(var,ts,dim):        return xr.corr(var,ts,dim=dim)
def calc_reg(var,ts,dim):         return xr.corr(var,ts,dim=dim) * var.std(dim=dim)
def calc_comp(var,ts,ind_ts,dim): return var[ind_ts,:].mean(dim=dim)



## Functions for basemap base in cyl projection
def basemap_cyl(contour, axes, lat, lon, lon0):
    
    m = Basemap(projection = 'cyl', 
                lon_0 = lon0, 
                resolution = 'c' ,
                ax = axes)

    m.fillcontinents(color = 'grey',lake_color = 'grey', alpha = 0.3)
    m.drawcoastlines(linewidth = 0.25)
    m.drawmapboundary(fill_color = 'white')

    contour_shifted, lon_shifted = shiftgrid(180+lon0, contour, lon, 
                                          start = False, cyclic=360)
    lon_new, lat_new = np.meshgrid(lon_shifted,lat)
    x, y = m(lon_new,lat_new)
    
    m.drawparallels(np.arange(-60.,61.,30.),
                    labels = [1,0,0,0], color = 'grey',linewidth=0.25)
    m.drawmeridians(np.arange(-180.,181.,60.),
                    labels = [0,0,0,1], color = 'grey',linewidth=0.25)
    
    return m, x, y, contour_shifted



## Functions for basemap base in South Korea
def basemap_korea(x, y, axes, latS, latN, lonL, lonR):
    
    m = Basemap(llcrnrlon=lonL, llcrnrlat=latS, 
                urcrnrlon=lonR, urcrnrlat=latN, 
                lat_0=(latS+latN)/2,     lon_0=(lonL+lonR)/2, 
                resolution='l',projection='lcc',epsg=4269, ax=axes)

    m.fillcontinents(color = 'grey',lake_color = 'grey', alpha = 0.05)
    m.drawcoastlines(linewidth = 0.35)
    m.drawmapboundary(fill_color = 'white')
    
    x, y = m(x.values,y.values)


    return m, x, y

