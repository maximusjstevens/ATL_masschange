#!/usr/bin/env python

'''
# Process_highrezGEOS.py

# Author: C. Max Stevens (@maximusjstevens)
# christopher.d.stevens-1@nasa.gov
#

Script to take the tiled, high-resolution outputs
from GEOS-LDAS and interpolate them onto the ICESat-2
grid.
'''

import numpy as np
import datetime
import xarray as xr
import pandas as pd
import calendar
import os
import sys
import glob
import time
import re
import dateutil.parser as dparser

from pathlib import Path

import geopandas as gpd
import rioxarray as rx               # Package to read raster data from hdf5 files
import rioxarray.merge
import pyproj #import Transformer, CRS, Proj  # libraries to allow coordinate transforms
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shapereader
# from shapely.geometry import LineString,Point,mapping, Polygon, box
import struct

import IS2view

from scipy.interpolate import griddata

def initialize_ds(y_g,x_g,_date_range,_vars):
    
    filler_array = (-9999*np.ones((len(y_g), len(x_g),len(_date_range)))).astype('float32')

    ds_ = xr.Dataset(
        data_vars=dict(
            dummy      = (["y","x","time"], filler_array),
        ),
    coords=dict(
        y    = (["y"], y_g),
        x    = (["x"], x_g),
        time = _date_range,
        reference_time = pd.Timestamp("1950-01-01"),
    ),
    attrs = dict(description="gridded GEOSLDAS ouputs",_FillValue=-9999))

    for _v in _vars:
        ds_[_v] = ds_['dummy'].copy()
    ds_ = ds_.drop_vars('dummy')

    return ds_

def grid_to_IS2(icesheet,YYYY,_ecode=None):

    if icesheet=='GrIS':
        _ecode = "EPSG:3413"
        atl15_10k = xr.open_dataset(f'/discover/nobackup/cdsteve2/IS2_data/ATL15/ATL15_GL_0318_10km_003_01.nc',group='delta_h').rio.write_crs(input_crs="EPSG:3413")
        for key in list(atl15_10k.keys()):
            atl15_10k[key].rio.write_nodata(np.nan, inplace=True)

    elif icesheet=='AIS':
        _ecode = "EPSG:3031"
        p_atl15 = Path('/discover/nobackup/cdsteve2/IS2_data/ATL15/')
        ds_l_10 = [IS2view.io.from_file(Path(p_atl15,f'ATL15_A{ii}_0324_10km_004_03.nc'),group='delta_h') for ii in [1,2,3,4]]
        atl15_10k = rioxarray.merge.merge_datasets(ds_l_10)
        for key in list(atl15_10k.keys()):
            atl15_10k[key].rio.write_nodata(np.nan, inplace=True)

    else:
        print('script presently configured only for AIS and GrIS. exiting.')
        sys.exit()
        ### specify "other" or something in the function call 
        ### and provide EPSG code; e.g. for Alaskan glaciers
        # if _ecode==None:
        #     print('EPSG not specified. Exiting.')
        #     sys.exit()
        # else:
        #     ### Need code here to build/import grid for other
        #     pass
        
    atlX,atlY = np.meshgrid(atl15_10k.x.values,atl15_10k.y.values)
    x_g = atl15_10k.x.values
    y_g = atl15_10k.y.values

    transformer = pyproj.Transformer.from_crs("EPSG:4326", _ecode)

    pHR = Path('/discover/nobackup/projects/gmao/polarm/lcandre2/outputs_long/LLI_M2_02_C1440_1980_1990/output/C1440x6C_GLOBAL/cat/ens0000')
    # for YYYY in np.arange(1981,1990):
    print(YYYY)
    out_path = Path('/discover/nobackup/cdsteve2/climate/LDAS_highres/LDAS_outputs')
    fn_out_yearly = f'LDAS_IS2_10km_daily_{YYYY}.nc'
    ds_monthly = []
    for MM in np.arange(1,13):
        _YM = f'{YYYY}-{MM}'
        _date_range = pd.date_range(_YM,pd.to_datetime(_YM) + pd.offsets.MonthEnd(n=0))

        fn_glc = f'LLI_M2_02_C1440_1980_1990.tavg24_1d_glc_Nt.{YYYY}{MM:02d}*_1200z.nc4'
        pF_glc = Path(pHR,f'Y{YYYY}/M{MM:02d}')
        flist_glc = sorted([_x for _x in pF_glc.glob(fn_glc)])

        fn_lfs = f'LLI_M2_02_C1440_1980_1990.tavg24_1d_lfs_Nt.{YYYY}{MM:02d}*_1200z.nc4'
        pF_lfs = Path(pHR,f'Y{YYYY}/M{MM:02d}')
        flist_lfs = sorted([_x for _x in pF_lfs.glob(fn_lfs)])

        ds_temp = atl15_10k.isel(time=0).copy()
        ###########################
        ### glc outputs
        for _ii, _fn in enumerate(flist_glc): # loop through all of the daily files for the month
            dt_string = re.search(r'\d{4}\d{2}\d{2}',str(_fn)).group()
            _timestamp = pd.to_datetime(datetime.datetime.strptime(dt_string, '%Y%m%d').date())
            with xr.open_dataset(_fn) as f_glc: 
                if _ii==0: 
                    glc_vars = [_x for _x in f_glc.variables if 'time' in f_glc[_x].dims]
                    ds_glc = initialize_ds(y_g,x_g,_date_range,glc_vars)
                
                df1 = f_glc.to_dataframe()
                ### need to configure below for regions
                df3 = df1[df1['lat']>58].copy()
                df3 = df3[((df3['lon']>-70) & (df3['lon']<-15))]
                ###
                df3['x_t'], df3['y_t'] = transformer.transform(df3['lat'].tolist(), df3['lon'].tolist())
                data_points = df3[['x_t','y_t']].values
                
                for _var in glc_vars:
                    gridded_data = griddata(data_points,df3[_var].values,(atlX,atlY),method='linear')            
                    ds_temp['dummy'] = (['y','x'],gridded_data)
                    ds_temp['dummy'] = ds_temp['dummy'].where(ds_temp['ice_area']>0)
                    ds_glc[_var].loc[{'time':_timestamp}] = ds_temp['dummy']
        ### end for glc
        ###########################                        

        ###########################
        ### lfs ouputs 
        for _ii, _fn in enumerate(flist_lfs): # loop through all of the daily files for the month
            dt_string = re.search(r'\d{4}\d{2}\d{2}',str(_fn)).group()
            _timestamp = pd.to_datetime(datetime.datetime.strptime(dt_string, '%Y%m%d').date())
            with xr.open_dataset(_fn) as f_lfs: 
                if _ii==0: 
                    lfs_vars = [_x for _x in f_lfs.variables if 'time' in f_lfs[_x].dims]
                    ds_lfs = initialize_ds(y_g,x_g,_date_range,lfs_vars)
                
                df1 = f_lfs.to_dataframe()
                ### need to configure below for regions
                df3 = df1[df1['lat']>58].copy()
                df3 = df3[((df3['lon']>-70) & (df3['lon']<-15))]
                ###
                df3['x_t'], df3['y_t'] = transformer.transform(df3['lat'].tolist(), df3['lon'].tolist())
                data_points = df3[['x_t','y_t']].values
                
                for _var in lfs_vars:
                    gridded_data = griddata(data_points,df3[_var].values,(atlX,atlY),method='linear')            
                    ds_temp['dummy'] = (['y','x'],gridded_data)
                    ds_temp['dummy'] = ds_temp['dummy'].where(ds_temp['ice_area']>0)
                    ds_lfs[_var].loc[{'time':_timestamp}] = ds_temp['dummy']
        ### end for lfs
        ###########################
        
        ds_combined = ds_lfs.merge(ds_glc)
        ds_monthly.append(ds_combined)
    ds_year = xr.concat(ds_monthly, dim='time').sortby('time')
    ds_year.to_netcdf(Path(out_path,fn_out_yearly))

    ### end month loop  

if __name__=='__main__':
    icesheet='GrIS'
    year_list = np.arange(1981,1990)
    i_yr = int(sys.argv[1])
    YYYY = year_list[i_yr]
    
    grid_to_IS2(icesheet,YYYY)





