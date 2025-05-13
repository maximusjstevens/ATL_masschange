#!/usr/bin/env python
'''
# MERRA_makezarr_discover_masschange.py

# Author: C. Max Stevens (@maximusjstevens)
# christopher.d.stevens-1@nasa.gov
#
# This script takes the annual, subsetted MERRA-2 files (netCDFs)
# and combines them into a zarr.
# 
# First run MERRA_concat...py
# make sure that .../zarr directory exists
# Then run this script.
# Then, each zarr needs to be zipped. (old?)

This script now uses a zarr.Storage.zipstore for the to_zarr write, 
thereby removing the need to create a zarr and zip (as discussed here:
https://github.com/zarr-developers/zarr-python/issues/756)

Zarr 3.0 seems to have broken this zip store capability (2/6/25)
'''

import math, cmath
import numpy as np
import scipy as sp
from numba import jit
import netCDF4 as nc
import h5py as h5
import datetime
import s3fs
import xarray as xr
import rioxarray as rx
import pandas as pd
import calendar
import os
import sys
import shutil
import glob
import time
import pickle
import re
import dateutil.parser as dparser
import pathlib
import subprocess as sub

class make_zarr:
    '''
    class to take the freq resolution files (one for each year), 
    combine them and save as zarr
    optionally can save a big netCDF will all years in one file.
    '''
    def __init__(self):
        pass

#     def read_freq_netcdfs(self,files):
#         def process_one_path(path):
#             print(path)
#             with xr.open_dataset(path) as ds:
#                 ds.load()
#                 return ds
#         paths = sorted(files)
#         datasets = [process_one_path(p) for p in paths]
#         combined = xr.concat(datasets, dim='time').sortby('time')
#         return combined

#     def read_merra_constants(self, cfile, LLbounds, clist):
#         '''
#         get the MERRA2 constants
#         '''
#         lat_max = LLbounds['lat_max']
#         lat_min = LLbounds['lat_min']
#         lon_max = LLbounds['lon_max']
#         lon_min = LLbounds['lon_min']

#         with xr.open_dataset(cfile) as ds:
#             dss = ds.sel({'lat': slice(lat_min,lat_max),'lon': slice(lon_min,lon_max)})[clist]
#             dss.load()  # load data to ensure we can use it after closing each original file
#         return dss

    def make_zarr(self,icesheet,LLbounds,out_path,d_in,saveZarr=True,saveBigNC=False,freq='1D'):
        
        if freq=='1D':
            freq_name = 'Daily'
        else:
            freq_name = freq

        dlist = [d_in]
        for decade in dlist:
            print(decade)
            zarr_name = pathlib.Path(f'M2_{icesheet}_{freq_name}_IS2mc_{decade}0.zarr')
            zarr_full = pathlib.Path(out_path,'zarr',zarr_name)
            
            if decade=='202': # assume we want to remake the 2020 zarr when we run this. 
                try:
                    shutil.rmtree(zarr_full)
                except OSError:
                    print('could not delete zarr')
            
            if os.path.exists(zarr_full): 
                print('zarr exists. will not remake.')
                continue

            # b_rem_path = pathlib.Path('/discover/nobackup/projects/icesat2/firn/ATL_masschange/CFM_forcing/AIS')
            b_rem_path = pathlib.Path('/discover/nobackup/cdsteve2/climate/MERRA2/AIS_FRICE') # Path to yearly netcdfs

            # AIS_remapped_bil_2024.nc
            allYearlyFiles = sorted(glob.glob(str(pathlib.Path(b_rem_path,'netCDF/4h',f'{icesheet}_remapped_*{decade}*conv.nc'))))
            # allYearlyFiles = sorted(glob.glob(str(pathlib.Path(b_rem_path,'netCDF/4h',f'MERRA2_{icesheet}_{freq_name}_*{decade}*_remapped.nc'))))
                
            with xr.open_mfdataset(allYearlyFiles,chunks=-1,parallel=True) as dsALL:            
                try:
                    dsALL = dsALL.drop_vars(['polar_Stereographic','spatial_ref'])
                except:
                    pass
                try:
                    for vv in ['TS_i','T2M_i','TScalc_i']:
                        dsALL[vv] = dsALL[vv].transpose("time",'y','x')
                except:
                    pass

                for data_var in dsALL.data_vars:
                    dsALL[data_var].encoding['compressor']=None
                
                dsALL = dsALL.chunk({'y':2,'x':14,'time':-1})
                zarr_out_fn = str(zarr_full) + '.zip'
                print('starting zarr write')
                now=time.time()
                dsALL.to_zarr(store=zarr_out_fn,mode='w',consolidated=True) # Be aware of where you are saving this; does not go to merra_path
                ttaken = time.time()-now
                print(f'zarr written successfully in {ttaken} s')

if __name__ == '__main__':
    
    cmd = 'module list'
    pp = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0]
    if 'cdo' not in str(pp):
        print('cdo not loaded. Remapping will not happen.')
    
    icesheet='AIS'
    freq='4h'

    dkey = int(sys.argv[1]) # this is the array value
    
    if icesheet=='GrIS':
        lat_min=55
        lat_max=90
        lon_min=-80
        lon_max=-10
        # lat_min=62
        # lat_max=68
        # lon_min=-50
        # lon_max=-42
    elif icesheet=='AIS':
        lat_min=-90
        lat_max=-60
        lon_min=-180
        lon_max=180

    LLbounds = dict(((k,eval(k)) for k in ('lat_min','lat_max','lon_min','lon_max')))

    # out_path = pathlib.Path(os.getenv('NOBACKUP'),f'climate/MERRA2/{icesheet}_FRICE')
    out_path = pathlib.Path('/discover/nobackup/projects/icesat2/firn/ATL_masschange/CFM_forcing/AIS/')
    if os.path.exists(out_path):
        pass
    else:
        os.makedirs(out_path)
    #allyears = np.arange(1980,2025) # years to resample
    dlist_in = ['198','199','200','201','202']
    d_in = dlist_in[dkey]
    print(f'Now making zarr for {d_in}.')
    MZ = make_zarr()
    MZ.make_zarr(icesheet,LLbounds,out_path,d_in,freq=freq)
    print('Done!')
