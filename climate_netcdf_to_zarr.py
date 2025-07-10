#!/usr/bin/env python
'''
climate_netcdf_to_zarr.py

Author: C. Max Stevens (@maximusjstevens)
christopher.d.stevens-1@nasa.gov

This script takes yearly climate data files 
and combines them into a zarr.

For Greenland, this took a bit over 1.5 hours per decade, so
SBATCH time should be ~2 hours.

Specifically, this script is designed to work with the MERRA-2
(possibly other) yearly files that have been regridded onto the 
ATL15 10km grid.

For file size management, this script creates a zarr zip store 
for each decade (e.g., the 1980 file has data from 1-1-1980 to 12-31-1989)

climate netCDFs come from:
/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}/netCDF/4h

and go to:
/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}/zarr

This script now uses a zarr.Storage.zipstore for the to_zarr write, 
thereby removing the need to create a zarr and zip, as I did in prior
iterations of this script. (Details on that process discussed here:
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
from pathlib import Path
import subprocess as sub

class make_zarr:
    '''
    class to take the freq resolution files (one for each year), 
    combine them and save as zarr
    optionally can save a big netCDF will all years in one very large file (not recommended).
    '''
    def __init__(self):
        pass

    def make_zarr(self,icesheet,LLbounds,out_path,d_in,saveZarr=True,saveBigNC=False,freq='1D'):
        
        if freq=='1D':
            freq_name = 'Daily'
        else:
            freq_name = freq

        dlist = [d_in]
        for decade in dlist:
            print(decade)
            # zarr_name = Path(f'M2_{icesheet}_{freq_name}_IS2mc_{decade}0.zarr')
            zarr_name = Path(f'M2_{icesheet}_{freq_name}_IS2mc_{decade}0_NOCONVOLUTION_NOCROP.zarr')
            # zarr_name = Path(f'LDAS_hr_{icesheet}_{freq_name}_IS2mc_{decade}0.zarr')
            zarr_dir  = Path(out_path,'zarr')
            zarr_dir.mkdir(parents=True,exist_ok=True)

            zarr_full = Path(zarr_dir,zarr_name)
            
            if decade=='202': # assume we want to remake the 2020 zarr when we run this. 
                try:
                    shutil.rmtree(zarr_full)
                except OSError:
                    print('could not delete zarr')
            
            if os.path.exists(zarr_full): 
                print('zarr exists. will not remake.')
                continue

            netcdf_path = Path(f'/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}') # Path to yearly netcdfs that will go into zarr.
            # netcdf_path = Path(f'/discover/nobackup/cdsteve2/climate/LDAS_highres/LDAS_outputs/{icesheet}') # Path to yearly netcdfs that will go into zarr.

            # allYearlyFiles = sorted(glob.glob(str(Path(netcdf_path,'netCDF/4h',f'*{icesheet}*{decade}*conv.nc'))))
            allYearlyFiles = sorted(glob.glob(str(Path(netcdf_path,'netCDF/4h',f'*{icesheet}*{decade}*NOCROP.nc'))))
            # allYearlyFiles = sorted(glob.glob(str(Path(netcdf_path,'netCDF/Daily',f'LDAS_{icesheet}*{decade}*.nc'))))
            ### allYearlyFiles is a list of the netCDFs for a certain decade.

            with xr.open_mfdataset(allYearlyFiles,chunks=-1,parallel=True) as dsALL:            
                try:
                    dsALL = dsALL.drop_vars(['polar_Stereographic','spatial_ref'])
                except:
                    pass
                try:
                    for vv in ['TS_i','T2M_i','TScalc_i']:
                        dsALL[vv] = dsALL[vv].transpose("time",'y','x') # this is an artifact of the netCDF-making process
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
    
    icesheet='GrIS'
    freq='4h'
    # freq='1d'

    try:
        dkey = int(sys.argv[1]) # this is the array value
    except:
        dkey = 0
        print(f'no decade specified; using {dkey}') 
    
    if icesheet=='GrIS':
        lat_min=55
        lat_max=90
        lon_min=-80
        lon_max=-10
    elif icesheet=='AIS':
        lat_min=-90
        lat_max=-60
        lon_min=-180
        lon_max=180

    LLbounds = dict(((k,eval(k)) for k in ('lat_min','lat_max','lon_min','lon_max')))

    # out_path = Path(f'/discover/nobackup/projects/icesat2/firn/ATL_masschange/CFM_forcing/{icesheet}/')
    out_path = Path(f'/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}')
    # out_path = Path(f'/discover/nobackup/cdsteve2/climate/LDAS_highres/LDAS_outputs/{icesheet}')
    
    dlist_in = ['198','199','200','201','202']
    d_in = dlist_in[dkey]
    print(f'Now making zarr for {d_in}.')
    MZ = make_zarr()
    MZ.make_zarr(icesheet,LLbounds,out_path,d_in,freq=freq)
    print('Done!')
