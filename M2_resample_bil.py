#!/usr/bin/env python

'''
M2_resample_bil.py
2025/02/18

Script to take subsetted MERRA-2 data (dsALL_YYYY.nc)
and use cdo tools to remap all variables to ATL15 10k 
grid with bilinear interpolation.

run at command line with:
python M2_resample_bil.py XXXX
where XXXX is the year to resample

or use slurm array with M2_bilinear_cdo.j:
sbatch --array=1980-2024 M2_bilinear_cdo.j

Took about 2 hours in test from command line on 
discover head node.
Less than an hour (35 min?) when run on a compute node with slurm
'''

import math, cmath
import numpy as np
import scipy as sp
# from numba import jit
# import netCDF4 as nc
# import h5py as h5
import datetime
# import s3fs
import xarray as xr
# import rioxarray as rx
# import pandas as pd
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

from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
# from scipy.spatial import KDTree
# import cartopy.crs as ccrs
# import pyproj
# from rasterio.enums import Resampling
from pathlib import Path
# import gc 

# cdo remapbil,/discover/nobackup/cdsteve2/ATL_masschange/downscale/ATL15_10km_AIS_gridfile.txt dsALL_2020.nc test_bil.nc

class cdo_bil:
    
    def __init__(self):
        pass

    def read_merra_constants(self, cfile, LLbounds, clist):
        '''
        get the MERRA2 constants
        '''
        lat_max = LLbounds['lat_max']
        lat_min = LLbounds['lat_min']
        lon_max = LLbounds['lon_max']
        lon_min = LLbounds['lon_min']
    
        with xr.open_dataset(cfile) as ds:
            dss = ds.sel({'lat': slice(lat_min,lat_max),'lon': slice(lon_min,lon_max)})[clist]
            dss.load()  # load data to ensure we can use it after closing each original file
        return dss  
    
    def cdo_resample(self, YY, LLbounds):

        cfile = '/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/MERRA2.const_2d_asm_Nx.00000000.nc4' # MERRA2 constants
        clist = ['FRLANDICE']
        ds_constants = self.read_merra_constants(cfile,LLbounds,clist)
    
        icesheet = 'AIS'
        atl15 = xr.open_dataset('/discover/nobackup/cdsteve2/ATL_masschange/ATL15_AIS_0324_10km_004_03_merged.nc')
        
        all_file = f'/discover/nobackup/cdsteve2/climate/MERRA2/AIS_FRICE/netCDF/4h/dsALL_{YY}.nc'
        infile = f'/discover/nobackup/cdsteve2/climate/MERRA2/AIS_FRICE/netCDF/4h/dsALL_{YY}_crop_conv.nc'

        with xr.open_dataset(all_file) as af_in:
            af = af_in.copy()

            for kk,dv in enumerate(af.data_vars):
                # print(dv)
                af[dv] = af[dv].where((ds_constants['FRLANDICE'].isel(time=0))>0.95,np.nan) # get rid of non-ice M2 pixels to not use for remapping
                for ii,tt in enumerate(af.time):                
                    ds_X = af[dv].isel(time=ii) # "ds_test" above
                        
                    ds_mask = xr.where(ds_X.notnull(),1.0,0.0)
                    ds_mask_c = gaussian_filter(ds_mask, sigma=1, mode='wrap')
                    
                    data_z = ds_X.where(ds_X.notnull(),0.0)
                    data_z_c = gaussian_filter(data_z, sigma=1,mode='wrap')

                    af['_tc'] = (['lat','lon'],data_z_c/ds_mask_c)
                    af['_tc'] = af['_tc'].where(af.lat<-61.5,np.nan)
                    # af[dv].loc[dict(time=tt)] = af['_tc'].values
                    af[dv].loc[dict(time=tt)] = xr.where(ds_X.notnull(),ds_X,af['_tc']).values
                    af = af.drop_vars(["_tc"])
            
            af.to_netcdf(infile)
           
        outfile = f'/discover/nobackup/cdsteve2/climate/MERRA2/AIS_FRICE/netCDF/4h/AIS_remapped_bil_{YY}_conv.nc'        
        
        sub.call(["cdo",f"remapbil,/discover/nobackup/cdsteve2/ATL_masschange/downscale/ATL15_10km_{icesheet}_gridfile.txt",infile,outfile])

        os.remove(infile)

if __name__ == '__main__':
    start_time = time.time()
    print(f'start_time: {start_time}')
    YY=int(sys.argv[1])

    lat_min=-90
    lat_max=-60
    lon_min=-180
    lon_max=180
    
    LLbounds = dict(((k,eval(k)) for k in ('lat_min','lat_max','lon_min','lon_max')))
    
    cdo=cdo_bil()
    cdo.cdo_resample(YY,LLbounds)
    finish_time = time.time()
    total_time = finish_time-start_time
    print('finished!')
    print(total_time/60)