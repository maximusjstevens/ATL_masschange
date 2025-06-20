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

For antarctica:
Took about 2 hours in test from command line on discover head node.
Less than an hour (35 min?) when run on a compute node with slurm

Greenland takes about 10 minutes per year
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
    
    def cdo_resample(self, YY, LLbounds,icesheet):

        cfile = '/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/MERRA2.const_2d_asm_Nx.00000000.nc4' # MERRA2 constants
        clist = ['FRLANDICE']
        ds_constants = self.read_merra_constants(cfile,LLbounds,clist)
    
        # if icesheet == 'AIS':
        #     atl15 = xr.open_dataset('/discover/nobackup/cdsteve2/ATL_masschange/ATL15_AIS_0324_10km_004_03_merged.nc')
        # elif icesheet=='GrIS':
        #     atl15 = xr.open_dataset('/discover/nobackup/cdsteve2/IS2_data/ATL15/ATL15_GL_0324_10km_004_03.nc',group='delta_h')
        
        all_file = f'/discover/nobackup/cdsteve2/climate/MERRA2/subsets/{icesheet}/M2_{icesheet}_{YY}.nc'
        infile = f'/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}/netCDF/4h/M2_{icesheet}_{YY}_crop_temp.nc'

        with xr.open_dataset(all_file) as af_in:
            af = af_in.copy()

            convolve = False
            crop = False

            if (convolve and not crop):
                print('Cannot convolve without cropping. Setting crop to true.')
                crop = True

            if crop:
                for kk,dv in enumerate(af.data_vars):
                    af[dv] = af[dv].where((ds_constants['FRLANDICE'].isel(time=0))>0.95,np.nan) # get rid of non-ice M2 pixels to not use for remapping
                    if convolve:
                        for ii,tt in enumerate(af.time):                
                            ds_X = af[dv].isel(time=ii) # "ds_test" above
                                
                            ds_mask = xr.where(ds_X.notnull(),1.0,0.0)
                            ds_mask_c = gaussian_filter(ds_mask, sigma=1, mode='wrap')
                            
                            data_z = ds_X.where(ds_X.notnull(),0.0)
                            data_z_c = gaussian_filter(data_z, sigma=1,mode='wrap')
        
                            af['_tc'] = (['lat','lon'],data_z_c/ds_mask_c)
                            if icesheet=='AIS':
                                af['_tc'] = af['_tc'].where(af.lat<-61.5,np.nan) # .where here retains values where true; nan elsewhere
                            # af[dv].loc[dict(time=tt)] = af['_tc'].values
                            af[dv].loc[dict(time=tt)] = xr.where(ds_X.notnull(),ds_X,af['_tc']).values
                            af = af.drop_vars(["_tc"])
                                    
            af.to_netcdf(infile) # this is just a temporary file with the MERRA-2 gridding but the non-ice pixels crop and the convolution applied; this is used for the remapping.
           
        if convolve:
            outfile = f'/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}/netCDF/4h/M2_{icesheet}_{YY}_ATL15-10k_bil_conv.nc'
        else:
            if crop:
                outfile = f'/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}/netCDF/4h/M2_{icesheet}_{YY}_ATL15-10k_bil.nc'
            else:
                outfile = f'/discover/nobackup/cdsteve2/climate/MERRA2/remapped/{icesheet}/netCDF/4h/M2_{icesheet}_{YY}_ATL15-10k_bil_NOCROP.nc'
        
        sub.call(["cdo",f"remapbil,/discover/nobackup/cdsteve2/IS2_data/gridfiles/ATL15_10km_{icesheet}_gridfile.txt",infile,outfile])

        os.remove(infile)

if __name__ == '__main__':
    start_time = time.time()
    print(f'start_time: {start_time}')
    YY=int(sys.argv[1])

    icesheet='GrIS'

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
    
    cdo=cdo_bil()
    cdo.cdo_resample(YY,LLbounds,icesheet)
    finish_time = time.time()
    total_time = finish_time-start_time
    print('finished!')
    print(total_time/60)