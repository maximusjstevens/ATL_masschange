#!/usr/bin/env python

'''
grid_CFM_masschange_init_timeres.py
===========================
This script initializes a zarr store that is
used to grid the CFM results. 

It is a variation of the older _init.py script. This one
creates 2 zarrs - one that is 1980-present at 5d res, the 
other 2018-present at 1d res.

It builds a dataset with variables FAC, filled with -9999.
Each variable has dimensions len(x_g), len(y_g),len(dtlist),
where x_g is the ATL15 grid x-dimension, y_g is the ATL15
grid y-dimension, and dtlist is the list of times in the
CFM results files (probably daily, 1980 to present).

This script is used in conjunction with 
grid_CFM_masschange_array_loop.py

srun --partition=hbv2 -N1 -n1 -c1 python /shared/home/cdsteve2/grid_CFM_masschange_init_timeres.py
'''

import numpy as np
import netCDF4 as nc
import h5py as h5
import datetime
import xarray as xr 
import pandas as pd
import calendar
import os
import sys
import glob
from pathlib import Path
import zarr
import socket

def decyeartodatetime(din,rounddate=False):
    def subdyd(din,rounddate):
        start = din
        year = int(start)
        rem = start - year
        base = datetime.datetime(year, 1, 1)
        result = base + datetime.timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * float(rem))
        if rounddate:
            result2 = result.replace(hour=0, minute=0, second=0, microsecond=0)
            return result2
        else:
            return result
    try:
        if din.size==1:
            return subdyd(din,rounddate)
        else:
            return [subdyd(dd,rounddate) for dd in din]
    except:
        return subdyd(din,rounddate)

def grid_CFM(zarr_name, icesheet, azure_drive='firnadls'):
    '''
    zarr_name: name of the zarr store, without extension 
    
    '''
    ##########################
    hh = socket.gethostname()
    if (('disc' in hh) or ('borg' in hh)):
        runloc = 'discover'
    elif 'gs615' in hh:
        runloc = 'local'
    else:
        runloc = 'azure'
    print(runloc,flush=True)
    
    ##########################
    
    z_ext='.zarr'
 
    zarr_name_1d = f'{zarr_name}_1d{z_ext}'
    zarr_name_5d = f'{zarr_name}_5d{z_ext}'

    ###############################
    
    if runloc=='azure':
        rdir      = Path('/mnt/firnadls/CFM_outputs/') # Path to the outputs. Need to update for icesheet
        grid_path = Path(f'/shared/home/cdsteve2/{azure_drive}/CFM_gridded/')
        ATLpath   = Path(f'/shared/home/cdsteve2/ATL') # path to ICESat-2 data for grid reference.
    
    elif runloc=='local':
        rdir      = Path(f'/Users/cdsteve2/research/ATL_masschange/CFM_outputs')
        grid_path = Path(f'/Users/cdsteve2/research/ATL_masschange/CFM_GrIS_gridded/')
        ATLpath   = Path(f'/Users/cdsteve2/research/ATL_masschange')

    elif runloc=='discover':
        rdir      = Path(f'/discover/nobackup/cdsteve2/ATL_masschange/CFMoutputs')
        grid_path = Path(f'/discover/nobackup/cdsteve2/ATL_masschange/CFM_gridded/')
        ATLpath   = Path(f'/discover/nobackup/cdsteve2/ATL_masschange/ATL')

    gridded_zarr_path_1d = Path(grid_path,f'{zarr_name_1d}') #path of the (to be created) zarr store
    gridded_zarr_path_5d = Path(grid_path,f'{zarr_name_5d}') #path of the (to be created) zarr store

    if os.path.isfile(gridded_zarr_path_1d):
        os.remove(gridded_zarr_path_1d)
    # if os.path.isfile(gridded_zarr_path_5d):
    #     os.remove(gridded_zarr_path_5d)
    
    ###############################
    ### open a results file from one pixel to get the time coordinate from the model runs
    ### these are somewhat arbitrary; just be sure to pick a pixel that completed.
    print(f'rdir: {rdir}')
    if runloc=='local':
        fl1 = glob.glob(f'{rdir}/*/CFMresults.hdf5')
        f_CFMresults = fl1[0]
        print(f_CFMresults)
    elif runloc=='azure':
        if icesheet == 'GrIS':
            f_CFMresults = '/mnt/firnadls/CFM_outputs/GrIS/CFMresults_-65000_-1315000_GSFC2020_LW-EMIS_eff_ALB-M2_interp/CFMresults.hdf5'
        elif icesheet == 'AIS':
            f_CFMresults = '/mnt/firnadls/CFM_outputs/AIS_A1/CFMresults_A1_2936_GSFC2020_LW-EMIS_eff_ALB-M2_interp/CFMresults.hdf5'
    elif runloc == 'discover':
        if icesheet == 'GrIS':
            f_CFMresults = '/discover/nobackup/cdsteve2/ATL_masschange/CFMoutputs/CFMresults_-65000_-1315000_GSFC2020_LW-EMIS_eff_ALB-M2_interp/CFMresults.hdf5'
        elif icesheet == 'AIS':
            f_CFMresults = '/discover/nobackup/cdsteve2/ATL_masschange/CFMoutputs/AIS/A1/CFMresults_A1_20011_GSFC2020_LW-EMIS_eff_ALB-M2_interp/CFMresults.hdf5'
    ###############################

    ###############################
    with xr.open_dataset(f_CFMresults) as _CFMresults:
        tvec = _CFMresults['DIP'][1:,0].values # vector of times in the results
        dti  = pd.DatetimeIndex(decyeartodatetime(tvec,rounddate=False)) # datetime index
        trez = round((np.diff(dti).mean()/1e9).astype(float)/86400) # resolution of the outputs in days

    dtlist_1d = pd.date_range(start=dti[0].round('D'),end=dti[-1].round('D'),freq=f'1D') # date times that are correctly rounded to the day.
    dtlist_5d = pd.date_range(start=dti[0].round('D'),end=dti[-1].round('D'),freq=f'5D') # date times that are correctly rounded to the day.
    
    if len(dtlist_1d)!=len(tvec):
        print('error generating time index. Exiting grid_CFM.py (line 90)')
        sys.exit()

    dtlist_1d = dtlist_1d[dtlist_1d>='2018'] # we are doing 2018-present for the daily outputs
    ###############################

    ###############################
    if icesheet=='GrIS':
        ATL_file = 'ATL15_GL_0318_10km_003_01.nc'
        with xr.open_dataset(Path(ATLpath,ATL_file),group='delta_h') as ATL15:
            x_g = ATL15.x.values
            y_g = ATL15.y.values
    elif icesheet=='AIS':
        ATL_file = 'ATL15_AIS_0324_10km_004_03_merged.nc'    
        with xr.open_dataset(Path(ATLpath,ATL_file)) as ATL15:
            x_g = ATL15.x.values
            y_g = ATL15.y.values
        if runloc=='discover':
            x_int = 1255000.
            y_int = -1505000.
            _ix   = np.where(x_g==x_int)[0][0]
            _iy   = np.where(y_g==y_int)[0][0]
            x_g   = x_g[_ix-10:_ix+10]
            y_g   = y_g[_iy-10:_iy+10]
            
    filler_xy = (-9999*np.ones((len(x_g), len(y_g)))).astype('float32')
    ###############################
            
    ###############################
    ### first make the full time series (1980-present) zarr at 5d resolution
    print('starting 5d zarr',flush=True)
    filler_5d = (-9999*np.ones((len(x_g), len(y_g),len(dtlist_5d)))).astype('float32')

    ds_5d = xr.Dataset(
        data_vars=dict(
            FAC      = (["x","y","time"], filler_5d, {'units':'m'}),
            SMB      = (["x","y","time"], filler_5d, {'units':'m i.e./day'}),
            SMB_a    = (["x","y","time"], filler_5d, {'units':'m i.e.'}),
            SNOWFALL = (["x","y","time"], filler_5d, {'units':'m i.e./day'}),
            SUBLIM   = (["x","y","time"], filler_5d, {'units':'m i.e./day'}),
            RAIN     = (["x","y","time"], filler_5d, {'units':'m i.e./day'}),
            RUNOFF   = (["x","y","time"], filler_5d, {'units':'m i.e./day'}),
            SMELT    = (["x","y","time"], filler_5d, {'units':'m i.e./day'}),
            TS       = (["x","y","time"], filler_5d, {'units':'K'}),
            SMB_RCI  = (["x","y"], filler_xy, {'units':'m i.e./day'}),
        ),
    coords=dict(
        x    = (["x"], x_g),
        y    = (["y"], y_g),
        time = dtlist_5d,
        reference_time = pd.Timestamp("1950-01-01"),
    ),
    attrs = dict(description="CFMoutputs",_FillValue=-9999))
    
    ds_5d = ds_5d.chunk({'y':1,'x':1,'time':-1})
    ds_5d.to_zarr(store=gridded_zarr_path_5d,mode='w',consolidated=True,write_empty_chunks=True,compute=False)
    ###############################

    ###############################
    ### now make the 2018-present zarr at 1d resolution
    print('starting 1d zarr',flush=True)
    filler_1d = (-9999*np.ones((len(x_g), len(y_g),len(dtlist_1d)))).astype('float32')

    ds_1d = xr.Dataset(
        data_vars=dict(
            FAC      = (["x","y","time"], filler_1d, {'units':'m'}),
            SMB      = (["x","y","time"], filler_1d, {'units':'m i.e./day'}),
            SMB_a    = (["x","y","time"], filler_1d, {'units':'m i.e.'}),
            SNOWFALL = (["x","y","time"], filler_1d, {'units':'m i.e./day'}),
            SUBLIM   = (["x","y","time"], filler_1d, {'units':'m i.e./day'}),
            RAIN     = (["x","y","time"], filler_1d, {'units':'m i.e./day'}),
            RUNOFF   = (["x","y","time"], filler_1d, {'units':'m i.e./day'}),
            SMELT    = (["x","y","time"], filler_1d, {'units':'m i.e./day'}),
            TS       = (["x","y","time"], filler_1d, {'units':'K'}),
            SMB_RCI  = (["x","y"], filler_xy, {'units':'m i.e./day'}),
        ),
    coords=dict(
        x    = (["x"], x_g),
        y    = (["y"], y_g),
        time = dtlist_1d,
        reference_time = pd.Timestamp("1950-01-01"),
    ),
    attrs = dict(description="CFMoutputs",_FillValue=-9999))
    
    ds_1d = ds_1d.chunk({'y':1,'x':1,'time':-1})
    ds_1d.to_zarr(store=gridded_zarr_path_1d,mode='w',consolidated=True,write_empty_chunks=True,compute=False)


if __name__ == '__main__':
    icesheet='AIS'
    
    zarr_name = f'CFM_gridded_{icesheet}'
    
    grid_CFM(zarr_name,icesheet)
    print('Done!')
