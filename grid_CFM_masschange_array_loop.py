#!/usr/bin/env python

'''
grid_CFM_masschange_array_loop.py
=================================
This script takes CFM outputs and puts them
on the ICESat-2 grid.

Run using CFM_grid_azure.j:
>>> sbatch --array=0-179 CFM_grid_azure_loop.j QUAD

To utilize parallel writing, which allows many CFM results
files to be put into the gridded file simultaneously, we use
a zarr store. This script puts each pixel's results into that
zarr.

You must first run grid_CFM_masschange_init.py, 
which builds the gridded zarr and fills it with zeros.

This script basically opens a CFM results file and puts some
(e.g., FAC evolution) into the zarr in the correct pixel. A bit 
of testing showed that there was substantial overhead to this
script if I just ran it in parallel with slurm (i.e., run this
script thousands of times, once for each pixel). Instead, it
uses a loop and handles 200 results files at a time. So, if you
have 20000 runs, this script will run 100 times. 

Notes on units:
SNOWFALL   	= f['Modelclimate'][1:,1] # m i.e./year, includes deposition
SUBLIM 	= f['Modelclimate'][1:,5] # m i.e./year, is negative so should be added to SMB
RAIN   	= f['Modelclimate'][1:,4] # m i.e./year
RUNOFF 	= f['runoff'][1:,1]/0.917*stps_per_year # [m i.e./year]: output is in w.e./timestep, so needs to be divided by 0.917. Values are positive, so needs to be subtracted in SMB calculation
SMELT 	= f['meltvol'][1:,1]/0.917*stps_per_year # [m i.e./year]: output is in w.e./timestep, so needs to be divided by 0.917. (checked units 25/04/09)
TS 		= f['Modelclimate'][1:,2] # K. Use TS from modelclimate rather than first column of temperature, as that is post-diffusion.

stps_per_year: This gets the same steps per year as is used in RCMpkl_to_spin.py, so ensures that converting between per year and per timestep values are correct.

old config name:  # configName = Path(f'/mnt/firnadls/CFM_outputs/CFMresults_{vv}_GSFC2020_LW-EMIS_eff_ALB-M2_interp/CFMconfig_{vv}_GSFC2020_LW-EMIS_eff_ALB-M2_interp.json')

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
import json
import zarr
import socket
import time
import traceback

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

def grid_CFM(zarr_name, icesheet, vv, quad, zipzarr=False, azure_drive='firnadls'):
    '''
    zarr_name: name of the zarr store, without extension 
    icesheet: 'AIS' or 'GrIS'
    vv: pixel number
    quad: which quadrant (AIS only)
    zipzarr: whether to write to zipped zarr or directory store
    azure_drive: 'firnadls' or 'firndata'. ADLS is cheap storate, firndata is fast.
    '''

    ########################
    hh = socket.gethostname()
    if 'disc' in hh:
        runloc = 'discover'
    elif 'GS407' in hh:
        runloc = 'local'
    else:
        runloc = 'azure'
    ########################

    if zipzarr:
        z_ext='.zarr.zip'
    else:
        z_ext='.zarr'
        
    if runloc=='azure':
        if icesheet=='AIS':
            rdir = Path(f'/mnt/firnadls/CFM_outputs/{icesheet}_{quad}')
            if 'add' in quad: #A1_add, A4_add, A1_add_2
                qq = quad.split('_')[0]
                rmid = f'CFMresults_{qq}_{vv}_GSFC2020_LW-EMIS_eff_ALB-M2_interp'
            else:
                rmid = f'CFMresults_{quad}_{vv}_GSFC2020_LW-EMIS_eff_ALB-M2_interp'
        else:
            rdir = Path(f'/mnt/firnadls/CFM_outputs/{icesheet}')
            rmid = f'CFMresults_{vv}_GSFC2020_LW-EMIS_eff_ALB-M2_interp'

        gridded_zarr_path_1d = Path(f'/shared/home/cdsteve2/{azure_drive}/CFM_gridded/{zarr_name}_1d{z_ext}') # path of the zarr store  
        gridded_zarr_path_5d = Path(f'/shared/home/cdsteve2/{azure_drive}/CFM_gridded/{zarr_name}_5d{z_ext}') # path of the zarr store

        if icesheet=='GrIS':
            pt_path = Path('/shared/home/cdsteve2/CommunityFirnModel/CFM_main/IS2_pixelstorun_GrIS.csv')
        elif icesheet=='AIS':
            pt_path = Path(f'/shared/home/cdsteve2/CommunityFirnModel/CFM_main/IS2_pixelstorun_AIS_{quad}_full.csv')
    
    elif runloc=='local':
        rdir = Path(f'/Users/cdsteve2/research/ATL_masschange/CFM_outputs')
        gridded_zarr_path = Path(f'/Users/cdsteve2/research/ATL_masschange/CFM_GrIS_gridded/{zarr_name}{z_ext}')
        ATLpath = Path(f'/Users/cdsteve2/research/ATL_masschange')
        pt_path = Path('/Users/cdsteve2/research/ATL_masschange/IS2_icepixels.csv')

    CFM_results_path = Path(rdir, rmid, 'CFMresults.hdf5')
    print(f'CFM_results_path is {CFM_results_path}')
        
    ##################################
    ### get x/y points, make sure they 
    ### match those for the run from json
    print(f'vv: {vv}') # pixel number
    M2pts = np.genfromtxt(pt_path,delimiter=',',skip_header=1)
    rw = M2pts[vv] #row with x/y pair in the points file
    xM = rw[0]
    yM = rw[1]
    
    try:
        configName = list(Path(rdir,rmid).glob('*.json'))[0]
        with open(configName, "r") as f:
            jsonString      = f.read()
            c          = json.loads(jsonString)    
        x_val_j = c['x_val']
        y_val_j = c['y_val']
    except:
        print('error with config open.')
        x_val_j = -9999
        y_val_j = -9999

    if ((xM!=x_val_j) or (yM!=y_val_j)):
        print('x or y values do not match')
        print(f'x_val: {x_val_j}, xM: {xM}')
        print(f'y_val: {y_val_j}, yM: {yM}')
    ##################################    

    ##################################
    ### get CFM data from results file
    if os.path.exists(CFM_results_path):
        
        pixel_exists = True
        print('File exists.')
        
        with xr.open_dataset(CFM_results_path) as ds_CFM_results:
            ### time stuff:
            _dectime      = ds_CFM_results['Modelclimate'][1:,0].load() #decimal time from CFMresults. 
            stps_per_year = 1/(_dectime.diff(dim='phony_dim_0').mean()).values 
            _dti_m        = pd.DatetimeIndex(decyeartodatetime(_dectime.values.astype('float64'),rounddate=False)) # datetime index
            trez          = round((np.diff(_dti_m).mean()/1e9).astype(float)/86400) # resolution of the outputs in days
            dti           = pd.date_range(start=_dti_m[0].round('D'),end=_dti_m[-1].round('D'),freq=f'{trez}D') # date times that are correctly rounded to the day.

            ###
            SNOWFALL = ds_CFM_results['Modelclimate'][1:,1] # m i.e./year, includes deposition
            SUBLIM   = ds_CFM_results['Modelclimate'][1:,5] # m i.e./year, is negative so should be added to SMB
            RAIN     = ds_CFM_results['Modelclimate'][1:,4] # m i.e./year
            RUNOFF   = ds_CFM_results['runoff'][1:,1]/0.917*stps_per_year # [m i.e./year]
            SMELT    = ds_CFM_results['meltvol'][1:,1]/0.917*stps_per_year # [m i.e./year]
            TS       = ds_CFM_results['Modelclimate'][1:,2] # K
            FAC      = ds_CFM_results['DIP'][1:,1] # m
            
            SMB         = (SNOWFALL+SUBLIM+RAIN-RUNOFF)/stps_per_year # m i.e./day
            SMB         = SMB.rename({SMB.dims[0]:'time'}).assign_coords(time=('time',dti))
            SMB_RCImean = SMB.sel(time=slice('1980','2019')).mean() # m i.e./day
            SMB_anomaly = (SMB-SMB_RCImean).cumsum() # m i.e.
            
            ds_CFM = SMB.to_dataset(name='SMB')
            ds_CFM['SMB_a']    = (('time'),SMB_anomaly.values) # m i.e.
            ds_CFM['SNOWFALL'] = (('time'),SNOWFALL.values/stps_per_year) # m i.e./day
            ds_CFM['SUBLIM']   = (('time'),SUBLIM.values/stps_per_year) # m i.e./day
            ds_CFM['RAIN']     = (('time'),RAIN.values/stps_per_year) # m i.e./day
            ds_CFM['RUNOFF']   = (('time'),RUNOFF.values/stps_per_year) # m i.e./day
            ds_CFM['SMELT']    = (('time'),SMELT.values/stps_per_year) # m i.e./day
            ds_CFM['TS']       = (('time'),TS.values) # K
            ds_CFM['FAC']      = (('time'),FAC.values) # m
            ds_CFM['SMB_RCI']  = SMB_RCImean
            
        ds_CFM_1d = ds_CFM.sel(time=slice('2018',None))
        ds_CFM_5d = ds_CFM.drop_vars(['SMB_RCI']).resample(time='5d').mean()
        ds_CFM_5d['SMB_RCI'] = SMB_RCImean
    else:
        pixel_exists = False
    ##################################    
            
    ##################################
    ### now put in 1day zarr store (2018-present)
    # with xr.open_dataset(gridded_zarr_path,engine='zarr') as ds:
    try:
        with zarr.open(gridded_zarr_path_1d, mode='a') as ds_Z_1d:
            dtlist = ds_Z_1d.time[:]
            
            # find coordinates in the zarr of the pixel
            y_g = ds_Z_1d.y[:]
            x_g = ds_Z_1d.x[:]
            iy = np.where(y_g==yM)[0][0]
            ix = np.where(x_g==xM)[0][0]
        
            print(f'xM:{xM}')
            print(f'yM:{yM}')
            print(f'ix:{ix}')
            print(f'iy:{iy}')
        
            if pixel_exists:
        
                dvars = list(ds_CFM_1d.data_vars)
                dvars.remove('SMB_RCI')
        
                for _dvar in dvars:
                    ds_Z_1d[_dvar][ix,iy,:] = ds_CFM_1d[_dvar].values
                ds_Z_1d['SMB_RCI'][ix,iy]    = ds_CFM_1d['SMB_RCI'].values

            else:
                # dvars = ['FAC','TS','SMB','SMB_a','SNOWFALL','SUBLIM','RAIN','RUNOFF','SMELT']
                # for dvar in dvars:
                #     ds_Z_1d[dvar][ix,iy,:]      = np.zeros(len(dtlist))
                # ds_Z_1d['SMB_RCI'][ix,iy] = 0.0
                print('File does not exist.')
        result_1d = True
    except Exception:
        print('error with 1 day write')
        traceback.print_exc()
        result_1d = False

    ###############
    ### now put in 5day zarr store (1980-present)
    try:
        with zarr.open(gridded_zarr_path_5d, mode='a') as ds_Z_5d:
            dtlist = ds_Z_5d.time[:]
            
            # find coordinates in the zarr of the pixel
            y_g = ds_Z_5d.y[:]
            x_g = ds_Z_5d.x[:]
            iy = np.where(y_g==yM)[0][0]
            ix = np.where(x_g==xM)[0][0]
        
            print(f'xM:{xM}')
            print(f'yM:{yM}')
            print(f'ix:{ix}')
            print(f'iy:{iy}')
        
            if pixel_exists:
        
                dvars = list(ds_CFM_5d.data_vars)
                dvars.remove('SMB_RCI')
        
                for _dvar in dvars:
                    ds_Z_5d[_dvar][ix,iy,:] = ds_CFM_5d[_dvar].values
                ds_Z_5d['SMB_RCI'][ix,iy]    = ds_CFM_5d['SMB_RCI'].values

            else:
                # dvars = ['FAC','TS','SMB','SMB_a','SNOWFALL','SUBLIM','RAIN','RUNOFF','SMELT']
                # for dvar in dvars:
                #     ds_Z_5d[dvar][ix,iy,:]      = np.zeros(len(dtlist))
                # ds_Z_5d['SMB_RCI'][ix,iy] = 0.0
                print('File does not exist.')
        result_5d = True
    except Exception:
        print('error with 5 day write')
        traceback.print_exc()
        result_5d = False

    return result_1d, result_5d       
       
if __name__ == '__main__':
    '''
    The loop does batches of 200 pixels. So, with ii=0, this will do 0-199, ii=1 gives 200-299, etc.
    np.divmod(pixels,batchsize) gives number of array jobs needed in sbatch and how to modify the 
    final array job.
    
    e.g.:
    GrIS has 20013 pixels. np.divmod(20013,200) gives (100,13). So, modify iend below to be 20013.

    generically: np.divmod(total_pixels, loop_size) gives (array_size, extras).

    So, modify below:
    if iend == loop_size*array_size:
        iend = loop_size*array_size + extras 
    the run:
    >>> sbatch --array=0-array_size CFM_grid_azure_loop.j    
    '''

    tic=time.time()
    ii = int(sys.argv[1])
    print('ii',ii)

    icesheet='GrIS'
    
    if icesheet=='GrIS':
        quad=None
    
    else:
        if len(sys.argv)>2:
            quad=sys.argv[2]
        else:
            quad='A1'
            print(f'Quad is {quad}.')
    
    custom_run=False
        
    zarr_name = f'CFM_gridded_{icesheet}'

    istart = ii*200
    iend = istart + 200 # +200 because arange below is non-inclusive, so array is e.g. 0-199, 200-399, etc.
    
    if icesheet=='GrIS':
        ### 20037 pixels
        ### --array==0-99 for sbatch
        if iend == 20000:
            iend = 20037
    elif icesheet=='AIS':
        if quad=='A1':
            ### 35909 pixels
            ### --array=0-179 for sbatch (istart = 35800)
            if iend == 36000:
                iend = 35909
        elif quad=='A2':
            ### 27283 pixels
            ### --array=0-136 for sbatch (istart = 27200)
            if iend == 27400:
                iend = 27283
        elif quad=='A3':
            ### 20130 pixels
            ### --array=0-100 for sbatch (istart = 20000)
            if iend == 20200:
                iend = 20130
        elif quad=='A4':
            ### 38900 pixels
            ### --array=0-194 for sbatch (istart = 38800)
            if iend == 39000:
                iend = 38900
                
    iarray = np.arange(istart,iend)

    if custom_run:
        print('GRIDDING CUSTOM SUBSET')
        ### build custom code here
        ### generally, icesheet and quad need to be the same as above
        ### import a list of pixel numbers from that quad that need to be gridded
        _df = pd.read_csv('missing_lists/missing_lists/missing_spin_A4_0508.txt',header=None)
        iarray = np.array(_df[0].values)
    ### end custom

    result_5d_list = []
    result_1d_list = []

    for jj,vv in enumerate(iarray):
        try:
            result_1d, result_5d = grid_CFM(zarr_name, icesheet, vv, quad) # call function to open CFM results file, process results, write to zarr
            if not result_1d:
                result_1d_list.append(vv)
                result_5d_list.append(vv)
        except KeyboardInterrupt:
            print('Keyboard interrupt')
            sys.exit()
        except Exception as e:
            print(f"Exception encountered for {vv}:", e)
        # if np.mod(jj,20)==0:
        check_time = time.time()-tic
        print(f'check_time ({jj}) = {check_time}')
    print(f'5d error pixels: {result_5d_list}')
    print(f'1d error pixels: {result_1d_list}')
    print('run time =' , time.time()-tic , 'seconds',flush=True)
