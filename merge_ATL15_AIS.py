#!/usr/bin/env python

'''
merge_ATL15_AIS.py
===========================
This script creates a single AIS ATL15 delta_h netCDF
by merging the four AIS quadrant ATL15 files. 

Only does the delta_h group from the original ATL15 netCDFs!

The outputted (merged) netCDF is used in the ATL mass change 
workflow: first to generate the gridfile (explained in the workflow notes) and then to initialize the gridded CFM zarr (grid_CFM_masschange_init_timeres.py)
'''

import os
import xarray as xr
import rioxarray
from pathlib import Path
import IS2view

def merge_ATL15(out_path,fn):
    print('merging!')
    p_atl15 = Path('/discover/nobackup/cdsteve2/IS2_data/ATL15')
    ds_l_10 = [IS2view.io.from_file(Path(p_atl15,f'ATL15_A{ii}_0324_10km_004_03.nc'),group='delta_h') for ii in [1,2,3,4]]
    ds10 = rioxarray.merge.merge_datasets(ds_l_10)
    ds10.to_netcdf(Path(out_path,fn))

if __name__=='__main__':
    out_path = Path('/discover/nobackup/cdsteve2/IS2_data/ATL15')
    fn = 'ATL15_AIS_0324_10km_004_03_merged.nc'
    merge_ATL15(out_path,fn)