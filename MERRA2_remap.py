#!/usr/bin/env python

'''
# MERRA2_remap.py

# Author: C. Max Stevens (@maximusjstevens)
# christopher.d.stevens-1@nasa.gov
#

Updates:
25/04: edited so that now there is a separate script to run prior to this one that
does the MERRA2 subsetting for the year/region.

25/01/31: edited to make script ice-sheet agnostic (i.e., work correctly for antarctica),
added some commenting, and added new outputs (raw, subsetted MERRA2, called dsALL within)
and a version of the remapped data in which I just use a bilinear interp for everything. 
The bilinear file still includes the temperatures based on MERRA2 lapse rates and ATL14.

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
from scipy.spatial import KDTree
import cartopy.crs as ccrs
import pyproj
from rasterio.enums import Resampling
from pathlib import Path

import IS2view
import rioxarray.merge

class MERRA_remap:
    '''
    Class to open subsetted MERRA-2 data (created by running MERRA2_subsetter_array.py)
    and remap it to the ICESat-2 10km grid. 

    The default behavior is to use a bilinear resampling. Previously, I used CDO tools
    to do a conservative remapping; however that produced a map of Greenland that looked
    a bit like a disco ball. 

    I also previously used ATL14 elevations along with local lapse rates from MERRA-2
    to try to produce a better temperature field. Mixed results on this.
    '''
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
    
    def T_lapserate(self,ds_M2_T,atl15_10k,icesheet):
        '''     
        ############################################
        ### temperature remapping using lapse rate
        ############################################
        use lapse rates calculated from MERRA-2 and ATL14 data to interpolate temperature fields.

        used a version of ATL14 sampled to ATL15 10k grid using reproject match (average)
        This resample is done in ATL14_repmatch.py
        (an older version of this function used used raw ATL14 data and xarray's 
        coarsen function, but this did not work correctly)
        '''

        with xr.open_dataarray(f'/discover/nobackup/cdsteve2/climate/MERRA2/ATL14_{icesheet}_h_10k_repmatch.nc') as ATL14in:
            ATL14 = ATL14in.copy()
            ATLX,ATLY = np.meshgrid(ATL14.x.values,ATL14.y.values)
            ATL14_xyz = np.array([ATLX.ravel(),ATLY.ravel(),ATL14.values.ravel()]).T
            ATL14_h = ATL14.values

        if icesheet=='GrIS':
            is2_epsg='3413'
        elif icesheet=='AIS':
            is2_epsg='3031'
        transformer = pyproj.Transformer.from_crs("EPSG:4326",f"EPSG:{is2_epsg}")

        LON,LAT = np.meshgrid(ds_M2_T.lon.values,ds_M2_T.lat.values)
        Lro,Lco = np.shape(LON)
        lon_rav = np.ravel(LON)
        lat_rav = np.ravel(LAT)

        ll_array = np.array([lon_rav,lat_rav]).T # array with all lat,lon pairs from MERRA-2
        
        ### MERRA-2 points reprojected into polar stereographic (this is an irregular grid)
        M2_xy_reproj = np.array([transformer.transform(ll_array[ii][1],ll_array[ii][0]) for ii,pair in enumerate(ll_array)])
        
        m2_z = sp.interpolate.griddata(np.array([ATLX.ravel(),ATLY.ravel()]).T,ATL14_h.ravel(),M2_xy_reproj,method='nearest') # m2_z is the heights of each MERRA-2 grid point according to ATL14
        
        ### This is an array with each row of the array being the x,y,z of each M2 grid point in polar stereographic
        M2_xyz_reproj = np.hstack((M2_xy_reproj,m2_z[:,np.newaxis]))

        ### Add elevation data to ATL15, based on nearest neighbor from ATL14. 
        atl15_10k['z'] = ATL14
        
        ### Create xyz array of atl15 grid points
        atlX,atlY = np.meshgrid(atl15_10k.x.values,atl15_10k.y.values)
        atlX_ro,atlX_co = np.shape(atlX)
        r2x = np.ravel(atlX)
        r2y = np.ravel(atlY)
        r2z = np.ravel(atl15_10k.z.values)
        ATL15_xyz_rav = np.array([r2x,r2y,r2z]).T # array of ATL15 x,y,z, raveled (z from ATL14)
        
        kd_tree1 = KDTree(M2_xy_reproj)
        kd_tree2 = KDTree(ATL15_xyz_rav[:,0:2])
        kdi = kd_tree2.query_ball_tree(kd_tree1,70000) # consider points within 70km

        ### now do the temperature interpolation based on elevation and lapse rate
        T_interp_dict = {} 
        for vv in ['TS','T2M','TScalc']:
            _da_vv = ds_M2_T[vv]
            #With this each colum is a time slice, each row is a raveled x/y (lon/lat)
            ATL_T = np.zeros((len(ATL15_xyz_rav),len(_da_vv.time)))

            tttall = (_da_vv.stack(xy=['lat','lon']).values).T
            for kk in range(len(ATL15_xyz_rav)):
                iiis = kdi[kk]
                if len(iiis)==0:
                    ided = np.nan * np.ones(len(_da_vv.time))
                else:
                    ttt = tttall[iiis]
                    tttmean = np.nanmean(ttt,axis=0) #mean at each time step (len=100)
                    tttstd = np.nanstd(ttt,axis=0) #std at each time step (len=100)

                    zzz1 = M2_xyz_reproj[:,-1][iiis]
                    zzz = np.tile(zzz1[:,None],np.shape(ttt)[1])
                    zzzmean = np.nanmean(zzz,axis=0)
                    zzzstd = np.nanstd(zzz,axis=0)

                    n = np.shape(ttt)[0]

                    cov = np.nansum((zzz - zzzmean) * (ttt - tttmean),axis=0)/(n)
                    cor = cov/(zzzstd * tttstd)
                    sloop = cov/(zzzstd**2)
                    interc = tttmean - zzzmean*sloop

                    ided = sloop*ATL15_xyz_rav[kk,-1]+interc

                ATL_T[kk,:] = ided

            T_interp_dict[f'{vv}_i'] = ATL_T.reshape(atlX_ro,atlX_co,len(_da_vv.time))              
            # ds_M2_10k[f'{vv}_i'] = (['y','x','time'],ATL_T.reshape(atlX_ro,atlX_co,len(_da_vv.time)))
            # ds_M2_10k[f'{vv}_i'] = ds_M2_10k[f'{vv}_i'].T.astype('float32') 

        return T_interp_dict
    ############################################
    ### end temperature remapping using lapse rate
    ############################################ 
    
    def M2remap(self,icesheet,LLbounds,out_path,YY,calc_melt=True,freq='1D',remap_type='bilinear',lapserate=False):
        '''
        main function to remap MERRA-2 data.
        '''

        if freq=='1D':
            freq_name = 'Daily'
        else:
            freq_name = freq

        tnow = time.time() #time it

        print(YY) # year
        
        ###############################
        ### Load subsetted MERRA-2 data
        subPath = Path(f'/discover/nobackup/cdsteve2/climate/MERRA2/subsets/{icesheet}')
        dsALL_fn = f'M2_{icesheet}_{YY}.nc'
        f_subset = pathlib.Path(subPath,dsALL_fn) 
        
        if os.path.exists(f_subset): # has been subsetted.
            dsALL = xr.open_dataset(f_subset)
        else:
            print('MERRA2 subset not found. Exiting')
            sys.exit()
    
        cfile = '/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/MERRA2.const_2d_asm_Nx.00000000.nc4' # MERRA2 constants
        clist = ['FRLANDICE']
        ds_constants = self.read_merra_constants(cfile,LLbounds,clist)

        for dv in dsALL.data_vars: # get rid of non-ice M2 pixels to not use for remapping
            dsALL[dv] = dsALL[dv].where((ds_constants['FRLANDICE'].isel(time=0))>0.95,np.nan)
        ###############################

        ###############################
        ### Load ATL15 ################
        if icesheet=='GrIS':
            atl15_10k = xr.open_dataset(f'/discover/nobackup/cdsteve2/IS2_data/ATL15/ATL15_GL_0318_10km_003_01.nc',group='delta_h').rio.write_crs(input_crs="EPSG:3413")
            for key in list(atl15_10k.keys()):
                atl15_10k[key].rio.write_nodata(np.nan, inplace=True)
        elif icesheet=='AIS':
            p_atl15 = pathlib.Path('/discover/nobackup/cdsteve2/IS2_data/ATL15/')
            ds_l_10 = [IS2view.io.from_file(pathlib.Path(p_atl15,f'ATL15_A{ii}_0324_10km_004_03.nc'),group='delta_h') for ii in [1,2,3,4]]
            atl15_10k = rioxarray.merge.merge_datasets(ds_l_10)
            for key in list(atl15_10k.keys()):
                atl15_10k[key].rio.write_nodata(np.nan, inplace=True)
        ############################### 

        ###############################
        ### do the remapping ##########

        ### bilinear ###
        if remap_type == 'bilinear':
            ds_M2_10k = ((dsALL.rename({'lat':'y','lon':'x'}).rio.write_crs(4326,inplace=True).rio.set_spatial_dims(x_dim="x", y_dim="y",
                    inplace=True).rio.write_coordinate_system(inplace=True)).rio.reproject_match(atl15_10k,resampling=Resampling.bilinear))
            
            if lapserate:
                ds_M2_T = dsALL[['TS','TScalc','T2M']]
                T_interp = self.T_lapserate(self,ds_M2_T,atl15_10k,icesheet)
                for _kk, _vv in T_interp.items():
                    ds_M2_10k[f'{_vv}_i'] = (['y','x','time'],_vv)
                    ds_M2_10k[f'{_vv}_i'] = ds_M2_10k[f'{_vv}_i'].T.astype('float32')
            
            fn_out_bil_remap = Path(out_path,f'netCDF/{freq}',f'MERRA2_{icesheet}_{freq}_{YY}_10Kgrid_bilinear.nc')
            ds_M2_10k.to_netcdf(fn_out_bil_remap) # save just the non-conserved, resampled variables 
        ### end bilinear ###

        ### conservative remap ###
        ### not recommended!
        elif remap_type == 'conservative': 
            save_noc = False

            out_full_con      = Path(out_path,f'netCDF/{freq}',f'MERRA2_{icesheet}_{freq}_{YY}_convars_M2grid.nc')
            out_full_con_rem  = Path(out_path,f'netCDF/{freq}',f'MERRA2_{icesheet}_{freq}_{YY}_convars_10Kgrid.nc')
            out_full_noc_rem  = Path(out_path,f'netCDF/{freq}',f'MERRA2_{icesheet}_{freq}_{YY}_nocvars_10Kgrid.nc')
            out_full_remapped = Path(out_path,f'netCDF/{freq}',f'MERRA2_{icesheet}_{freq}_{YY}_10Kgrid_conservative.nc')

            conservative_fields = ['SMELT', 'EFLUX', 'EVAP', 'HFLUX', 'LWGAB', 'PRECCU', 'PRECLS', 'PRECSN', 'SMELT', 'SWGDN', 'SWGNT','SWGUP']
            non_con_fields = ['ALBEDO', 'TS', 'TScalc', 'HLML', 'T2M','EMIS_eff']

            if not os.path.exists(out_full_con):
                dsALL[conservative_fields].to_netcdf(out_full_con) #save netcdf (just the variables to be conservatively remapped.)
        
            ### non-conservative fields 
            ### optionally saved in their own netCDF
            ### bilinear resample for these.
            if os.path.exists(out_full_noc_rem):
                ds_M2_10k = xr.open_dataset(out_full_noc_rem)
            else:
                ds_M2_10k = ((dsALL[non_con_fields].rename({'lat':'y','lon':'x'}).rio.write_crs(4326,inplace=True).rio.set_spatial_dims(x_dim="x", y_dim="y",
                            inplace=True).rio.write_coordinate_system(inplace=True)).rio.reproject_match(atl15_10k,resampling=Resampling.bilinear))
                
                if lapserate:
                    ds_M2_T = dsALL[['TS','TScalc','T2M']]
                    T_interp = self.T_lapserate(self,ds_M2_T,atl15_10k,icesheet)
                    for _kk, _vv in T_interp.items():
                        ds_M2_10k[f'{_vv}_i'] = (['y','x','time'],_vv)
                        ds_M2_10k[f'{_vv}_i'] = ds_M2_10k[f'{_vv}_i'].T.astype('float32')            
            
                if save_noc:
                    ds_M2_10k.to_netcdf(out_full_noc_rem) # save just the non-conserved, resampled variables
        
            if os.path.exists(out_full_con_rem): # need to delete previous conservative remap effort because a previous failed run will have created an incomplete netCDF.
                try:
                    os.remove(out_full_con_rem)
                except OSError:
                    print(f'Failed to delete {str(out_full_con_rem)}') 

            tic = time.time()
        
            try:
                ### Use CDO tools to do the conservative remap
                ### the gridfile needs to be created before this will work.
                ### details on how to do that are in the pipeline notes. 
                sub.call(["cdo",f"remapcon,/discover/nobackup/cdsteve2/ATL_masschange/downscale/ATL15_10km_{icesheet}_gridfile.txt",out_full_con,out_full_con_rem])
                # sub.call(["cdo","remapcon,/gpfsm/dnb33/cdsteve2/ATL_masschange/downscale/ATL15_{}_10km_gridfile_edited.txt",out_full_con,out_full_con_rem])
                print('conservative remapping success')
            except:
                print('conservative remapping failed')

            toc = time.time()
            print(f'conservative remap took {toc-tic} seconds')

            with xr.open_dataset(out_full_con_rem) as ds_con_rem:
                ds_full = ds_con_rem.merge(ds_M2_10k)
                ds_full.to_netcdf(out_full_remapped)
                print(f'Complete. Output file is: {out_full_remapped}')
            
        telap = (time.time()-tnow)/60
        print('iteration time:', telap)
        tnow=time.time()              

if __name__ == '__main__':
    
    remap_type='bilinear'
    icesheet='AIS'
    print(f'ice sheet is {icesheet}')
    freq='4h'

    if remap_type=='conservative':
        cmd = 'module list'
        pp = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0]
        if 'cdo' not in str(pp):
            print('cdo not loaded. Conservative remapping will not happen. exiting.')
            sys.exit()
    
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

    out_path = pathlib.Path(os.getenv('NOBACKUP'),f'climate/MERRA2/{icesheet}')
    if os.path.exists(out_path):
        pass
    else:
        os.makedirs(out_path)
    # allyears = np.arange(1980,2025) # years to resample
    year_list = np.genfromtxt('M2_years.txt')
    ykey = int(sys.argv[1])
    YY = int(year_list[ykey])
    Mc = MERRA_remap()
    Mc.M2remap(icesheet,LLbounds,out_path,YY,freq=freq)
    print(f'Done with making yearly files with {freq} resolution.')