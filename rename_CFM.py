#!/usr/bin/env python

'''
rename_CFM.py

This script is used to rename CFM results directories and json files.

It was built specifically to deal with my azure runs, when the 
directory names referred to a pixel number from a list of pixels to run.
When I did additional runs to fill in missing pixels, the numbering restarted at
zero. This will assign pixel numbers from a different "pixel to run" csv list.

TAKES HEAVY EDITING OF FILENAMES AND PATHS TO WORK. DOUBLE CHECK TO GET IT RIGHT!
'''

import numpy as np 
import pandas as pd
import io
from pathlib import Path
import json
import os
import traceback

def rename_CFM():
    
    ### lists of pixels. Change as needed
    px_dest = pd.read_csv('pixels_to_run/IS2_pixelstorun_AIS_A1_add.csv') # new numbers (we will remap to these) 
    px_source = pd.read_csv('pixels_to_run/IS2_pixelstorun_AIS_A1_add_2.csv') # current numbering

    ### keep track of indices in the dataframes before merge
    px_dest['_i_dest'] = px_dest.index 
    px_source['_i_source'] = px_source.index

    ### merge dataframes, and select pixels that are in both lists 
    df_all = px_dest.merge(px_source.drop_duplicates(), on=['x','y'], how='left', indicator=True)
    
    df_sel = df_all[df_all['_merge']=='both'].copy().reset_index()
    df_sel['_i_dest'] = df_sel['_i_dest'].astype(int)
    df_sel['_i_source'] = df_sel['_i_source'].astype(int)

    ### path with directory/json names that need to be changed.
    p_tochange = Path('/discover/nobackup/cdsteve2/ATL_masschange/CFMoutputs/AIS_A1_add_2')

    for ii,rw in df_sel.iterrows():
        # print(rw['_i_x'])
        # print(rw['_i_y'])
        try:
            ### change below as needed!
            old_dir = f'CFMresults_A1_{ii}_GSFC2020_LW-EMIS_eff_ALB-M2_interp'
            old_json = f'CFMconfig_AIS_{ii}_GSFC2020_LW-EMIS_eff_ALB-M2_interp.json'

            ### just a check that we are dealing with the correct pixel
            configName = Path(p_tochange,old_dir,old_json)
            
            with open(configName, "r") as f:
                jsonString      = f.read()
                c          = json.loads(jsonString)    
            x_val_j = c['x_val']
            y_val_j = c['y_val']
        
            xM = rw['x']
            yM = rw['y']
        
            if ((xM!=x_val_j) or (yM!=y_val_j)):
                print('x or y values do not match')
                print('not renaming.')
                print(f'x_val: {x_val_j}, xM: {xM}')
                print(f'y_val: {y_val_j}, yM: {yM}')
            else:
                new_dir = f'CFMresults_A1_{rw["_i_x"]}_GSFC2020_LW-EMIS_eff_ALB-M2_interp'
                new_json = f'CFMconfig_AIS_{rw["_i_x"]}_GSFC2020_LW-EMIS_eff_ALB-M2_interp.json'
                
                os.rename(Path(p_tochange,old_dir,old_json), Path(p_tochange,old_dir,new_json))
                os.rename(Path(p_tochange,old_dir), Path(p_tochange,new_dir))
        except Exception:
            print(f'Error with {ii}')
            print(rw)
            traceback.print_exc()

if __name__=='__main__':
    rename_CFM()
    
    