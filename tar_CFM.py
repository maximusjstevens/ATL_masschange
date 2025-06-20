#!/usr/bin/env python

'''
tar_CFM.py

Script to create a tarball with CFM results directories.

Use with a slurm script to parallelize.

This uses a looping scheme, as running this script
for a single file is inefficient (too much overhead)

when running with CFM_tar_azure.j, 
use sbatch --array=0-99 CFM_tar_azure.j
'''

import subprocess
from pathlib import Path
import sys
import time
import numpy as np

def create_tar_gz(dkey):
    tardir = '/shared/home/cdsteve2/firnadls/tarballs/'
    rname = f'CFMresults_{dkey}_GSFC2020_LW-EMIS_eff_ALB-M2_interp'
    tar_name = tardir + rname + '.tar.gz'
    source_dir = '/shared/home/cdsteve2/firnadls/CFM_outputs/' + rname
    command = ['tar', '-czvf', tar_name, source_dir]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully created {tar_name}")
    else:
        print(f"Error creating {tar_name}.tar.gz: {result.stderr}")

if __name__ == '__main__':
    # dkey = int(sys.argv[1]) # this is the array value
    tic=time.time()
    ii = int(sys.argv[1])
    print('ii',ii)
    istart = ii*200
    iend = istart + 200
    if iend == 20000:
        iend = 20013
    iarray = np.arange(istart,iend)

    for jj,vv in enumerate(iarray):
        dkey = vv
        create_tar_gz(dkey)
        # grid_CFM(fname, vv) # call function to open CFM results file, process results, write to zarr
        # if np.mod(jj,20)==0:
        check_time = time.time()-tic
        print(f'check_time ({jj}) = {check_time}')
    print('run time =' , time.time()-tic , 'seconds')