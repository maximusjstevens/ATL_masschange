#!/usr/bin/env python

'''
dask_rechunk.py
===============

This script rechunks the gridded zarr on Azure. 

The zarr produced with the gridding script has chunking 
of x=1,y=1,time=-1, which is not efficient chunking for 
e.g. moving the file around. This script uses the rechunker
package to rechunk the zarr file. The resultant chunk sizes
seem to be a reasonalbe size based on optimal chunk size documentation
I could find.

This script also controls the slurm operation (to use a node effectively)

run using:
>>> python dask_rechunk.py

(probably should use a screen session to keep the terminal free)

Running for AIS takes ~2.5 hours.
'''

from dask_jobqueue import SLURMCluster
from rechunker import rechunk
import dask.array as dsa
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import zarr
from pathlib import Path


def zchunk(dataset, icesheet):
    cluster = SLURMCluster(
    job_extra_directives=['--partition=hbv2','-o logs/out_dask','-e logs/err_dask'],
    processes=20,
    cores=120,
    memory='456GB',
    walltime='03:00:00')
    cluster.scale(1) #the number of nodes to request

    print(cluster.job_script())

    client = Client(cluster)

    if dataset=='5d':
        timechunk = 3288
    elif dataset=='1d':
        timechunk=2557

    if icesheet=='AIS':
        xchunk = 21
        ychunk = 233
    elif icesheet=='GrIS':
        xchunk = 100 # need to figure these out
        ychunk = 100

    target_chunks = {
            "FAC": {"time": timechunk, "x": xchunk, "y": ychunk},
            "RAIN": {"time": timechunk, "x": xchunk, "y": ychunk},
            "RUNOFF": {"time": timechunk, "x": xchunk, "y": ychunk},
            "SMB": {"time": timechunk, "x": xchunk, "y": ychunk},
            "SMB_a": {"time": timechunk, "x": xchunk, "y": ychunk},
            "SMELT": {"time": timechunk, "x": xchunk, "y": ychunk},
            "SNOWFALL": {"time": timechunk, "x": xchunk, "y": ychunk},
            "TS": {"time": timechunk, "x": xchunk, "y": ychunk},
            "SMB_RCI": {"x": xchunk, "y": ychunk},
            "time": None,  # don't rechunk this array
            "reference_time": None,  # don't rechunk this array
            "x": None,
            "y": None,
        }
    max_mem = "60MB"
       
    ff = Path(f'/shared/home/cdsteve2/firnadls/CFM_gridded/CFM_gridded_{icesheet}_{dataset}.zarr/')
    sg = zarr.open_consolidated(ff)
    target_store = f"/shared/home/cdsteve2/firnadls/CFM_gridded/CFM_gridded_{icesheet}_{dataset}_rechunked.zarr"
    temp_store = f"/shared/home/cdsteve2/firnadls/CFM_gridded/CFM_gridded_{icesheet}_{dataset}_tmp.zarr"

    array_plan = rechunk(sg, target_chunks, max_mem, target_store, temp_store=temp_store)
    with ProgressBar():
        array_plan.execute()

if __name__=='__main__':
    icesheet='AIS'
    dataset = '5d'
    zchunk(dataset,icesheet)
