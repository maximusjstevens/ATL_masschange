from dask_jobqueue import SLURMCluster
from rechunker import rechunk
import dask.array as dsa
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import zarr


def zchunk(dataset):
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

    target_chunks = {
            "FAC": {"time": timechunk, "x": 21, "y": 233},
            "RAIN": {"time": timechunk, "x": 21, "y": 233},
            "RUNOFF": {"time": timechunk, "x": 21, "y": 233},
            "SMB": {"time": timechunk, "x": 21, "y": 233},
            "SMB_a": {"time": timechunk, "x": 21, "y": 233},
            "SMELT": {"time": timechunk, "x": 21, "y": 233},
            "SNOWFALL": {"time": timechunk, "x": 21, "y": 233},
            "TS": {"time": timechunk, "x": 21, "y": 233},
            "SMB_RCI": {"x": 21, "y": 233},
            "time": None,  # don't rechunk this array
            "reference_time": None,  # don't rechunk this array
            "x": None,
            "y": None,
        }
    max_mem = "60MB"
       
    ff = f'~/firnadls/CFM_gridded/CFM_gridded_AIS_{dataset}.zarr/'
    sg = zarr.open_consolidated(ff)
    target_store = f"firnadls/CFM_gridded/CFM_gridded_AIS_{dataset}_rechunked_c.zarr"
    temp_store = f"firnadls/CFM_gridded/CFM_gridded_AIS_{dataset}_tmp_c.zarr"

    array_plan = rechunk(sg, target_chunks, max_mem, target_store, temp_store=temp_store)
    with ProgressBar():
        array_plan.execute()

if __name__=='__main__':
    
    dataset = '1d'
    zchunk(dataset)
