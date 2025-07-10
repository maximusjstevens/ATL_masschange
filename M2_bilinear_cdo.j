#!/bin/bash

#SBATCH --nodes=1
#SBATCH -o logs/container_out_%04A_%04a
#SBATCH -e logs/container_err_%04A_%04a
#SBATCH --time=0:30:00
#SBATCH --account=s2656


# MERRA_concat_discover_masschange_array.j
# This should be a one and done script to run the MERRA_concat_discover_masschange_array.py
# code. Run using:
# >>> sbatch --array=0-4 make_zarr.j
# (I did not test the log functionality yet.)
# old budget: s2441
# removed: SBATCH --ntasks-per-node=1

module load python/GEOSpyD/Min4.11.0_py3.9
module load cdo

source /usr/local/other/python/GEOSpyD/4.11.0_py3.9/2022-04-28/etc/profile.d/conda.sh
conda activate MSpy311

ii=$SLURM_ARRAY_TASK_ID

echo $ii
srun -N1 -n1 -c1 python M2_resample_bil.py $ii
