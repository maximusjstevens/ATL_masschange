#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/container_out_%04A_%04a
#SBATCH -e logs/container_err_%04A_%04a
#SBATCH --time=0:30:00
#SBATCH --account=s2656

# This should be a one and done script to run the MERRA_concat_discover.py
# code. Run using:
# >>> sbatch MERRA_concat_discover.j
# (I did not test the log functionality yet.)
# old budget: s2441

module load python/GEOSpyD/Min24.4.0-0_py3.11
module load cdo

ii=$SLURM_ARRAY_TASK_ID

srun -N1 -n1 -c1 python MERRA2_subsetter_array.py $ii




