#!/bin/bash

#SBATCH --nodes=1
#SBATCH -o logs/container_out_%04A_%04a
#SBATCH -e logs/container_err_%04A_%04a
#SBATCH --time=1:00:00
#SBATCH --account=s2656
#SBATCH --qos=debug

# This should be a one and done script to run the MERRA_concat_discover.py
# code. Run using:
# >>> sbatch --array=0-44 MERRA_concat_discover.j
# (I did not test the log functionality yet.)
# old budget: s2441
# removed: SBATCH --ntasks-per-node=1

#module load python/GEOSpyD/Min4.11.0_py3.9
module load cdo

source /usr/local/other/python/GEOSpyD/4.11.0_py3.9/2022-04-28/etc/profile.d/conda.sh
conda activate MSpy311

srun -N1 -n1 -c1 python MERRA_makezarr_discover_masschange.py
