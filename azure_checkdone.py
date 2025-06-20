import numpy as np
import os
import sys
from pathlib import Path

def checkdone(icesheet, quad=None):
    m_dir_list = []
    m_spin_list = []
    m_res_list = []

    if icesheet=='GrIS':
        _range = 20013
    else:
        if quad == 'A1':
            _range = 35909
            qn = 'A1'
        elif quad == 'A1_add':
            _range = 3614
            qn = 'A1'
        elif quad == 'A1_add_2':
            _range = 554
            qn = 'A1'
        elif quad == 'A2':
            _range = 27283
            qn = 'A2'
        elif quad == 'A3':
            _range = 20130
            qn = 'A3'
        elif quad == 'A4':
            _range = 38900
            qn = 'A4'
        elif quad == 'A4_add':
            _range = 2008
            qn = 'A4'

    for ii in range(_range):
        if quad is not None:
            opath = Path(f'/shared/home/cdsteve2/firnadls/CFM_outputs/{icesheet}_{quad}')
            dd = f'CFMresults_{qn}_{ii}_GSFC2020_LW-EMIS_eff_ALB-M2_interp/'
        else:
            opath = Path(f'/shared/home/cdsteve2/firnadls/CFM_outputs/{icesheet}')
            dd = f'CFMresults_{ii}_GSFC2020_LW-EMIS_eff_ALB-M2_interp/'

        dir_path = Path(opath,dd)
        spin_path = Path(opath,dd,'CFMspin.hdf5')
        res_path = Path(opath,dd,'CFMresults.hdf5')

        if os.path.exists(dir_path):
            pass
        else:
            m_dir_list.append(ii)

        if os.path.exists(spin_path):
            pass
        else:
            m_spin_list.append(ii)

        if os.path.exists(res_path):
            pass
        else:
            m_res_list.append(ii)
    
    if quad is not None:
        np.savetxt(f'missing_dirs_{quad}_0604.txt', m_dir_list, fmt='%i')
        np.savetxt(f'missing_spin_{quad}_0604.txt', m_spin_list, fmt='%i')
        np.savetxt(f'missing_res_{quad}_0604.txt', m_res_list, fmt='%i')   
    else:
        np.savetxt('missing_dirs_GrIS_0604.txt', m_dir_list, fmt='%i')
        np.savetxt('missing_spin_GrIS_0604.txt', m_spin_list, fmt='%i')
        np.savetxt('missing_res_GrIS_0604.txt', m_res_list, fmt='%i')

if __name__=='__main__':
    icesheet='AIS'
    quad = sys.argv[1]
    checkdone(icesheet,quad=quad)
