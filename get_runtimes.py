#!/usr/bin/env python

import numpy as np
from pathlib import Path
import os
import sys
import re
import pandas as pd

xytime = np.zeros((20013,3))

for pix in range(20013):
    rtime = np.nan
    xx = np.nan
    yy = np.nan
    logfile = f'logs/container_out_10416_{pix:04d}'
    if os.path.exists(logfile):
        with open(logfile) as f:
            for ll, line in enumerate(f):
                if ll==2:
                    xx = float(line.split(' ')[-1])
                    yy = float(line.split(' ')[-2])
                if line.startswith('run time'):                
                    rtime = float(re.findall(r'\d+',line)[0])
    xytime[pix,:] = xx,yy,rtime
df = pd.DataFrame(xytime,columns=['x','y','rtime'])
df.to_csv('dftest.csv',na_rep='NaN')
