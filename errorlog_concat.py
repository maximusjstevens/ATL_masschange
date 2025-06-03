#!/usr/bin/env python

import numpy as np
from pathlib import Path
import os
import sys
import re

mis = np.genfromtxt('missing_res118.txt').astype(int)

# filenames = ['file1.txt', 'file2.txt', ...]
with open('errors.txt', 'w') as outfile:
    for ii,mr in enumerate(mis):
        logfile = f'logs/container_err_10416_{mr:04d}'
        with open(logfile) as infile:
            outfile.write(str(mr)+'\n')
            for line in infile:
                outfile.write(line)
            outfile.write('------------------------------------------\n')
