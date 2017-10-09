#!/usr/bin/python
import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np
__author__ = "maxtom"
__email__  = "hitmaxtom@gmail.com"

def getScans(velo_files):
    """Helper method to parse velodyne binary files into a list of scans."""
    scan_list = []
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        scan_list.append(scan.reshape((-1, 4)))

    return scan_list

def getScan(velo_file):
    """Helper method to parse velodyne binary files into a list of scans."""
    scan = np.fromfile(velo_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan
