# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pickle
import os
import glob
import numpy as np
import json
import pandas as pd


# Consolidating features
pkl = "../data/pkl"
labels="../data/labels"


filenames= glob.glob(pkl+"/*.pickle" )
filenames = [os.path.basename(x) for x in filenames]