PACKAGE_AUTHOR = "Student"
PACKAGE_VERSION = 1.0
print("Initializing BACKTEST Package")
import feather
import numpy as np
import pandas as pd
import os
from BackTest.Preprocessing import update
if not os.path.exists('./data/data.h5'):
    update()

