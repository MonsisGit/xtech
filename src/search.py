# usual imports #
import os
import numpy as np
import pandas as pd

# visualization imports #
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread

# consistent plots #
from pylab import rcParams

rcParams['figure.figsize']= 12,5
rcParams['xtick.labelsize']= 12
rcParams['ytick.labelsize']= 12
rcParams['axes.labelsize']= 12

# ignore unwanted warnings #
import warnings
warnings.filterwarnings(action='ignore',message='^internal gelsd')
# designate directory to save the images #

TRAIN_PATH = os.path.join("/data/train")
TEST_PATH = os.path.join("/data/test")