'''
run mcmc to find best fit params given the data using L1 and L2
'''
L = 2
full = ['btfr','stfr','kmos3d','kross']
# current = full[2]
# print('running MCMC for', current)
#current = input('Enter dataset: ')
import emcee
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib import pyplot as plt
import time              
import corner
from extract import extract_data,extract_direct,extract_rad,kross,kmos
import pandas as pd 
# import matplotlib
# import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase 


df = pd.read_csv