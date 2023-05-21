import emcee
import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib as mpl
from matplotlib import pyplot as plt# import pyplot from matplotlib
import time               # use for timing functions
import corner
import pandas as pd 

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

file='data/KMOS3D_KROSS.csv'

df=pd.read_csv(file)

# Name,logRgas_lerr,logRgas,logRgas_uerr,Re,Re_err,Mstar,Mstar_err,MH2,MH2_err,MHI,MHI_err,Mgas,Mgas_err,Mbar,Mbar_err,VRout,VRout_err,Rgas,Rgas_err

Name = df['Name'].to_numpy()
Mgas,Mgas_err,Mbar,Mbar_err=df['Mgas'].to_numpy(),df['Mgas_err'].to_numpy(),df['Mbar'].to_numpy(),df['Mbar_err'].to_numpy()
VRout,VRout_err=df['VRout'].to_numpy(),df['VRout_err'].to_numpy()
Mstar,Mstar_err=df['Mstar'].to_numpy(),df['Mstar_err'].to_numpy()

logVRout,logMbar,logMbar_err,logVRout_err=np.log10(VRout), np.log10(Mbar),Mbar_err/2.303/Mbar,VRout_err/2.303/VRout
logMstar,logMstar_err=np.log10(Mstar),Mstar_err/2.303/Mstar
print(np.median(logMbar_err),np.median(logMstar_err),np.median(logVRout_err))
print(np.min(logVRout),np.max(logVRout))
print(np.min(logMbar),np.max(logMbar))
print(np.min(logMstar),np.max(logMstar))
