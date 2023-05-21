'''
plot the BTFR and STFR data from the kross/kmoss data
'''
import numpy as np
from matplotlib import pyplot as plt# import pyplot from matplotlib
import pandas as pd 

def Mstar_charateristic(R, Re, Mtot,R_err,Re_err,Mtot_err): #freeman1977
    '''
    R is the radii within which you want a certain mass
    i.e. R=Rout
    '''
    Rd=0.59*Re
    Rd_err=0.59*Re_err
    Mr = Mtot*(1- ((1+(R/(Rd)))* np.exp(-(R/Rd))))
    Mr_err = Mtot_err*(1- ((1+(R/(Rd)))* np.exp(-(R/Rd))))+R_err*Mtot*(np.exp(-(R/Rd))*R/(Rd**2))-Rd_err*Mtot*(np.exp(-(R/Rd))*R**2/Rd**3)
    return [Mr,Mr_err] 

def MGas_charateristic(R, Rgas, Mtot,R_err,Rgas_err,Mtot_err): #freeman1977
    '''
    R is the radii within which you want a certain mass
    R=Rout
    '''
 
    Mr = Mtot*(1- ((1+(R/(Rgas)))* np.exp(-(R/Rgas))))
    Mr_err = Mtot_err*(1- ((1+(R/(Rgas)))* np.exp(-(R/Rgas))))+R_err*Mtot*(np.exp(-(R/Rgas))*R/(Rgas**2))-Rgas_err*Mtot*(np.exp(-(R/Rgas))*R**2/Rgas**3)
    return [Mr,Mr_err] 


file='data\KROSSKMOS_2023_UPDATED.csv'
df=pd.read_csv(file)

# Name,logRgas_lerr,logRgas,logRgas_uerr,Re,Re_err,Mstar,Mstar_err,MH2,MH2_err,MHI,MHI_err,Mgas,Mgas_err,Mbar,Mbar_err,VRout,VRout_err,Rgas,Rgas_err

Name = df['Name'].to_numpy()
N = len(Name)
Mgas,Mgas_err,Mbar,Mbar_err=df['Mgas'].to_numpy(),df['Mgas_err'].to_numpy(),df['Mbar'].to_numpy(),df['Mbar_err'].to_numpy()
VRout,VRout_err=df['VRout'].to_numpy(),df['VRout_err'].to_numpy()
Rgas,Rgas_err=df['Rgas'].to_numpy(),df['Rgas_err'].to_numpy()
Re,Re_err=df['Re'].to_numpy(),df['Re_err'].to_numpy()
Mstar,Mstar_err=df['Mstar'].to_numpy(),df['Mstar_err'].to_numpy()

# MH2,MH2_err=df['MH2'].to_numpy(),df['MH2_err'].to_numpy()
# MHI,MHI_err=df['MHI'].to_numpy(),df['MHI_err'].to_numpy()

RH2, RH2_err = Rgas, Rgas_err
RHI, RHI_err = 2*Rgas, 2*Rgas_err
Rout, Rout_err = 0.59*3.2*1.5*(Re),0.59*3.2*1.5*(Re_err)

# MH2_sers,MH2_err_sers = np.empty(N),np.empty(N)
# MHI_sers,MHI_err_sers = np.empty(N),np.empty(N)
Mstar_sers,Mstar_err_sers = np.empty(N),np.empty(N)
Mgas_sers,Mgas_err_sers = np.empty(N),np.empty(N)
# print(MH2,MH2_err)
# quit()
for i in range(N):
    # mh2 = MGas_charateristic(Rout[i], RH2[i], MH2[i], Rout_err[i], RH2_err[i], MH2_err[i])
    # MH2_sers[i]=mh2[0]
    # MH2_err_sers[i]=mh2[1]

    # mh1 = MGas_charateristic(Rout[i], RHI[i], MHI[i], Rout_err[i], RHI_err[i], MHI_err[i])
    # MHI_sers[i]=mh1[0]
    # MHI_err_sers[i]=mh1[1]
    
    mgas = MGas_charateristic(Rout[i], Rgas[i], Mgas[i], Rout_err[i], Rgas_err[i], Mgas_err[i])
    Mgas_sers[i]=mgas[0]
    Mgas_err_sers[i]=mgas[1]
    
    mstar = Mstar_charateristic(Rout[i], Re[i], Mstar[i], Rout_err[i], Re_err[i], Mstar_err[i])
    Mstar_sers[i]=mstar[0]
    Mstar_err_sers[i]=mstar[1]

# Mgas_sers, Mgas_err_sers = MH2_sers+MHI_sers,np.sqrt(MH2_err_sers**2+MHI_err_sers**2)
Mbar_sers, Mbar_err_sers = Mgas_sers+Mstar_sers,np.sqrt(Mgas_err_sers**2+Mstar_err_sers**2)

logVRout,logMbar_sers,logMbar_err_sers,logVRout_err=np.log10(VRout), np.log10(Mbar_sers),Mbar_err_sers/2.303/Mbar_sers,VRout_err/2.303/VRout
logMstar_sers,logMstar_err_sers=np.log10(Mstar_sers),Mstar_err_sers/2.303/Mstar_sers

def extract_direct():
    return [Mbar,Mbar_err,Mstar,Mstar_err,Mgas,Mgas_err,VRout,VRout_err]

def extract_data():
    return [logMbar_sers,logMbar_err_sers, logMstar_sers,logMstar_err_sers,logVRout,logVRout_err]

def extract_rad():
    return [Rgas,Rgas_err,Re,Re_err]

def kross():
    return []

def kmos():
    return []