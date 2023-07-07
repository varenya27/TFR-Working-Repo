'''
run mcmc to find best fit params given the data using L1 and L2
'''
data = ['COMBINED']

import emcee
import numpy as np
from matplotlib import pyplot as plt# import pyplot from matplotlib
import time               # use for timing functions
import pandas as pd
import pickle 

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['mathtext.default'] = 'regular'

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

min_ = -2.
max_ = 12.
log_min_scat = -5.
log_max_scat = 0.
min_scat = -0.
max_scat = 1.

def logprior(theta):
    lp = 0.
    m, c, scat_int = theta
    if min_<m<max_ and min_<c<max_ and min_scat<scat_int<max_scat:
        return 0
    else: 
        return -np.inf

def loglikelihood1(theta, y, x, err_y, err_x):
    # likelihood L1 vertical:
    m, c, sigma_int = theta
    #    sigma_int=10**logsigma_int
    sigma2 = err_y**2+(m*err_x)**2 + sigma_int**2
    #    sigma2 = err_y**2  + sigma_int**2 
    md = straight_line(theta,x)
    return  -0.5 * np.sum( (y-md)**2/sigma2 + np.log(sigma2))

def loglikelihood2(theta, y, x, err_y, err_x):
    # likelihood L2 orthogonal:
    m, c, sigma_int = theta
    sigma2 = (m**2*err_x**2)/(m**2+1)+(m**2*err_y**2)/(m**2+1)+sigma_int**2
    md = straight_line(theta,x)
    delta = ( (y-md)**2) / (m**2+1)
    return -0.5 * np.sum(np.log(2*np.pi*sigma2)+(delta/(sigma2)))

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) # get the prior
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood2(theta, y, x, err_y, err_x)
    
for L in data:
    f = 'data/'+L+'_FDM_cat.csv' 

    df = pd.read_csv(f)
    df=df.drop(df[df.Redshift>1].index)

    VRout,VRout_err,Mbar_Rout,Mbar_Rout_err,Mstar_Rout,Mstar_Rout_err = df.VRout.to_numpy(),df.VRout_err.to_numpy(),df.Mbar_Rout.to_numpy(),df.Mbar_Rout_err.to_numpy(),df.Mstar_Rout.to_numpy(),df.Mstar_Rout_err.to_numpy()
    Mgas_Rout,Mgas_Rout_err = Mbar_Rout-Mstar_Rout,Mbar_Rout_err-Mstar_Rout_err 
    logMbar = np.log10(Mbar_Rout)
    logMstar = np.log10(Mstar_Rout)
    logVRout = np.log10(VRout)
    logMbar_err = Mbar_Rout_err/(2.303*Mbar_Rout)
    logMstar_err = Mstar_Rout_err/(2.303*Mstar_Rout)
    logVRout_err = VRout_err/(2.303*VRout)

    y, err_y, x, err_x= logMstar,logMstar_err,logVRout,logVRout_err


    Nens = 300 # number of ensemble points
    ndims = 3
    Nburnin = 500  #500 # number of burn-in samples
    Nsamples = 3000  #500 # number of final posterior samples

    argslist = (y, x, err_y, err_x)
    p0 = []
    for i in range(Nens):
        pi = [
            np.random.uniform(min_,max_), 
            np.random.uniform(min_,max_),
            np.random.uniform((min_scat), (max_scat))]
        p0.append(pi)

    # set up the sampler    
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

    t0 = time.time() # start time
    sampler.run_mcmc(p0, Nsamples + Nburnin,progress=True);
    t1 = time.time()

    timeemcee = (t1-t0)
    print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

    # extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    #plots
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    FF = open(f"sampler_{L}_STFR_zless1.pkl", "wb")
    pickle.dump(flat_samples,FF)