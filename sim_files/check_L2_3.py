likelihood__ = 2
noise__ = 3

'''
running sims for different noise/likelihood values
'''
import pandas as pd
import emcee
import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib
import time               # use for timing functions
import corner
def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

min_ = -5.
max_ = 12.
log_min_scat = -5.
log_max_scat = 0.
min_scat = 0.
max_scat = 1.

def logprior(theta):
    lp = 0.
    m, c, scat_int = theta

    if min_<m<max_ and min_<c<max_ and min_scat<scat_int<max_scat:
        return 0
    else: 
        return -np.inf

def loglikelihood(theta, y, x, err_y, err_x):
    if(likelihood__==1):
        #simple likelihood L1:
        m, c, sigma_int = theta
        sigma2 = err_y**2+(m*err_x)**2 + sigma_int**2
        md = straight_line(theta,x)

        return  -0.5 * np.sum( (y-md)**2/sigma2 + np.log(sigma2))
    if(likelihood__==2):
        #Lelli/Tian likelihood L2:
        m, c, sigma_int = theta
        sigma2 = (m**2*err_x**2)/(m**2+1)+(m**2*err_y**2)/(m**2+1)+sigma_int**2
        md = straight_line(theta,x)
        delta = ( (y-md)**2) / (m**2+1)
        return -0.5 * np.sum(np.log(2*np.pi*sigma2)+(delta/(sigma2)))

def loglikelihood1(theta, y, x, err_y, err_x):
    #simple likelihood L1:

    m, c, sigma_int = theta
    sigma2 = err_y**2+(m*err_x)**2 + sigma_int**2
    md = straight_line(theta,x)

    return  -0.5 * np.sum( (y-md)**2/sigma2 + np.log(sigma2))

def loglikelihood2(theta, y, x, err_y, err_x):
    #Lelli/Tian likelihood L2:

    m, c, sigma_int = theta
    sigma2 = (m**2*err_x**2)/(m**2+1)+(m**2*err_y**2)/(m**2+1)+sigma_int**2
    md = straight_line(theta,x)
    delta = ( (y-md)**2) / (m**2+1)
    return -0.5 * np.sum(np.log(2*np.pi*sigma2)+(delta/(sigma2)))

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) # get the prior
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta, y, x, err_y, err_x)

label = 'btfr'

m={'btfr':4, 'stfr':3.6} #slope
b={'btfr':2.3, 'stfr':1.5} #intercept

y_err = {'btfr':0.3121, 'stfr':0.2001} #median logMbar,logMstar errors
x_err  = 0.0899 #median logv error
x_noise=[0.05,0.10,0.2,.4,.7]
y_noise=[0.12,0.25,0.5,.75,.9]
N = 250

results_m,results_b,results_me,results_be=[],[],[],[]

Nens = 100 #300 # number of ensemble points
ndims = 3
Nburnin = 500  #500 # number of burn-in samples
Nsamples = 3000  #500 # number of final posterior samples
for k in range(1000):
    print('iteration:',k+1)

    x = (.5*np.random.rand(N))+1.8 #generating x axis points
    y = m['btfr'] * x + b['btfr']
    print(np.min(y), np.max(y))
    print(np.min(x), np.max(x))

    x += x_noise[noise__]*np.random.uniform(-1,1,size=N)
    # x += x_noise[i]*np.random.randn(N)
    y += y_noise[noise__]*np.random.uniform(-1,1,size=N)
    # y += y_noise[i]*np.random.randn(N)

    
    i=int(N/2) 
    # print("{:f} {:f} {:f} {:f}".format(y[i],err_y[i],x[i],err_x[i]))
    # err_y = np.array([y_err['btfr']] * N)
    # err_x = np.array(x_err* N)
    err_y = np.random.rand(N)*y_err['btfr']
    err_x = np.random.rand(N)*x_err

 
    argslist = (y, x, err_y, err_x)#normal
    # argslist = (y, x, err_y,)#chi
    p0 = []
    for i in range(Nens):
        pi = [
            np.random.uniform(min_,max_), 
            np.random.uniform(min_,max_),
            np.random.uniform((min_scat), (max_scat))]
        p0.append(pi)

    # set up the sampler    
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
    # pass the initial samples and total number of samples required
    t0 = time.time() # start time
    sampler.run_mcmc(p0, Nsamples + Nburnin,progress=True);
    t1 = time.time()

    timeemcee = (t1-t0)
    print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

    # extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    #plots
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    m_final = np.percentile(flat_samples[:, 0], [16, 50, 84])[1]
    c_final = np.percentile(flat_samples[:, 1], [16, 50, 84])[1]
    scat_final = np.percentile(flat_samples[:, 2], [16, 50, 84])[1]
    Med_value = [m_final,c_final,scat_final]

    line,hi,lo=[],[],[]
    labels=['slope = ','intercept = ','intrinsic scatter = ',]
    for i in range(ndims):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        # print(round(mcmc[1],3), round(q[0],3), round(-q[1],3))
        # print(labels[i],round(mcmc[1],3), round(q[0],4), round(-q[1],4))
        line.append(mcmc[1])
        hi.append(q[0])
        lo.append(q[1])

    results_b.append(line[1])
    results_m.append(line[0])
    results_me.append(np.sqrt(hi[0]**2+lo[0]**2))
    results_be.append(np.sqrt(hi[1]**2+lo[1]**2))

    with open('results_L{}_{}.txt'.format(likelihood__,noise__),'a') as fp:
        fp.write(str(results_m[k])+' '+str(results_me[k])+' '+str(results_b[k])+' '+str(results_be[k])+'\n')
