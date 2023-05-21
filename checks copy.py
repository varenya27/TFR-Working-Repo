noise__ = 2
likelihood__ = 2
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

min_ = -6.
max_ = 12.
log_min_scat = -5.
log_max_scat = 0.
min_scat = 0.
max_scat = 1.

def logprior(theta):
    lp = 0.
    m, c, scat_int = theta

    # if min_<m<max_ and min_<c<max_ and min_scat<scat_int<max_scat:
    if min_<m<max_ and min_<c<max_ and min_scat<(scat_int)<max_scat:
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
        # sigma_int = 10**sigma_int
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


# # Name,logRgas_lerr,logRgas,logRgas_uerr,Re,Re_err,Mstar,Mstar_err,MH2,MH2_err,MHI,MHI_err,Mgas,Mgas_err,Mbar,Mbar_err,VRout,VRout_err,Rgas,Rgas_err
label = 'btfr'
file='data/KMOS3D_KROSS.csv'
df=pd.read_csv(file)


# Name = df['Name'].to_numpy()
Mgas,Mgas_err,Mbar,Mbar_err=df['Mgas'].to_numpy(),df['Mgas_err'].to_numpy(),df['Mbar'].to_numpy(),df['Mbar_err'].to_numpy()
Mstar,Mstar_err=df['Mstar'].to_numpy(),df['Mstar_err'].to_numpy()
VRout,VRout_err=df['VRout'].to_numpy(),df['VRout_err'].to_numpy()
logVRout,logMbar,logMbar_err,logVRout_err=np.log10(VRout), np.log10(Mbar),Mbar_err/2.303/Mbar,VRout_err/2.303/VRout
logMstar,logMstar_err=np.log10(Mstar),Mstar_err/2.303/Mstar

m={'btfr':4, 'stfr':3.6} #slope
b={'btfr':2.3, 'stfr':1.5} #intercept

y_err = {'btfr':0.3121, 'stfr':0.2001} #median logMbar,logMstar errors
x_err  = 0.0899 #median logv error
x_noise=[0.05,0.10,0.2,.4,.7]
y_noise=[0.12,0.25,0.5,.75,.9]
N = 250

results_m,results_b,results_me,results_be=[],[],[],[]
# fp = open('synthdata.txt','a')
# fp2= open('results.txt','a')
Nens = 100 #300 # number of ensemble points
ndims = 3
Nburnin = 500  #500 # number of burn-in samples
Nsamples = 3000  #500 # number of final posterior samples
for k in range(1):
    print('iteration:',k+1)

    x = (.5*np.random.rand(N))+1.8 #generating x axis points
    y = m['btfr'] * x + b['btfr']
    print(np.min(y), np.max(y))
    print(np.min(x), np.max(x))

    # x += x_noise[i]*np.random.randn(N)
    # y += y_noise[noise__]*np.random.uniform(-1,1,size=N)
    y += y_noise[noise__]*np.random.randn(N)

    
    i=int(N/2) 
    # print("{:f} {:f} {:f} {:f}".format(y[i],err_y[i],x[i],err_x[i]))
    # err_y = np.array([y_err['btfr']] * N)
    # err_x = np.array(x_err* N)
    err_y = np.random.rand(N)*y_err['btfr']
    err_x = np.random.rand(N)*x_err

    
    argslist = (y, x, err_y, err_x)#normal
    argslist = (logMbar, logVRout, logMbar_err,logVRout_err)#normal
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

    figure = corner.corner(
        flat_samples,
        # title_fmt=".2E",
        # levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), 
        levels=(0.68,0.95), 
        labels=[r"Slope", r"Intercept", r"Intrinsic Scatter"], 
        # quantiles=[0.16,0.84], 
        # range=[(m_final-m_final/2,m_final+m_final/2), (c_final-c_final/5,c_final+c_final/5), (scat_final-scat_final/2,scat_final+scat_final/2)],
        range=[(1.1,1.9), (6.4,8.), (-5.4,-1)],
        show_titles=True, 
        label_kwargs={"fontsize": 12},
        title_kwargs={"fontsize": 10}
    );

    axes = np.array(figure.axes).reshape((ndims, ndims))
    for i in range(ndims):
        ax = axes[i, i]
        ax.axvline(Med_value[i], color="r")
    for yi in range(ndims):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(Med_value[xi], color="r")
            ax.axhline(Med_value[yi], color="r")
            ax.plot(Med_value[xi], Med_value[yi], "sr")
    # figure.savefig('check_figs/   '+str(k)+'crnr.png',format='png', dpi=300)
    # figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10}, range=[(a_ML-0.4,a_ML+0.4), (b_ML-0.6,b_ML+0.6), (s_ML-0.1,s_ML+0.1)])

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
        # display(Math(txt))

    results_b.append(line[1])
    results_m.append(line[0])
    results_me.append(np.sqrt(hi[0]**2+lo[0]**2))
    results_be.append(np.sqrt(hi[1]**2+lo[1]**2))
    # print(results_b,results_m)
    # print(str(results_m[k])+' '+str(results_b[k]))
    with open('results_L{}_{}.txt'.format(likelihood__,noise__),'a') as fp2:
        fp2.write(str(results_m[k])+' '+str(results_me[k])+' '+str(results_b[k])+' '+str(results_be[k])+'\n')
    w,h=9,7
    plt.figure(figsize=(w, h))
    plt.ylim(6.5, 14)
    plt.xlim(1.325,3.125)

    x_line=np.linspace(0,3.5)
    y_best = line[0]*x+line[1]
    err0 = np.sqrt(np.mean((y_best-y)**2))
    # plt.plot(x, y_best,'-',color='#6d4b4b',linewidth=4)
    # plt.plot(x,y_best-err0,'--',color='#6d4b4b')
    # plt.plot(x,y_best+err0,'--',color='#6d4b4b')
    # plt.fill_between (x, y_best-err0,y_best+err0, color='#54bebe', hatch='\\\\\\\\', alpha=0.5, zorder=0, )
    plt.axline((0, line[1]), slope=line[0],linestyle= '-',color='#6d4b4b',linewidth=3, label='y={}x+{}'.format(str(round(line[0],2)),str(round(line[1],2))))
    plt.axline((0, line[1]+err0), slope=line[0],linestyle= '--',color='#6d4b4b',linewidth=2)
    plt.axline((0, line[1]-err0), slope=line[0],linestyle= '--',color='#6d4b4b',linewidth=2)

    plt.errorbar(x, y,y_err[label],x_err, fmt='h', ms=5, color='#54bebe', mfc='#54bebe', mew=1, ecolor='#76c8c8', alpha=0.5, capsize=2.0, zorder=0, label='Synthetic Data');
    plt.errorbar(logVRout, logMbar,logMbar_err,logVRout_err, fmt='h', ms=5, color='#d7658b', mfc='#d7658b', mew=1, ecolor='#df979e', alpha=0.5, capsize=2.0, zorder=0, label='Real Data (BTFR)');
    plt.axhline(y=8.5, xmin=0.0, xmax=3.5, color='grey',label='$y_i\in[8.5,12],\,\,\, x_i\in[1.5,3]$',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.axhline(y=12, xmin=0.0, xmax=3.5, color='grey',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.axvline(x=1.5, color='grey',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.axvline(x=3, color='grey',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
    plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.set_figheight(6)
# fig.set_figwidth(8)
# fig.suptitle('Deviations')
# ax1.hist(results_m/results_me, 10, density=False)
# ax1.set_title('Slope')
# ax1.set_xlim(0,1)
# ax2.hist(results_b/results_be, 10, density=False)
# ax2.set_title('Intercept')
# plt.savefig('histogram.png')


# plt.show()
