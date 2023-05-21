'''
run mcmc to find best fit params given the data using L1 and L2
'''
L = 2

import emcee
import numpy as np
from matplotlib import pyplot as plt# import pyplot from matplotlib
import time               # use for timing functions
import corner
from extract import extract_data,extract_direct,extract_rad,kross,kmos


def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

logMbar_sers,logMbar_err_sers, logMstar_sers,logMstar_err_sers,logVRout,logVRout_err=extract_data()
Mbar,Mbar_err,Mstar,Mstar_err,Mgas,Mgas_err,VRout,VRout_err = extract_direct()
Rgas,Rgas_err,Re,Re_err = extract_rad()

kr_bar,kr_bar_e,kr_star,kr_star_e,kr_v,kr_v_e = logMbar_sers[:169],logMbar_err_sers[:169], logMstar_sers[:169],logMstar_err_sers[:169],logVRout[:169],logVRout_err[:169]
km_bar,km_bar_e,km_star,km_star_e,km_v,km_v_e = logMbar_sers[169:],logMbar_err_sers[169:], logMstar_sers[169:],logMstar_err_sers[169:],logVRout[169:],logVRout_err[169:]
min_ = -2.
max_ = 12.
log_min_scat = -5.
log_max_scat = 0.
min_scat = 0.1
max_scat = 1.

def logprior(theta):
    m, c, scat_int = theta
    if min_<m<max_ and min_<c<max_ and min_scat<scat_int<max_scat:
        return 0
    else: 
        return -np.inf

def loglikelihood1(theta, y, x, err_y, err_x):
    # likelihood L1 vertical:
    m, c, sigma_int = theta
    sigma2 = err_y**2+(m*err_x)**2 + sigma_int**2
    #    sigma2 = err_y**2  + sigma_int**2 
    md = straight_line(theta,x)
    return  -0.5 * np.sum( (y-md)**2/sigma2 + np.log(sigma2))

def loglikelihood2(theta, y, x, err_y, err_x):
    # likelihood L2 orthogonal:
    m, c, sigma_int = theta
    # sigma_int=10**sigma_int #if doing log-uniform priors on scatter
    sigma2 = (m**2*err_x**2)/(m**2+1)+(m**2*err_y**2)/(m**2+1)+sigma_int**2
    md = straight_line(theta,x)
    delta = ( (y-md)**2) / (m**2+1)
    return -0.5 * np.sum(np.log(2*np.pi*sigma2)+(delta/(sigma2)))

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) # get the prior
    if not np.isfinite(lp):
        return -np.inf
    if L==1:
        return lp + loglikelihood1(theta, y, x, err_y, err_x)
    return lp + loglikelihood2(theta, y, x, err_y, err_x)

Nens = 300 # number of ensemble points
ndims = 3
Nburnin = 500  #500 # number of burn-in samples
Nsamples = 3000  #500 # number of final posterior samples

y, err_y, x, err_x= km_bar,km_bar_e,km_v,km_v_e
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
    # range=[(1,4), (2.5,7), (0.095,0.13)],
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
figure.savefig(f'corner{L}_KMOS3D.png',format='png', dpi=300)
# figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10}, range=[(a_ML-0.4,a_ML+0.4), (b_ML-0.6,b_ML+0.6), (s_ML-0.1,s_ML+0.1)])

line,hi,lo=[],[],[]
# results = '\n'+v+' at time '+(time.asctime( time.localtime(time.time()) )[11:19])+'\n'+'slope intercept'
labels=['slope = ','intercept = ','intrinsic scatter = ',]
for i in range(ndims):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    # print(round(mcmc[1],3), round(q[0],3), round(-q[1],3))
    print(labels[i],round(mcmc[1],3), round(q[0],4), round(-q[1],4))
    # results+=labels[i]+str(round(mcmc[1],3))+ ' '+str(round(q[0],4))+' '+ str(round(-q[1],4))+'\n'
    line.append(mcmc[1])
    hi.append(q[0])
    lo.append(q[1])
    # display(Math(txt))
# with open('results.txt','a') as f:
#     f.write(results)
plt.figure(figsize=(9,7))
# line[2]=10**line[2]

#bestfit with errors on m,c
# x_line=np.linspace(min(x)-0.5,max(x)+0.5)
# plt.plot(x_line, line[0]*x_line+line[1],'-',color='#303030',linewidth=4,label=f'Best fit: y={round(line[0],2)}x+{round(line[1],2)}')
# plt.plot(x_line, (line[0]+hi[0])*x_line+line[1]+hi[1],'--',color='#303030')
# plt.plot(x_line, (line[0]-lo[0])*x_line+line[1]-lo[1],'--',color='#303030')
# plt.fill_between (x_line, (line[0]+hi[0])*x_line+line[1]+hi[1], (line[0]-lo[0])*x_line+line[1]-lo[1], color='#b3a4a4', hatch='\\\\\\\\', alpha=0.5, zorder=0, )


#bestfit with scatter
x_line=np.linspace(min(x)-0.5,max(x)+0.5)
x_line=np.linspace(min(x)-0.5,max(x)+0.5)
y_line = line[0]*x_line+line[1]
plt.plot(x_line, y_line,'-',color='#303030',linewidth=4,label=f'y={round(line[0],2)}x+{round(line[1],2)}')
plt.plot(x_line, y_line+( 3*line[2]*np.sqrt(line[0]**2+1) ) ,'--',color='#303030')
plt.plot(x_line, y_line-( 3*line[2]*np.sqrt(line[0]**2+1) ),'--',color='#303030')
plt.fill_between (x_line, y_line+( 3*line[2]*np.sqrt(line[0]**2+1) ), y_line-( 3*line[2]*np.sqrt(line[0]**2+1) ), color='#b3a4a4', hatch='\\\\\\\\', alpha=0.5, zorder=0, label='3$\sigma$ scatter')

#plotting data
plt.ylim(8.0, 12.0)
plt.xlim(min(x)-0.5,max(x)+0.5)
X,Y= x,y
Xerr,Yerr =err_x, err_y
plt.errorbar(x, y,err_y,err_x, alpha=1, fmt='h', ms=5, mfc='darksalmon',color='salmon', mew=1, ecolor='darkgrey', capsize=2.0, zorder=0, label='Individual Data');

plt.tick_params(direction='inout', length=7, width=2)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.title('m={}, b={}'.format(round(line[0],2),round(line[1],2)))
plt.xlabel('$log V_{R_{out}}$')
ylabel = '$log M_{bar}$'
plt.ylabel(ylabel)
# plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
plt.legend()
plt.title(f'Likelihood {L} | KMOS3D Data | BTFR')
plt.savefig(f'besfit{L}_KMOS3D.png')
plt.show()
