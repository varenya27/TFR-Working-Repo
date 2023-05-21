'''
run mcmc to find best fit params given the data using L1 and L2
'''
L = 2

import emcee
import numpy as np
from matplotlib import pyplot as plt# import pyplot from matplotlib
import time               # use for timing functions
import corner
from extract import extract_data,extract_direct,extract_rad
import pandas as pd

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

logMbar_sers,logMbar_err_sers, logMstar_sers,logMstar_err_sers,logVRout,logVRout_err=extract_data()
Mbar,Mbar_err,Mstar,Mstar_err,Mgas,Mgas_err,VRout,VRout_err = extract_direct()
Rgas,Rgas_err,Re,Re_err = extract_rad()

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
    if L==1:
        return lp + loglikelihood1(theta, y, x, err_y, err_x)
    return lp + loglikelihood2(theta, y, x, err_y, err_x)
    

def lnprob_vertical(x, y_arr, x_arr, err_y_arr,  err_x_arr,):
    #Likelihood function for vertical scatter
    slope, intercept, sigma = x[0], x[1], x[2]
    ndim = 3
    min_ = -10.
    max_ = 10.
    min_scat = -5.
    max_scat = 2.
    if sigma<0: return -1.e300
    if np.log(sigma) < -5. or np.log(sigma)>2. or slope < -10 or slope > 10 or intercept < -10 or intercept > 10:
        return -1.e300
    # Expected M for given V
    mean = intercept + slope * np.array(x_arr)  
    # Difference between point and line in vertical direction
    dist = np.array(y_arr) - mean
    # Sum up in quadrature contributions to scatter in vertical direction
    scatter = np.sqrt(np.array(err_y_arr)*np.array(err_y_arr) + slope*slope*np.array(err_x_arr)*np.array(err_y_arr) + sigma*sigma)
    chi_sq = dist*dist / (scatter*scatter)  
    # Define Gaussian likelihood
    L = np.exp(-chi_sq/2.) / (np.sqrt(2.*np.pi) * scatter)
    if np.min(L) < 1.e-300:
        return -1.e300
    return np.sum(np.log(L))

Nens = 500 # number of ensemble points
ndims = 3
Nburnin = 500  #500 # number of burn-in samples
Nsamples = 3000  #500 # number of final posterior samples

y, err_y, x, err_x= logMstar_sers,logMstar_err_sers,logVRout,logVRout_err
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
    # range=[(1.4,3), (3.7,7), (0.13,0.2)],
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
figure.savefig(f'corner{L}_STFR.png',format='png', dpi=300)
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


df = pd.read_csv('past_stfr.txt',sep=' ')
paper_stfr = df['paper']
m_past_stfr=df['m'].to_numpy()
b_past_stfr=df['b'].to_numpy()
# b_past_stfr = m_past_stfr/zero_past_stfr
c=['teal','orange','red','blue','green']
for i in range(len(m_past_stfr)):
    y_line = m_past_stfr[i]*x_line+b_past_stfr[i]
    plt.plot(x_line, y_line,'-',color=c[i],linewidth=2,label=paper_stfr[i])



plt.ylim(8.0, 12.0)
plt.xlim(min(x)-0.5,max(x)+0.5)
X,Y= x,y
Xerr,Yerr =err_x, err_y
Y1,Y1_e,X1,X1_e = logMstar_sers,logMstar_err_sers,logVRout,logVRout_err
# Y1,Y1_e,X1,X1_e = logMbar_sers,logMbar_err_sers,logVRout,logVRout_err
plt.errorbar(X1, Y1,Y1_e,X1_e, alpha=0.5, fmt='h', ms=5, mfc='white',color='#242424', mew=1, ecolor='grey', capsize=2.0, zorder=0, label='Individual Data');
# plt.errorbar(X1, Y1, fmt='o', ms=5, color='orange', mfc='orange', mew=1, ecolor='gray', alpha=0.5, capsize=2.0, zorder=0, label='Individual Data');
# plt.errorbar(X, Y, xerr=Xerr, yerr=Yerr, fmt='h', ms=10, color='orangered', mfc='orange', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=5, label='Binned Data');
# plt.errorbar(X, Y, fmt='h', ms=12, color='orangered', mfc='none', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=6);
plt.scatter(X1, Y1,c=Mgas/Mbar,marker='H',cmap='magma',edgecolors='#242424',alpha=0.75, )
# plt.colorbar(label='$R_e$')
plt.colorbar(label='$f_{gas}$' if True else '$R_e$')
plt.tick_params(direction='inout', length=7, width=2)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.title('m={}, b={}'.format(round(line[0],2),round(line[1],2)))
plt.xlabel('$log V_{R_{out}}$')
ylabel = '$log M_{star}$'
plt.ylabel(ylabel)
# plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
plt.legend()
plt.title(f'Likelihood {L} | STFR')
plt.savefig(f'besfit{L}_STFR.png')
plt.show()
