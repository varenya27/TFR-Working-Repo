'''
plots synthetic data along with the real data for different noise values
'''


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from extract import extract_data,extract_direct, extract_rad
from matplotlib.ticker import AutoMinorLocator
plt.rcParams['axes.labelsize']=16
# matplotlib.use('Agg')
label = 'btfr'

logMbar_sers,logMbar_err_sers, logMstar_sers,logMstar_err_sers,logVRout,logVRout_err=extract_data()
Mbar,Mbar_err,Mstar,Mstar_err,Mgas,Mgas_err,VRout,VRout_err = extract_direct()
Rgas,Rgas_err,Re,Re_err = extract_rad()
m={'btfr':4, 'stfr':3.6} #slope
b={'btfr':2.3, 'stfr':1.5} #intercept

y_err = {'btfr':0.3121, 'stfr':0.2001} #median logMbar,logMstar errors
x_err  = 0.0899 #median logv error
x_noise=[0.05,0.10,0.2,.4,.6]
y_noise=[0.12,0.25,0.5,.6,.9]
N = 250

for i in range(len(x_noise)):
    x = (.5*np.random.rand(N))+1.8 #generating x axis points
    y = m['btfr'] * x + b['btfr']
    print(np.min(y), np.max(y))
    print(np.min(x), np.max(x))


    # y += np.abs(f_true * y) * np.random.randn(N)
    # x += x_noise[i]*np.random.randn(N)
    x += x_noise[i]*np.random.uniform(-1,1,size=N)
    y += y_noise[i]*np.random.uniform(-1,1,size=N)
    # y += y_noise[i]*np.random.randn(N)

    fig= plt.figure(figsize=(8.0,5.5), dpi=300) #figsize=(width,height)
    ax1 = fig.add_subplot(1,1,1)
    plt.ylim(6.5, 14)
    plt.xlim(1.325,3.125)
    # plt.ylim(8.0, 12.0)
    # plt.xlim(min(x)-0.5,max(x)+0.5)

    plt.errorbar(x, y,y_err[label],x_err, fmt='h', ms=5, color='#54bebe', mfc='#54bebe', mew=1, ecolor='grey', alpha=0.5, capsize=2.0, zorder=0, label='Synthetic Data');
    plt.errorbar(logVRout, logMbar_sers,logMbar_err_sers,logVRout_err, fmt='h', ms=5, color='#d7658b', mfc='#d7658b', mew=1, ecolor='grey', alpha=0.5, capsize=2.0, zorder=0, label='Real Data (BTFR)');
    # plt.scatter(logVRout, logMbar_sers,c=list(Re))
    # plt.colorbar()
    plt.tick_params(direction='inout', length=7, width=2)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    # plt.title('BTFR: Synthetic v. Real Data')

    plt.axhline(y=8.5, xmin=0.0, xmax=3.5, color='grey',label='$y_i\in[8.5,12],\,\,\, x_i\in[1.4,3]$',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.axhline(y=12, xmin=0.0, xmax=3.5, color='grey',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.axvline(x=1.4, color='grey',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
    plt.axvline(x=3, color='grey',alpha=0.75,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')

    axis=np.linspace(1.3,2.7)
    plt.axline((0, b[label]), slope=m[label],linestyle= '-',color='#6d4b4b',linewidth=3, label='$y={}x+{}$'.format(m[label],b[label]))

    # plt.plot(axis,m[label]*axis+b[label],label='$y={}x+{}$'.format(m[label],b[label]),color='black')
    # plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
    plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
    plt.text(2.5, 7.5, 'y noise = {}'.format(y_noise[i]),color='darkred',fontsize=12)
    plt.text(2.5, 7.1, 'x noise = {}'.format(x_noise[i]),color='darkred',fontsize=12)

    # plt.show()
    xminorLocator = AutoMinorLocator(10) #minor tick locator
    yminorLocator = AutoMinorLocator(10) #minor tick locator

    plt.tick_params(direction='inout', length=7, width=2)
    ax1.xaxis.set_minor_locator(xminorLocator) #printing minor ticks
    ax1.yaxis.set_minor_locator(yminorLocator) #printing minor ticks
    ax1.tick_params(direction = 'in', axis='x',which='minor',bottom=True, length=3) #giving size and direction to minor ticks
    ax1.tick_params(direction = 'in', axis='x',which='minor',top=True, length=3) #giving size and direction to minor ticks
    ax1.tick_params(direction = 'in', axis='y',which='minor',bottom=True, length=3) #giving size and direction to minor ticks
    ax1.tick_params(direction = 'in', axis='y',which='minor',right=True, length=3) #giving size and direction to minor ticks
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.tight_layout()
    plt.savefig('synthetic_data_plots/xscat/synthetic'+str(i+1)+'.png')
