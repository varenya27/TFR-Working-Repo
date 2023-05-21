'''
plot the BTFR and STFR data from the kross/kmoss data
'''
import numpy as np
from matplotlib import pyplot as plt# import pyplot from matplotlib
import pandas as pd 
from extract import extract_data
from extract import extract_data,extract_direct,extract_rad,kross,kmos
import matplotlib
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['mathtext.fontset'] = 'stixsans'
# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.labelsize']=16


def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

logMbar_sers,logMbar_err_sers, logMstar_sers,logMstar_err_sers,logVRout,logVRout_err=extract_data()
Mbar,Mbar_err,Mstar,Mstar_err,Mgas,Mgas_err,VRout,VRout_err = extract_direct()
Rgas,Rgas_err,Re,Re_err = extract_rad()
kr_bar,kr_bar_e,kr_star,kr_star_e,kr_v,kr_v_e = logMbar_sers[:169],logMbar_err_sers[:169], logMstar_sers[:169],logMstar_err_sers[:169],logVRout[:169],logVRout_err[:169]
km_bar,km_bar_e,km_star,km_star_e,km_v,km_v_e = logMbar_sers[169:],logMbar_err_sers[169:], logMstar_sers[169:],logMstar_err_sers[169:],logVRout[169:],logVRout_err[169:]

current = 'kross'
if current == 'kmos3d':
    y1, err_y1, x1, err_x1= km_bar,km_bar_e,km_v,km_v_e
else:
    y, err_y, x, err_x= kr_bar,kr_bar_e,kr_v,kr_v_e

y1, err_y1, x1, err_x1= km_bar,km_bar_e,km_v,km_v_e
y, err_y, x, err_x= kr_bar,kr_bar_e,kr_v,kr_v_e


plt.figure(figsize=(8.0,5.5), dpi=300)
x_line=np.linspace(min(VRout)-0.5,max(VRout)+0.5)
# plt.plot(x_line, line[0]*x_line+line[1],'-',color='darkorange',linewidth=4)
# plt.plot(x_line, (line[0]+hi[0])*x_line+line[1]+hi[1],'--',color='darkorange')
# plt.plot(x_line, (line[0]-lo[0])*x_line+line[1]-lo[1],'--',color='darkorange')
# plt.fill_between (x_line, (line[0]+hi[0])*x_line+line[1]+hi[1], (line[0]-lo[0])*x_line+line[1]-lo[1], color='peachpuff', hatch='\\\\\\\\', alpha=0.5, zorder=0, )

plt.ylim(8.5, 12.5)
plt.xlim(1,3.5)

# plt.errorbar(logVRout, logMbar_sers,logMbar_err_sers,logVRout_err, fmt='h', ms=5, color='#d7658b', mfc='#d7658b', mew=1, ecolor='#df979e', alpha=0.5, capsize=2.0, zorder=0, label='Real Data (BTFR)');
# plt.errorbar(logVRout, logMstar_sers,logMstar_err_sers,logVRout_err,alpha=0.5, fmt='h', ms=5, mfc='red',color='#242424', mew=1, ecolor='grey', capsize=2.0, zorder=0, label='STFR');
# plt.errorbar(logVRout, logMbar_sers,logMbar_err_sers,logVRout_err,alpha=0.5, fmt='h', ms=5, mfc='blue',color='#242424', mew=1, ecolor='grey', capsize=2.0, zorder=0, label='BTFR');
# plt.errorbar(logVRout, np.log10(Mstar),Mstar_err/2.303/Mstar,logVRout_err, fmt='h', ms=5, color='#df979e', mfc='#df979e', mew=1, ecolor='#df979e', alpha=0.5, capsize=2.0, zorder=0, label='STFR');
# plt.scatter(logVRout, logMstar_sers,c=list(Re),marker='H',alpha=0.75,cmap='viridis' )
# plt.scatter(logVRout, logMstar_sers,c=Mgas/Mbar,marker='H',cmap='magma',edgecolors='#242424',alpha=0.75, )
# plt.colorbar()
# plt.errorbar(X1, Y1, fmt='o', ms=5, color='orange', mfc='orange', mew=1, ecolor='gray', alpha=0.5, capsize=2.0, zorder=0, label='Individual Data');
# plt.errorbar(X, Y, xerr=Xerr, yerr=Yerr, fmt='h', ms=10, color='orangered', mfc='orange', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=5, label='Binned Data');
# plt.errorbar(X, Y, fmt='h', ms=12, color='orangered', mfc='none', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=6);
# if current == 'kmos3d':
plt.errorbar(x1, y1,err_y1,err_x1, alpha=1, fmt='h', ms=5, mfc='darksalmon',color='salmon', mew=1, ecolor='darkgrey', capsize=2.0, zorder=0, label='KMOS3D Data');
# else:
plt.errorbar(x, y,err_y,err_x, alpha=0.5, fmt='h', ms=5, mfc='cornflowerblue',color='cornflowerblue', mew=1, ecolor='darkgrey', capsize=2.0, zorder=0, label='KROSS Data');

plt.tick_params(direction='inout', length=7, width=2)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('$\log V_{R_{out}}$')
ylabel = '$\log M_{bar}$'
plt.ylabel(ylabel)
# plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
plt.legend()
plt.savefig('data.png')
plt.show()
