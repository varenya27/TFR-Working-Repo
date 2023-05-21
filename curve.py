'''
scipy.curve_fit to do param estimation for the data
'''

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from extract import extract_data,extract_direct
def line(x, m, c):
    return c+m*x

logMbar_sers,logMbar_err_sers, logMstar_sers,logMstar_err_sers,logVRout,logVRout_err=extract_data()
Mbar,Mbar_err,Mstar,Mstar_err,Mgas,Mgas_err,VRout,VRout_err = extract_direct()

v='Vout'
y, err_y, x, err_x= logMstar_sers,logMstar_err_sers,logVRout,logVRout_err
param, param_cov = curve_fit(line, x, y,sigma=err_y, absolute_sigma=True)
perr = np.sqrt(np.diag(param_cov))
y_result = param[0]*x+param[1]
print("parameters: ",param)
print("covariance of parameters: ",param_cov)
print('error: ',perr)

plt.figure(figsize=(9,7))
plt.errorbar(x,y,yerr=err_y,xerr=err_x, fmt='h', ms=5, color='#54bebe', mfc='#54bebe', mew=1, ecolor='#54bebe', alpha=0.5, capsize=2.0, zorder=0, label='Data Points');
# plt.plot(x, y_result, '--', label ="fit")
plt.axline((0, param[1]), slope=param[0],linestyle= '-',color='#6d4b4b',linewidth=2, label='Best Fit Line')

plt.ylim(8.5, 12.5)
plt.xlim(1,3.5)
plt.title(f'y = {round(param[0],2)}x+ {round(param[1],2)}')
plt.legend()
plt.xlabel('$log V_{R_{out}}$')
plt.ylabel('$log M_{star}$')
plt.show()