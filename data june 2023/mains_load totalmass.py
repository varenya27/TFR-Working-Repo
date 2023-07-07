'''
run mcmc to find best fit params given the data using L1 and L2
'''
data = ['COMBINED']
L=data[0]
fres = open('results_STFR.csv','w')
fres.write('data,m,m_err,b,b_err,intscat,intscat_err,1sigma,2sigma,3sigma\n')
import numpy as np
from matplotlib import pyplot as plt# import pyplot from matplotlib
import corner
import pandas as pd
import pickle 

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['mathtext.default'] = 'regular'

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y
    
for L in data:
    f = 'data/'+L+'_FDM_cat.csv' 

    df = pd.read_csv(f)
    VRout,VRout_err,Mbar,Mbar_err,Mstar,Mstar_err = df.VRout.to_numpy(),df.VRout_err.to_numpy(),df.Mbar.to_numpy(),df.Mbar_err.to_numpy(),df.Mstar.to_numpy(),df.Mstar_err.to_numpy()
    Mgas,Mgas_err = Mbar-Mstar,Mbar_err-Mstar_err 
    logMbar = np.log10(Mbar)
    logMstar = np.log10(Mstar)
    logVRout = np.log10(VRout)
    logMbar_err = Mbar_err/(2.303*Mbar)
    logMstar_err = Mstar_err/(2.303*Mstar)
    logVRout_err = VRout_err/(2.303*VRout)

    y, err_y, x, err_x= logMstar,logMstar_err,logVRout,logVRout_err


    Nens = 300 # number of ensemble points
    ndims = 3
    Nburnin = 500  #500 # number of burn-in samples
    Nsamples = 3000  #500 # number of final posterior samples


    FF = open(f"pickle/sampler_{L}_STFR.pkl", "rb")
    flat_samples = pickle.load(FF)

    m_final = np.percentile(flat_samples[:, 0], [16, 50, 84])[1]
    c_final = np.percentile(flat_samples[:, 1], [16, 50, 84])[1]
    scat_final = np.percentile(flat_samples[:, 2], [16, 50, 84])[1]
    Med_value = [m_final,c_final,scat_final]

    figure = corner.corner(
        flat_samples,
        figsize=(11,9),
        # title_fmt=".2E",
        title_fmt = '.3f',
        # levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), 
        levels=(0.68,0.90,0.99), 
        labels=[r"m", r"b", r"$\sigma_{{int}}$"], 
        # quantiles=[0.16,0.84], 
        range=
                [(0.7,6), (-3,8), (-0.03,0.21)] if L=='KMOS3D'
            else([(1,4), (2,7.2), (-0.02,0.1)] if L=='KROSS'
            else([(0.,7), (-4,8.5), (-0.05,0.35)] if L == 'KGES'
                 else([(1,3.8), (2.6,7), (-0.02,0.07)])
            )),        
        show_titles=True, 
        label_kwargs={"fontsize": 16},
        title_kwargs={"fontsize": 14},
        color = '#003153',alpha=0.1,fill_contours = 1,
        # contour_kwargs={"color": 'r'}
        
    );
    axis_color='#d64267'
    axes = np.array(figure.axes).reshape((ndims, ndims))
    for i in range(ndims):
        ax = axes[i, i]
        ax.axvline(Med_value[i], color=axis_color)
    for yi in range(ndims):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(Med_value[xi], color=axis_color)
            ax.axhline(Med_value[yi], color=axis_color)
            ax.scatter(Med_value[xi], Med_value[yi], color=axis_color)


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
    # line = [2.005,5.806,0.145]
    #bestfit with scatter
    x_line=np.linspace(min(x)-0.5,max(x)+0.5)
    x_line=np.linspace(min(x)-0.5,max(x)+0.5)
    y_line = line[0]*x_line+line[1]
    y_fit =[]
    for n in range(len(x)):
        y_fit.append(np.interp(x[n],x_line,y_line))
    scat = np.sqrt(np.median((y-np.array(y_fit))**2))
    plt.plot(x_line, y_line,'-',color='#303030',linewidth=4,label=f'y={round(line[0],2)}x+{round(line[1],2)}')
    plt.plot(x_line, y_line+( 3*scat ) ,'--',color='#303030')
    plt.plot(x_line, y_line-( 3*scat ),'--',color='#303030')
    plt.fill_between (x_line, y_line+( 3*scat ), y_line-( 3*scat ), color='#b3a4a4', hatch='\\\\\\\\', alpha=0.5, zorder=0, label='3$\sigma$ scatter')

    # paper_btfr,btfr,z_btfr,m_past_btfr,zero_past_btfr,chi_btfr,int_scat_past_btfr = np.loadtxt('past_btfr.txt',unpack=True)
    # b_past_btfr = zero_past_btfr
    fres.write(f'{L},{line[0]},{(hi[0]+lo[0])/2},{line[1]},{(hi[1]+lo[1])/2},{line[2]},{(hi[2]+lo[2])/2},{scat},{2*scat},{3*scat}\n')


    df = pd.read_csv('past_stfr.txt',sep=' ')
    paper_btfr = df['paper']
    m_past_btfr=df['m'].to_numpy()
    b_past_btfr=df['b'].to_numpy()
    # b_past_btfr = m_past_btfr/zero_past_btfr
    c=['teal','orange','red','blue','green']
    for i in range(len(m_past_btfr)):
        y_line = m_past_btfr[i]*x_line+b_past_btfr[i]
        plt.plot(x_line, y_line,'-',color=c[i],linewidth=2,label=paper_btfr[i])




    plt.ylim(8.0, 12.0)
    plt.xlim(min(x)-0.5,max(x)+0.5)
    X,Y= x,y
    Xerr,Yerr =err_x, err_y
    Y1,Y1_e,X1,X1_e = logMstar,logMstar_err,logVRout,logVRout_err
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
    plt.xlabel('$\log V_{R_{out}}$')
    ylabel = '$\log M_{star}$'
    plt.ylabel(ylabel)
    # plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
    plt.legend()
    plt.title(f'Data: {L} | STFR')
    plt.savefig(f'besfit{L}_STFR.png')
    # plt.show()
