import numpy as np
from matplotlib import pyplot as plt

m={'btfr':4, 'stfr':3.6} #slope
b={'btfr':2.3, 'stfr':1.5} #intercept
y_noise=[0.12,0.25,0.5,.75,.9]

for noise in range(5):
    for like in range(1,3):
        results_m,results_b,results_me,results_be=[],[],[],[]
        file = r'histograms/results_xscatter/results_L{}_{}.txt'.format(like,noise)
        with open(file,'r') as f:
            for line in f:
                line = line.split()
                results_m.append(float(line[0])-m['btfr'])
                results_me.append(float(line[1]))
                results_b.append(float(line[2])-b['btfr'])
                results_be.append(float(line[3]))

        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
        fig.set_figheight(5.5)
        fig.set_figwidth(8)
        # fig.suptitle('Deviations - L{} | Noise = {}'.format(like,y_noise[noise]))
        ax1.hist(np.array(results_m)/np.array(results_me), 30, density=False, color='#c86558')
        ax1.set_title('Slope',fontsize=16,fontweight='bold')
        ax1.set_xlabel(r'$\frac{m_i-m}{\sigma_m}$',fontsize=18)
        ax1.set_ylabel(r'Frequency',fontsize=14)
        ax2.set_xlabel(r'$\frac{b_i-b}{\sigma_b}$',fontsize=18)
        # ax1.set_xlim(0,1)
        ax2.hist(np.array(results_b)/np.array(results_be), 30, density=False, color='#c86558')
        ax2.set_title('Intercept',fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('histograms/xscat/uniform/histogramresults_L{}_{}.png'.format(like,noise))