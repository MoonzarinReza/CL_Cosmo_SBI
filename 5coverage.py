import numpy as np
from matplotlib import pyplot as plt

i=np.genfromtxt('con_int.txt')
i=i/230
x=np.arange(0.025,1.0001,0.025)
abc=['red', 'lime',  'cyan',  'blue',  'purple']
for jj in range(5):
    plt.plot(x,i[:,jj], color=abc[jj], linestyle='dashed', linewidth=1.0)
#plt.plot(x,x)
plt.xlabel('Confidence Interval')
plt.ylabel('Fraction of Spectra Recovered')
plt.xlim([0,1])
plt.ylim([0,1])
#plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
plt.legend(["$\Omega_m$", "$\Omega_b$", "$h$", "$n_s$", "$\sigma_8$"])

plt.plot(x,x, linewidth=1.0, color='black')





