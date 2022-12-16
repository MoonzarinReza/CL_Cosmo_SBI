import torch
from torch import zeros, ones
import sbi
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
#importing all packages used 
import numpy as np
#import matplotlib.pyplot as plt
import math
import emcee
import random
#import time

from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import interpolate
from scipy import integrate
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import interpolate
from scipy import integrate


n_tr=78000
n_te1=78100
n_te2=79980

a_np=np.genfromtxt("randomized_starts.txt")
a_np2=a_np[:,:]
x_np1 = a_np2[:, 5:21]

theta_np5 = a_np2[:, 0:5]
s1,s2=np.shape(theta_np5)
theta_np1=np.zeros((s1,s2))
theta_np1[:,0]=theta_np5[:,1]
theta_np1[:,1]=theta_np5[:,2]
theta_np1[:,2]=theta_np5[:,0]
theta_np1[:,3]=theta_np5[:,3]
theta_np1[:,4]=theta_np5[:,4]





print(np.shape(x_np1))
print(np.shape(theta_np1))

prior_len = theta_np1.shape[1]

prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)

sums=np.sum(x_np1, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf]
x_np1 = x_np1[contains_no_inf]

print(np.shape(x_np1))
print(np.shape(theta_np1))


theta_np = theta_np1[0:n_tr,:]
x_np = x_np1[0:n_tr,:]

#print(np.shape(x_np))
#print(np.shape(theta_np))

# turning into tensors

theta = torch.as_tensor(theta_np, dtype=torch.float32)
theta1 = torch.as_tensor(theta_np1, dtype=torch.float32)


x = torch.as_tensor(x_np, dtype=torch.float32)
x1 = torch.as_tensor(x_np1, dtype=torch.float32)


# SNLE, SNRE, SNPE
inferer = SNPE(prior, density_estimator="mdn", device="cpu")
#inferer = SNLE(prior, device="cpu")  

# Append training data
inferer = inferer.append_simulations(theta, x)
 

density_estimator =inferer.train()
posterior = inferer.build_posterior(density_estimator)  

xx=np.genfromtxt('sbi_analytical_input.txt')
yy=np.genfromtxt('sbi_analytical_output.txt')

s3,s4=np.shape(xx)
counts=s3
ranks=np.zeros((counts,s4))
cn=40
ci=np.zeros((cn,5))


for i in range(counts):
    #print(i)
    x_onew=yy[i]
    samples1 = posterior.sample((10000,), x=x_onew)
    np_samples1=samples1.numpy()
    np_samples1=np.sort(np_samples1,axis=0)
    
    clow=np.zeros((cn,5)) 
    chigh=np.zeros((cn,5)) 
    xstart=4875
    xend=5125
    for ccc in range(cn):
        
        for mmm in range(5):
            clow[ccc,mmm]=np_samples1[xstart,mmm]
            chigh[ccc,mmm]=np_samples1[xend,mmm]
        xstart=xstart-124
        xend=xend+124
    
    for ddd in range(cn):
        for mmm in range(5):
            if(xx[i,mmm]>clow[ddd,mmm] and xx[i,mmm]<chigh[ddd,mmm]):
                ci[ddd,mmm]=ci[ddd,mmm]+1

        
    

    

np.savetxt('con_int.txt', ci)





