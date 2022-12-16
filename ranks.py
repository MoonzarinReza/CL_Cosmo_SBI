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


n_tr=46000
n_te1=46100
n_te2=46500


theta_np1 = np.genfromtxt("parameters49750.txt")

x_npm = np.genfromtxt("halomass49750.txt")

x_npc = np.genfromtxt("halocount49750.txt")

x_np1=np.concatenate((x_npm,x_npc), axis=1)


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

inde1=random.randint(n_te1, n_te2)

x_o = torch.as_tensor(x_np1[inde1], dtype=torch.float32)
theta_truth=theta_np1[inde1]


samples = posterior.sample((100000,), x=x_o)
np_samples=samples.numpy()
#np_samples=np.sort(np_samples)



def extract_power(mass_arr, mfunc):
    mass_arr_p = np.log10(mass_arr)
    mfunc_n = mfunc*(10**5)
    mass_arr_log = np.log(mass_arr)
    return mass_arr_p, mfunc_n, mass_arr_log
    
def lnpo(mass, min, max, test_fun):
  
  if (mass < min) or (mass > max):
    return -np.inf
  if (test_fun(mass)==0 or test_fun(mass)<0):
      print(test_fun(mass))
      print(mass)
      return -np.inf
  else:
      return math.log(test_fun(mass))
   #log likelihood is required by emcee

rz=[0,0.5]
def interpolate_MCMC(mass_array_p, mass_arr_log, mfunc, mfunc_n, mass_range, sample_num, redshift):
  
  min, max = mass_range
  
  min2, max2 = np.log(np.power(10, mass_range))
  
  interpolate_mfunc = interpolate.interp1d(mass_array_p, mfunc_n)
  interpolate_mfunclog = interpolate.interp1d(mass_arr_log, mfunc)
  
  test_arr = np.linspace(min, max, 5000)
  f_test = interpolate_mfunc(test_arr)
  
  
  val, err = integrate.quad(interpolate_mfunc, min, max)
  
  val2, err2 = integrate.quad(interpolate_mfunclog, min2, max2)
  sample_num2 = int(val2*10**9)
  
  test_fun = interpolate.interp1d(test_arr, (f_test/val))
  
  randomlist = []
  for i in range(20):
      n = random.uniform(min, max)
      randomlist.append(n)
  random_arr = np.array(randomlist)

  #backend setup
  ndim, nwalkers = 1, 20

  processed_sample_num = sample_num2 // (ndim * nwalkers)
  p0 = random_arr.reshape((nwalkers, ndim))
  the_random = random.uniform(1.0, 100.0)
  locals()["sample" + str(the_random)] = emcee.EnsembleSampler(nwalkers, ndim, lnpo, args = [min, max, test_fun])
  
  pos, prob, state = locals()["sample" + str(the_random)].run_mcmc(p0, 25000) #burn-in sequence
  locals()["sample" + str(the_random)].reset()
  
  
  locals()["sample" + str(the_random)].run_mcmc(pos, processed_sample_num)
  
  mass_chain = locals()["sample" + str(the_random)].chain
  locals()["sample" + str(the_random)].reset() #prevent storing error
  return test_fun, mass_chain.flatten()

def mass_sampling(mass_range, sample_num, params_full, redshift, mdef = '200c', model = 'bocquet16'):
  
  min, max = mass_range
  mass_arr = np.logspace(min, max, num = 200, base = 10)
  
  
  
  
  params = {'flat': True, 'H0': params_full[2]*100, 'Om0': params_full[0], 'Ob0': params_full[1], 'sigma8': params_full[4], 'ns': params_full[3]}
  cosmology.setCosmology('myCosmo', **params)
  mfunc = mass_function.massFunction(mass_arr, redshift, mdef = mdef, model = model, q_out = 'dndlnM')
  #print(mfunc)
  mass_arr_p, mfunc_n, mass_arr_log = extract_power(mass_arr, mfunc)
  test_func, prim_mass_sample = interpolate_MCMC(mass_arr_p, mass_arr_log, mfunc, mfunc_n, mass_range, sample_num, redshift)
  return test_func, prim_mass_sample

def unit_switch(prim_mass_sample):
  mass_sample = np.power(10, prim_mass_sample)
  return mass_sample

#sets up intial values

index = [0, 0.5]
mass_range = [13.0, 17.0]
mass_chain = []
num_variations = 2



mass_bin = [10, 20, 45, 80, 200, 10000000]
mass_bin_len = len(mass_bin) - 1



sample_num = 100000
counts=100

ranks=np.zeros((counts,8))




for i in range(counts):
    print(i)
    
    inde=random.randint(2, 99998)
    
    
    
    params_full=np_samples[inde,0:5]
    halocount=np.zeros(mass_bin_len*2)
    halomass=np.zeros(mass_bin_len*2)
    
    for w in range(len(rz)):
        
        test_func, mass_chain = mass_sampling(mass_range, sample_num, params_full, redshift = rz[w])
        mass_chain = np.power(10, mass_chain)
        log_mass = np.log(mass_chain)
        MA = np_samples[inde,5]
        MB = np_samples[inde,6]
        lnsigma =  np_samples[inde,7]
        mu = MA + MB*(log_mass - np.log(3.0*10**14.0))
        sigma = lnsigma      
        lnL = np.random.normal(mu, sigma, len(mu))
        lamda = np.exp(lnL)
        
        for kk in range(mass_bin_len):      
            mass_lo=mass_bin[kk]
            mass_up=mass_bin[kk+1]
    
            ind, = np.where( (lamda >= mass_lo) & (lamda < mass_up))
            abc=kk+w*5
            #print(abc)
            halocount[abc]=len(ind)
            halomass[abc] = np.log10(np.mean(mass_chain[ind]))
    x_onew=np.concatenate((halomass,halocount), axis=0)
    #posterior.leakage_correction(x_onew)
    samples1 = posterior.sample((100,), x=x_onew)
    np_samples1=samples1.numpy()
    for mmm in range(8):
        ranks[i,mmm]=len((np.where(np_samples[inde,mmm]>np_samples1[:,mmm]))[0])
        

np.savetxt('ranks.txt', ranks)            



