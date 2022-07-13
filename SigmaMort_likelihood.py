import numpy as np
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from colossus.cosmology import cosmology
from colossus.lss import mass_function


def setup(options):
    section = option_section
    log10bins = options[section, "Bins"]
    volume = options[section, "volume"]
   
    output_file = 'quijote_fiducial_summary.txt'
    output = np.genfromtxt(output_file)
    cov = np.cov(output.T)
    mean_val = np.mean(output.T, axis=1)
    zinv=np.linalg.inv(cov)
    return log10bins, volume, zinv, cov, mean_val

def execute(block, config):

    log10bins, volume, zinv, cov, mean_val = config

    h = block['cosmological_parameters', 'h0']
    Omega_m = block['cosmological_parameters', 'omega_m']
    Omega_b = block['cosmological_parameters', 'omega_b']
    ns = block['cosmological_parameters', 'n_s']
    sigma_8 = block['cosmological_parameters', 'sigma8_input']
   
    #0.3174, 0.049, 0.6711, 0.9624, 0.834
    
    
    
    params = {'flat': True, 'H0': 100.0*h, 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8': sigma_8, 'ns': ns}
    
    cosmo = cosmology.setCosmology('myCosmo', params)
    #cosmology.setCosmology('WMAP9')
    mass_array = np.logspace(13.0, 16.0, num = 200, base=10.0)
    

    mfunc_fof = mass_function.massFunction(mass_array, 0.0, mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
    numfactor = 1.0/0.9484319366619587 
    numfactor2 = 1.0/0.9304373265412987
    shif = np.random.multivariate_normal(np.zeros(len(mean_val)), cov)
    #plt.loglog(mass_array, mfunc_fof)
    #plt.show()
    nums, aver_masses = predictions(mass_array, mfunc_fof, volume, log10bins)
    #print( nums, aver_masses)
    mfunc_fof2 = mass_function.massFunction(mass_array, 0.5, mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
    nums2, aver_masses2 = predictions(mass_array, mfunc_fof2, volume, log10bins)

    ## assemble theory data vector
    nums=nums*numfactor
    nums2=nums2*numfactor2
    theory = np.hstack((nums, aver_masses, nums2, aver_masses2))
    print(theory, shif)

    
    ## shifts_things
    nums_rand = nums + shif[0:4]
    aver_masses_rand = aver_masses + shif[4:8]
    nums2_rand = nums2 + shif[8:12]
    aver_masses2_rand = aver_masses2 + shif[12:16]
    ### output
    output_file_M = 'sbi_analytical_output.txt'
    with open(output_file_M, "a") as text_file:
         for item in nums_rand:
             text_file.write(str(item)+' ')
         for item in aver_masses_rand:
             text_file.write(str(item)+' ')
         for item in nums2_rand:
             text_file.write(str(item)+' ')
         for item in aver_masses2_rand:
             text_file.write(str(item)+' ')
         text_file.write('\n')
         text_file.close()
    #input parameters
    input_file_NC = 'sbi_analytical_input.txt'
    array = [Omega_m, Omega_b, h, ns, sigma_8]
    with open(input_file_NC, "a") as text_file:
         for item in array:
             text_file.write(str(item) + ' ')
         text_file.write('\n')
         text_file.close()


    ### to be completed for likelihood
    loglike = 0.0
    delta2 = theory - mean_val
    weight2 = np.linalg.inv(cov)
    loglike2 = -0.5 * np.dot(delta2, np.dot(weight2, delta2))
    #loglike = loglike + loglike2
    block[section_names.likelihoods, 'SigmaMort_Like_like'] = loglike
  
    return 0

def cleanup(config):
    #nothing to clean up
    return 0

def cleanup(config):
    #nothing to clean up
    return 0

def predictions(masses, mass_func, volume, log10bins):

    bins = 10.0**log10bins
    num_bin = len(bins)-1
    nums = np.zeros(num_bin)
    aver_masses = np.zeros(num_bin)

    lnm = np.log(masses)
    func = interpolate.interp1d(lnm, mass_func)
    func_mass = interpolate.interp1d(lnm, masses * mass_func)
    for ii in range(num_bin):
        upper = np.log(bins[ii+1])
        lower = np.log(bins[ii])

        total_mass,total_mass_err = integrate.quad(func_mass, lower, upper)
        total_dens, total_dens_err = integrate.quad(func, lower, upper)
        aver_masses[ii] = np.log10(total_mass/total_dens)
        nums[ii] = volume * total_dens
    return nums, aver_masses
    




