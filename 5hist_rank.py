from matplotlib import pyplot as plt
import numpy as np

parameters=[" $\Omega_m$", " $\Omega_b$", " $h$", " $n_s$", " $\sigma_8$"]

x=np.genfromtxt('ranks.txt')
kk=0
for ii in range(5):
    
    plt.subplot(1,5,ii+1)
    plt.hist(x[:,ii], 10, density=False, color=['blue'], histtype='step')
             
    plt.title('Rank for' + parameters[ii])
    
    plt.tight_layout()
             
        
            
          
         
    
        
       
        
        
         
    
    
    


    
    





