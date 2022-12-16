from matplotlib import pyplot as plt
import numpy as np

parameters=[" $\Omega_m$", " $\Omega_b$", " $h$", " $n_s$", " $\sigma_8$", " $M_A$", " $ M_B$", " $lnsigma$"]

x=np.genfromtxt('ranks.txt')
kk=0
for ii in range(2):
    for jj in range(4):
        if(ii==0):
             plt.subplot(2,4,jj+1)
             plt.hist(x[:,kk], 100, density=False, color=['blue'], histtype='step')
             x5=np.arange(0,101,1)
             y1=np.full((len(x5),1),4)
             y2=np.full((len(x5),1),0)
             plt.plot(x5,y1, linewidth=2, color='red')
             plt.plot(x5,y2, linewidth=2, color='red')
             plt.ylim(-1,10)
            
            
            

             
             plt.title('Rank for' + parameters[kk])
             kk=kk+1
             plt.tight_layout()
             
        else:
            plt.subplot(2,4, jj+5)
            plt.hist(x[:,kk], 100, density=False, color=['red'], histtype='step')
            x5=np.arange(0,101,1)
            y1=np.full((len(x5),1),4)
            y2=np.full((len(x5),1),0)
            plt.plot(x5,y1, linewidth=2, color='blue')
            plt.plot(x5,y2, linewidth=2, color='blue')
            plt.ylim(-1,10)
            
            
            
            plt.title('Rank for' + parameters[kk])
            kk=kk+1
            plt.tight_layout()
            
          
         
    
        
       
        
        
         
    
    
    


    
    




