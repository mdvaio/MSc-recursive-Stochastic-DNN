

from matplotlib import pyplot as plt 
import numpy as np  
from scipy.stats import chisquare
from scipy import stats
   

num=500
Y_dropoutt1 = np.zeros(num)
Y_dropoutt2 = np.zeros(num)
Y_dropoutt3 = np.zeros(num)


fig_num = [1,2,3]
#fig_num = [4,5,6]
#fig_num = [7,8,9]




plt.figure(fig_num[0])
t = 1000

for i in range(num):
    Y_dropoutt1[i] = best_model.predict(x_test[t,:].reshape(1,30))*normalizador

plt.hist(Y_dropoutt1, bins = 10) 
plt.hist(Y_dropoutt1, bins = 20) 
plt.hist(Y_dropoutt1, bins = 30) 
plt.hist(Y_dropoutt1, bins = 50) 
plt.hist(Y_dropoutt1, bins = 100) 

#plt.title("Histogram: " + str(num) + ' Samples, t = ' + str(t))
 
plt.legend(['bins = 10',
            'bins = 20',
            'bins = 30',
            'bins = 50',
            'bins = 100']
            )
plt.ylabel('Samples', fontsize=13)
plt.xlabel('SS values', fontsize=13)
plt.show()



plt.figure(fig_num[1])
t = 7000
for i in range(num):
    Y_dropoutt2[i] = best_model.predict(x_test[t,:].reshape(1,30))*normalizador

plt.hist(Y_dropoutt2, bins = 10) 
plt.hist(Y_dropoutt2, bins = 20) 
plt.hist(Y_dropoutt2, bins = 30) 
plt.hist(Y_dropoutt2, bins = 50) 
plt.hist(Y_dropoutt2, bins = 100) 

#plt.title("Histogram: " + str(num) + ' Samples, t = ' + str(t))

plt.legend(['bins = 10',
            'bins = 20',
            'bins = 30',
            'bins = 50',
            'bins = 100']
            )
plt.ylabel('Samples', fontsize=13)
plt.xlabel('SS values', fontsize=13)
plt.show()



#plt.figure(fig_num[2])
#t=5500
#for i in range(num):
#    Y_dropoutt3[i] = best_model.predict(x_test[t,:].reshape(1,30))*normalizador
#
#
#
#
#
#oi = np.random.normal(np.zeros(50000))
#stats.normaltest(oi)
#stats.normaltest(Y_dropoutt3)
#
#plt.figure(fig_num[2])
#plt.hist(Y_dropoutt3, bins = 10) 
#plt.hist(Y_dropoutt3, bins = 20) 
#plt.hist(Y_dropoutt3, bins = 30) 
#plt.hist(Y_dropoutt3, bins = 50) 
#plt.hist(Y_dropoutt3, bins = 100) 
#    
#plt.title("Histogram: " + str(num) + ' Samples, t = ' + str(t))
#
#plt.legend(['bins = 10',
#            'bins = 20',
#            'bins = 30',
#            'bins = 50',
#            'bins = 100']
#            )
#plt.ylabel('Samples')
#plt.xlabel('SS values')
#plt.show()
#
#
#
#
#
#
#
#
#
#
#
#oi = np.random.normal(np.zeros(50000000))
#stats.normaltest(Y_dropoutt3)
#chisquare(oi)
#plt.figure(100)
#plt.hist(oi, bins = 10) 
#plt.hist(oi, bins = 20) 
#plt.hist(oi, bins = 30) 
#plt.hist(oi, bins = 50) 
#plt.hist(oi, bins = 100) 
#    
#plt.title("OI Histogram: " + str(num) + ' Samples, t = 5500')
#
#plt.legend(['bins = 10',
#            'bins = 20',
#            'bins = 30',
#            'bins = 50',
#            'bins = 100']
#            )
#plt.ylabel('Samples')
#plt.xlabel('SS values')
#plt.show()



                    
                    
                    
                    
                    
                    
                    
                    
                    
                    