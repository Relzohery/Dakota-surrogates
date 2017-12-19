# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:35:04 2017

@author: R A B A B
"""

import matplotlib.pylab as plt
from functions import *
from reduced_dakota import *


I = 2 + len(Ins)  # columns to be skipped , inputs
Results_file = np.loadtxt('./dak_evaluation', skiprows = 1)
time = Results_file[:, 1]
results = Results_file[:,I:]
X_test = X_test[:, 1:]
for i in range(0, len(names)):
    fig = plt.figure(i, figsize=(8,6))
    ax1 = fig.add_subplot(1,2,1)
    ax1.semilogy(time, results[:,i], marker =  'D', markersize = 5, label = 'Surrogate')
    ax1.semilogy(time, X_test[i,:], marker = 'D', label = 'True')
    err = (abs(results[:,i] - X_test[i,:])/X_test[i,:])*100
    max_err = max(err)
    err_mean = np.mean(err)
    err_frob = LA.norm(abs(results[:, i] - X_test[i,:]))/LA.norm(X_test[i, :])
    ax1.set_xlabel('Time (days)', size = 14)
    ax1.set_ylabel('Atomic density (atom/barn.cm)',size = 14)
    ax1.legend()
    ax1.set_title(' mean {:1.3}%'.format( err_mean))
    ax2 = fig.add_subplot(1,2,2)
    ax2.semilogy(time, err, 'o', time, err,'s')
    ax2.set_ylabel('Relative error %',size = 14)
    ax2.set_xlabel('Time (days)', size = 14)
    ax2.set_title('Frob. {:1.3}%'.format(err_frob))
    ax2.legend(('Segment 1', 'Segment -2'))
    fig.suptitle( Outs[i], horizontalalignment = 'center', size = 16)

    fig.tight_layout()
    #plt.savefig('./results/{}_no_reduction'.format(Nuclides_of_interest[j]))


####################### Errors #############3
Er_percentage = 100* abs(results.T -X_test)/X_test
max_Re_err = np.amax(Er_percentage, axis = 1 )
mean_Re_err = np.mean(Er_percentage, axis = 1)
Er_frob = 100*LA.norm(abs(results.T - X_test))/LA.norm(X_test)
err_dict = dict(zip(Outs, max_Re_err))
max_err_nuclide = max(err_dict.items(), key = lambda x:x[1])
mean_err_dict = dict(zip(Outs, mean_Re_err))
max_mean_nuclide = max(mean_err_dict.items(), key = lambda x:x[1])