# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:04:46 2017

@author: R A B A B
"""

from functions import *
import os

r = 3  # number of reduced basis
R = 2
fname = 'full_600.mat'
Ic = 'Ic_2.mat'
Interested = True
Nuclides_of_interest = ['U235', 'U238', 'Am241', 'Cs137', 'Cs134', 'Mo95', 'Pu239']
#Nuclides_of_interest = None
No_seg = 3
Input_reduction = True
Out_reduction = False
X_train, dt_train = data(fname, start = 0, step = 2)
X_test, dt_test = data(fname, start = 1, step = 2 )
X_test = data_of_interest( X_test, Nuclides_of_interest)[0]
Ins, Outs, In_data, Out_data, basis, data = Ins_and_outs(X_train, r, R, Nuclides_of_interest, Input_reduction = Input_reduction, out_reduction = Out_reduction)
training_file (In_data, Out_data, Ins, Outs, dt_train)
eval_points(X_test, dt_test, basis,Training = True, reduction = Input_reduction)
dakota_writer (data, X_test, basis, Ins, Outs, Input_reduction)
if Nuclides_of_interest:
    names= Nuclides_of_interest
else:
    selected_isotopes = names

if __name__ == '__main__':
    os.system('dakota dakota.inp')