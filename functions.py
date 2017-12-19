# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:00:53 2017

@author: R A B A B
"""
import numpy as np
import matplotlib.pylab as plt
from numpy import linalg as LA
import scipy.io as spio
from Isotopes import *


No_seg = 3
def data(fname,  start = 0, step = 1, File = 'mat'):
    if File == 'mat':
        mat = spio.loadmat(fname)
        X = mat['data'][:, start::step]
        time = mat['DAYS']
        print(time.shape)
        dt = (time[1] - time[0])[0] * step
    return(X, dt)


def Pca(X, r):
    """
    This function takes the training data or the sanpshpts matrix
    and performs SVD on it
    inputs:
        X : Training data
        r : number of truncated basis
    outputs:
        U: the orthonormal basis
    """
    data_centered = X - np.mean(X, axis = 0)
    U, S, V = LA.svd(X[:,:-1])
    return U[:,:r]

def training_file(Inputs, Outputs, Ins, Outs, dt):
   """
   This function process and write the training data file that dakota needs
   to train the surrogate
   """
   header = "%eval_id  interface  time   "
   for In in Ins:
    header+= In  +'  '
   for Out in Outs:
    header+= Out + "  "
   training_data = ''
   for i in range(len(Outputs[0] ) -1):
        training_data += str(i+1) + '  NO_ID  ' + str(dt*(i+1)) + ' '
        for j in range(len(Inputs)):
            training_data += str(Inputs[j, 0]) + ' '
        for k in range(len(Outputs)):
            training_data += str(Outputs[ k ,i+1])+ ' '
        training_data += '\n'
   training_file = header + '\n' + training_data
   with open('./training_data','w') as f:
        f.write(training_file)


def eval_points(test, dt_test, basis, Training = False, reduction = True):
    """
    This function prepares the file that contains the points
    at which the surrogate will be evaluated
    """
    evals = ''
    if reduction:
        test = basis.T.dot(test)
    for i in range(len(test[0]) -1):
        if Training:
            evals += str((dt_test/2) + dt_test*(i+1)) + ' '
        else:
            evals += str(dt_test*(i+1))
        for j in range(len(test)):
            evals +=str(test[j,0]) + ' '
        evals += '\n'
    with open ('./evals', 'w') as f:
        f.write(evals)

def data_of_interest( X, nuclides_of_interest):
    all_names = []
    names_of_interest = []
    for i in range(1, No_seg+1):
        for name in names:
          all_names.append(name + '_' + str(i))
        if nuclides_of_interest:
           for name in nuclides_of_interest:
              names_of_interest.append(name + '_' + str(i))
           data_used = np.zeros((len(nuclides_of_interest)*No_seg, X.shape[1]))
    if nuclides_of_interest:
          names_interest = []
          j = 0
          for  index, isotope in enumerate(all_names):
              for n in names_of_interest:
                  if isotope == n:
                      names_interest.append(isotope)
                      data_used[j ,:] = X[index, :]
                      j+=1
          all_names = names_interest
          X = data_used
    return X, all_names

def Ins_and_outs( X, r, R, nuclides_of_interest = None, Input_reduction = True, out_reduction = False):
    data, all_names = data_of_interest(X, nuclides_of_interest)
    Inputs_data = data
    Outputs_data = data
    if Input_reduction:
        basis_r = Pca(data, r )
        Inputs_names = ['r_' + str(i) for i in range(1, r + 1)]
        Inputs_data = basis_r.T.dot(data)
    else:
        Inputs_names = all_names
        Inputs_data = data
        basis = None
    if out_reduction:
        basis_R= Pca(data, R)
        Outputs_names = ['R_' + str(i) for i in range(1, R+1)]
        Outputs_data = basis_R.T.dot(data)
    else:
        Outputs_names = all_names
        Outputs_data = data
    return Inputs_names, Outputs_names, Inputs_data, Outputs_data, basis_r, data

def reduced_data(basis, X):
    reduced_data = basis.T.dot(X)
    return reduced_data




def dakota_writer (data, test, basis, Inputs, Outputs, reduction = True):
    if reduction :
        print(basis.shape)
        total_data =  basis.T.dot(test) # projecting all the data onto the active subspace to determine our bounds
    else:
        total_data = data
    upper_bounds= [ str(i) for i in np.max(total_data, axis = 1)]
    print(upper_bounds)
    lower_bounds = [ str(i) for i in np.min(total_data, axis = 1)]
    print(lower_bounds)
    upper_bounds.insert(0, '3000') # upper time
    lower_bounds.insert(0, '0')
    no_outputs = len(Outputs)
    inputs_names = ['"time"']
    Outputs = ["'" + i + "'" for i in Outputs]
    Inputs = ["'" + i + "'" for i in Inputs]
    for i in Inputs:
        inputs_names.append(i)
    no_inputs = len(inputs_names)
    with open ('dakota_template','r') as f:
        f = f.read().format('"dak_evaluation"','"training_data"', no_inputs, " ".join(lower_bounds), " ".join(upper_bounds), " ".join(inputs_names), "\n".join(Outputs), no_outputs)
    with open('./dakota.inp','w') as g:
        g.write(f)

#def compute_errors(true, predict):
#
#RE = abs(true - predict)/true
#RE_mean = np.mean(RE, axis = 0)
#Frob = LA.norm(true - predict)/LA.norm(true)