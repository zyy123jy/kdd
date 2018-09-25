from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd 
from pandas import DataFrame, read_csv
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy.matlib
from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Mixture, Logistic, Laplace, PointMass, Gamma)

def test_model():
    X_test = np.array(pd.read_csv('./X_test.txt', sep=",", header = None))
    Y1_test = np.array(pd.read_csv('./Y1_test.txt', sep=",", header = None))
    Y2_test = np.array(pd.read_csv('./Y2_test.txt', sep=",", header = None))
    Y5_test = np.array(pd.read_csv('./Y5_test.txt', sep="," ,header = None))
    T_test = np.array(pd.read_csv('./T_test.txt', sep=",", header = None))

    Y1_test = np.array(Y1_test)/mmse_scale
    where_are_NaNs = np.isnan(Y1_test)
    Y1_test[where_are_NaNs] = 0
    
    Y2_test = np.array(Y2_test)/adas_scale
    where_are_NaNs = np.isnan(Y2_test)
    Y2_test[where_are_NaNs] = 0

    Y5_test = np.array(Y5_test)/cd_scale
    where_are_NaNs = np.isnan(Y5_test)
    Y5_test[where_are_NaNs] = 0
    Mask_T = (T_test>0)*1
    T_test = np.array(Mask_T*(T_test-base_age)/time_scale)
    Me = 2
    Md = 5
    D = np.size(X_test,1)
    Ne = np.size(X_test,0)
    a0 = np.reshape(np.array(pd.read_csv('./a0.txt', sep=",", header = None),dtype='f'),(Md))
    b0 = np.reshape(np.array(pd.read_csv('./b0.txt', sep=",", header = None),dtype='f'),(1))
    w0 = np.reshape(np.array(pd.read_csv('./w0.txt', sep=",", header = None),dtype='f'),(D,1))
    v0 = np.reshape(np.array(pd.read_csv('./v0.txt', sep=",", header = None),dtype='f'),(D,1))
    h0 = np.reshape(np.array(pd.read_csv('./h0.txt', sep=",", header = None),dtype='f'),(Md))
    c0 = np.reshape(np.array(pd.read_csv('./c0.txt', sep=",", header = None),dtype='f'),(Md))
    sigma_y0 = np.reshape(np.array(pd.read_csv('./sigma_y0.txt', sep=",", header = None),dtype='f'),(Md))
    sigma_s0 = np.reshape(np.array(pd.read_csv('./sigma_s0.txt', sep=",", header = None),dtype='f'),(Md))
    sigma_q0 = np.reshape(np.array(pd.read_csv('./sigma_q0.txt', sep=",", header = None),dtype='f'),(Md))

    past_visit = 4
    for iter_tt in range(0,past_visit):
            Me = iter_tt       
            print("known time points")
            print(Me)
            Xt = tf.placeholder(tf.float32, [Ne,D])
            Tt = tf.placeholder(tf.float32, [Ne,Me])  
            Maskt = tf.placeholder(tf.float32, [Md-2,Ne,Me])
            Yt1 = tf.placeholder(tf.float32,[Ne,Me])
            Yt2 = tf.placeholder(tf.float32,[Ne,Me])    
            Yt5 = tf.placeholder(tf.float32,[Ne,Me])  
# models   
            s1 =  Normal(loc=tf.matmul(Xt,w0) + b0, scale=sigma_s0[0]*tf.ones([Ne,1],tf.float32))
            q11 = Normal(loc=tf.matmul(Xt,v0) + a0[0], scale=sigma_q0[0]*tf.ones([Ne,1],tf.float32))
            q12 = Normal(loc=tf.matmul(Xt,v0) + a0[1], scale=sigma_q0[1]*tf.ones([Ne,1],tf.float32))
            q15 = Normal(loc=tf.matmul(Xt,v0) + a0[4], scale=sigma_q0[4]*tf.ones([Ne,1],tf.float32))
        
            q_s1 =  Normal(loc=tf.Variable(tf.random_normal([Ne,1])),
                   scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))    
            q_q11 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),
                   scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))
            q_q22 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),
                   scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))
            q_q55 = Normal(loc=tf.Variable(tf.random_normal([Ne,1])),
                   scale=tf.nn.softplus(tf.Variable(tf.random_normal([Ne,1]))))
            
            Zp1 = tf.sigmoid(tf.abs(s1)*(Tt-q11))*Maskt[0,:,:]    
            Zp2 = tf.sigmoid(tf.abs(s1)*(Tt-q12))*Maskt[1,:,:]       
            Zp5 = tf.sigmoid(tf.abs(s1)*(Tt-q15))*Maskt[2,:,:]       
            Yt1 = Normal(loc=(-c0[0]*Zp1+h0[0])*Maskt[0,:,:],scale = sigma_y0[0]*tf.ones([Ne,Me],tf.float32))
            Yt2 = Normal(loc=(c0[1]*Zp2+h0[1])*Maskt[1,:,:],scale = sigma_y0[1]*tf.ones([Ne,Me],tf.float32))
            Yt5 = Normal(loc=(c0[4]*Zp5+h0[4])*Maskt[2,:,:],scale = sigma_y0[4]*tf.ones([Ne,Me],tf.float32))
            data = {Xt:X_test, Tt:T_test[:,0:Me], Maskt:Mask_test[:,:,0:Me], Yt1:Y_test1[:,0:Me], 
                    Yt2:Y_test2[:,0:Me],Yt5: Y_test5[:,0:Me]}
            inference = ed.KLqp({s1:q_s1, q11:q_q11, q12:q_q22, q15:q_q55},data) 
            inference.run(n_iter=50000)  
            zhat1 = Mask_test[0,:,:]*(tf.sigmoid((T_test-q_q11)*q_s1))
            Yhat1 = -c0[0]*zhat1+h0[0]
            zhat2 = Mask_test[1,:,:]*tf.sigmoid((T_test-q_q22)*q_s1) 
            Yhat2 = c0[1]*zhat2+h0[1]
            zhat5 = Mask_test[2,:,:]*tf.sigmoid((T_test-q_q55)*q_s1) 
            Yhat5 = c0[4]*zhat5+h0[4]
            err1 = (abs(Mask_test[0,:,Me:Mt]*(Y_test1[:,Me:Mt]-Yhat1[:,Me:Mt])))
            err2 = (abs(Mask_test[1,:,Me:Mt]*(Y_test2[:,Me:Mt]-Yhat2[:,Me:Mt])))
            err11 = tf.reduce_sum(err1)
            err22 = tf.reduce_sum(err2)
            err5 = (abs(Mask_test[2,:,Me:Mt]*(Y_test5[ :,Me:Mt]-Yhat5[:,Me:Mt])))
            err55 = tf.reduce_sum(err5)    
            n1 = tf.count_nonzero(Mask_test[0,:,Me:Mt])
            n2 = tf.count_nonzero(Mask_test[1,:,Me:Mt])    
            n5 = tf.count_nonzero(Mask_test[2,:,Me:Mt])   
            E1[iter,iter_tt] = err11.eval()/n1.eval() 
            E2[iter,iter_tt] = err22.eval()/n2.eval()
            E5[iter,iter_tt] = err55.eval()/n5.eval()        
            np.savetxt( './result/'+str(iter) + str(iter_tt)+"E1.txt", E1, delimiter=",")
            np.savetxt( './result/'+str(iter) + str(iter_tt)+"E2.txt", E2, delimiter=",")
            np.savetxt( './result/'+str(iter) + str(iter_tt)+"E5.txt", E5, delimiter=",")
            print("Mean absolute error on test data:")        
            print(E1)
            print(E2) 
            print(E5)
