#### %matplotlib inline

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
def train_model():
    mmse_scale = 20
    adas_scale = 50
    cdr_scale = 10
    time_scale = 15
    base_year = 60
    X_data = pd.read_csv('./X.txt', sep="\t", header = None)
    Y1 = pd.read_csv('./Y1.txt', sep="\t", header = None)
    Y2 = pd.read_csv('./Y2.txt', sep="\t", header = None)
    Y3 = pd.read_csv('./Y3.txt', sep="\t", header = None)
    Y4 = pd.read_csv('./Y4.txt', sep="\t", header = None)
    Y5 = pd.read_csv('./Y5.txt', sep="\t", header = None)
    T_data = np.array(pd.read_csv('./T.txt', sep="\t", header = None))

    Y1 = np.array(Y1)/mmse_scale
    where_are_NaNs = np.isnan(Y1)
    Y1[where_are_NaNs] = 0

    Y2 = np.array(Y2)/adas_scale
    where_are_NaNs = np.isnan(Y2)
    Y2[where_are_NaNs] = 0

    Y3 = np.array(Y3)
    where_are_NaNs = np.isnan(Y3)
    Y3[where_are_NaNs] = 0

    Y4 = np.array(Y4)
    where_are_NaNs = np.isnan(Y4)
    Y4[where_are_NaNs] = 0

    Y5 = np.array(Y5)/cdr_scale
    where_are_NaNs = np.isnan(Y5)
    Y5[where_are_NaNs] = 0

    n0 = np.size(T_data,0)
    m0 = np.size(T_data,1)
    
    X_data = np.array(X_data)
    where_are_NaNs = np.isnan(X_data)
    X_data[where_are_NaNs] = 0
    
    Mask_T = (T_data>0)*1
    T_data = np.array((T_data-base_year))/time_scale
    T_data = np.array(Mask_T*(T_data))
    tmp = np.reshape(np.array(Y3[:,0]),(n0,1))
    X_data = np.concatenate((X_data,tmp),axis=1)
    tmp = np.reshape(np.array(Y4[:,0]),(n0,1))
    X_data = np.concatenate((X_data,tmp),axis=1)
    D = np.size(X_data,1)
    E1 = np.zeros([30,5])
    E2 = np.zeros([30,5])
    E5 = np.zeros([30,5])
    
    for iter in range(0,1):
         # validation 
           
        index = np.arange(0,n0)
        print(index)
        #index = np.delete(index,idx)
        index_test = index[(iter)*145:(iter+1)*145]  
        index_val = index[(iter+1)*145:(iter+2)*145]  
        index_train = index
        index_train = np.delete(index_train, index_test)
      
        X_train = X_data[index_train,:]
        Y_train1 = Y1[index_train,0:10]
        Y_train2 = Y2[index_train,0:10]
        Y_train3 = Y3[index_train,0:10]
        Y_train4 = Y4[index_train,0:10]
        Y_train5 = Y5[index_train,0:10]
        T_train = T_data[index_train,0:10]    
    
        X_val = X_data[index_val,:]
        Y_val1 = Y1[index_val,0:10]
        Y_val2 = Y2[index_val,0:10]
        Y_val3 = Y3[index_val,0:10]
        Y_val4 = Y4[index_val,0:10]
        Y_val5 = Y5[index_val,0:10]
        T_val = T_data[index_val,0:10]    
       # index_test = idx
        T_test = T_data[index_test,0:10]
        Y_test1  = Y1[index_test,0:10]
        Y_test2  = Y2[index_test,0:10]
        Y_test5  = Y5[index_test,0:10]
        X_test = X_data[index_test,:]
     
        N = np.size(T_train,0)
        M = np.size(T_train,1)
        Ne = np.size(T_test,0)
        Mt = np.size(T_test,1)   
        D = np.size(X_test,1)
    
        err10 = 100
        err20 = 100
        err50 = 100
        Md = 5 # modality
        Mask_train = np.zeros((Md,N,M))
        Mask_train[0,:,:] = (Y_train1>0)*1
        Mask_train[1,:,:] = (Y_train2>0)*1
        Mask_train[2,:,:] = (Y_train3>0)*1
        Mask_train[3,:,:] = (Y_train4>0)*1
        Mask_train[4,:,:] = ((Y_train5-Y_train5)==0)*1
        Mask_test = np.zeros((Md-2,Ne,Mt))
        Mask_test[0,:,:] = (Y_test1>0)*1
        Mask_test[1,:,:] = (Y_test2>0)*1
        Mask_test[2,:,:] = ((Y_test5-Y_test5)==0)*1      
        Mask_val = np.zeros((Md-2,Ne,Mt))
        Mask_val[0,:,:] = (Y_val1>0)*1
        Mask_val[1,:,:] = (Y_val2>0)*1
        Mask_val[2,:,:] = ((Y_val5-Y_val5)==0)*1  
        for i_val in range(0,1): 
            X = tf.placeholder(tf.float32, [N,D])
            T = tf.placeholder(tf.float32, [N,M])
            Mask = tf.placeholder(tf.float32, [Md,N,M])
            Yhat1 = tf.placeholder(tf.float32, [N,M])
            Yhat2 = tf.placeholder(tf.float32, [N,M])
            Yhat3 = tf.placeholder(tf.float32, [N,M])
            Yhat4 = tf.placeholder(tf.float32, [N,M])
            Yhat5 = tf.placeholder(tf.float32, [N,M])
            sigma_q = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([Md,1]))),name = 'sigma_q')
            sigma_s = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([Md,1]))),name='sigma_s')
            sigma_y = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([Md,1]))),name='sigma_y')
            sigma_c = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([1]))),name='sigma_c')
            sigma_h = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([1]))),name='sigma_h')
          # parameters ininitialization
            h = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([Md]))),name='h')
            c = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([Md]))),name='c')
            w = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([D,1]))),name='w')
            b = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([1]))),name='b')
            v = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([D,1]))),name='v')
            a = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([Md,1]))),name='a')
              
       # models
            sm1 = Normal(loc=(tf.matmul(X,w) + b),scale = sigma_s[0]*tf.ones([N,1],tf.float32))       
            qm1 = Normal(loc=tf.matmul(X,v) + a[0],scale = sigma_q[0]*tf.ones([N,1],tf.float32)) 
            qm2 = Normal(loc=tf.matmul(X,v) + a[1],scale = sigma_q[1]*tf.ones([N,1],tf.float32)) 
            qm3 = Normal(loc=tf.matmul(X,v) + a[2],scale = sigma_q[2]*tf.ones([N,1],tf.float32)) 
            qm4 = Normal(loc=tf.matmul(X,v) + a[3],scale = sigma_q[3]*tf.ones([N,1],tf.float32)) 
            qm5 = Normal(loc=tf.matmul(X,v) + a[4],scale = sigma_q[4]*tf.ones([N,1],tf.float32)) 
                   
            Z1 = tf.sigmoid(tf.abs(sm1)*(T-qm1))*Mask[0,:,:]     
            Z2 = tf.sigmoid(tf.abs(sm1)*(T-qm2))*Mask[1,:,:]      
            Z3 = tf.sigmoid(tf.abs(sm1)*(T-qm3))*Mask[2,:,:]     
            Z4 = tf.sigmoid(tf.abs(sm1)*(T-qm4))*Mask[3,:,:]      
            Z5 = tf.sigmoid(tf.abs(sm1)*(T-qm5))*Mask[4,:,:]      
          
            Yhat1 = Normal(loc=(-c[0]*Z1 + h[0])*Mask[0,:,:],scale = sigma_y[0]*tf.ones([N,M],tf.float32))
            Yhat2 = Normal(loc=(c[1]*Z2 + h[1])*Mask[1,:,:],scale = sigma_y[1]*tf.ones([N,M],tf.float32))
            Yhat3 = Normal(loc=(c[2]*Z3 + h[2])*Mask[2,:,:],scale = sigma_y[2]*tf.ones([N,M],tf.float32))
            Yhat4 = Normal(loc=(c[3]*Z4 + h[3])*Mask[3,:,:],scale = sigma_y[3]*tf.ones([N,M],tf.float32))
            Yhat5 = Normal(loc=(c[4]*Z5 + h[4])*Mask[4,:,:],scale = sigma_y[4]*tf.ones([N,M],tf.float32))
            
            qq1 = Normal(loc=tf.Variable(tf.random_normal([N,1])),
                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,1]))),name='qq1')
            qq2 = Normal(loc=tf.Variable(tf.random_normal([N,1])),
                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,1]))),name='qq2')
            qq3 = Normal(loc=tf.Variable(tf.random_normal([N,1])),
                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,1]))),name='qq3')
            qq4 = Normal(loc=tf.Variable(tf.random_normal([N,1])),
                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,1]))),name='qq4')
            qq5 = Normal(loc=tf.Variable(tf.random_normal([N,1])),
                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,1]))),name='qq5')   
            qs1 = Normal(loc=tf.Variable(tf.random_normal([N,1])),
                 scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,1]))),name='qs1')    
           
            latent_vars = {qm1:qq1, qm2:qq2,qm3:qq3,qm4:qq4,qm5:qq5, sm1:qs1 } 
                    
            data = {X:X_train, T:T_train, Yhat1:Y_train1, Yhat2:Y_train2, Yhat3:Y_train3, 
                    Yhat4:Y_train4,Yhat5:Y_train5, Mask:Mask_train}        
            inference = ed.KLqp(latent_vars,data)
            inference.run(n_iter=50000)  
            X_val =  tf.cast(X_val,tf.float32)
            q11 = tf.matmul(X_val,v) + a[0]
            q22 = tf.matmul(X_val,v) + a[1]
            q55 = tf.matmul(X_val,v) + a[4]
            s1 = tf.matmul(X_val,w) + b
            
            zval1 = Mask_val[0,:,:]*(tf.sigmoid((T_val-q11)*s1))
            Yval1 = -c[0]*zval1+h[0]
            zval2 = Mask_val[1,:,:]*tf.sigmoid((T_val-q22)*s1) 
            Yval2 = c[1]*zval2+h[1]
            zval5 = Mask_val[2,:,:]*tf.sigmoid((T_val-q55)*s1) 
            Yval5 = c[4]*zval5+h[4]
            n1 = tf.count_nonzero(Mask_val[0,:,:])
            n2 = tf.count_nonzero(Mask_val[1,:,:])    
            n5 = tf.count_nonzero(Mask_val[2,:,:])  
            err1 = (abs(Mask_val[0,:,:]*(Y_val1[:,:]-Yval1[:,:])))
            err2 = (abs(Mask_val[1,:,:]*(Y_val2[:,:]-Yval2[:,:])))
            err5 = (abs(Mask_val[2,:,:]*(Y_val5[:,:]-Yval5[:,:])))
            err1 = tf.reduce_sum(err1)
            err2 = tf.reduce_sum(err2)
            err5 = tf.reduce_sum(err5)
            err1 = err1.eval()/n1.eval()
            err2 = err2.eval()/n2.eval()
            err5 = err5.eval()/n5.eval()
            if (err1+err2+err5)<(err10+err20+err50):
                  v0 = v
                  w0 = w
                  a0 = a
                  b0 = b
                  c0 = c
                  h0 = h
                  sigma_s0 = sigma_s
                  sigma_q0 = sigma_q
                  sigma_y0 = sigma_y
                  err10 = err1
                  err20 = err2
                  err50 = err5
                 
            saver = tf.train.Saver()
            sess = ed.get_session()
            save_path = saver.save(sess, "./posterior.ckpt")
            print("Inference model saved in file: %s" % save_path) 
            np.savetxt( "a0.txt", a0.eval(), delimiter=",")
            np.savetxt( "b0.txt", b0.eval(), delimiter=",")
            np.savetxt( "c0.txt", c0.eval(), delimiter=",")
            np.savetxt( "h0.txt", h0.eval(), delimiter=",")
            np.savetxt( "w0.txt", w0.eval(), delimiter=",")
            np.savetxt( "v0.txt", v0.eval(), delimiter=",")
            np.savetxt( "sigma_s0.txt", sigma_s0.eval(), delimiter=",")
            np.savetxt( "sigma_q0.txt", sigma_q0.eval(), delimiter=",")
            np.savetxt( "sigma_y0.txt", sigma_y0.eval(), delimiter=",")
            np.savetxt( "Y1_test.txt", Y_test1*20, delimiter=",")
            np.savetxt( "Y2_test.txt", Y_test2*50, delimiter=",")
            np.savetxt( "Y5_test.txt", Y_test5*10, delimiter=",")
            np.savetxt( "X_test.txt", X_test, delimiter=",")
            np.savetxt( "X_data.txt", X_data, delimiter=",")
            np.savetxt( "T_test.txt", T_test*20+60, delimiter=",")
    # testing using MAP to estimate s and q