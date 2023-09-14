import pandas as pd
import numpy as np
import pywt
import os
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings 
warnings.filterwarnings("ignore")
import mkl
mkl.set_num_threads(1)
from wandb.keras import WandbCallback
from imblearn.over_sampling import RandomOverSampler
from backbone_model import *
import tensorflow as tf
import time

def ROS(X,Y):
    X_r = X.reshape(X.shape[0], -1)
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(X_r, Y)   
    X_ros=X_ros.reshape(X_ros.shape[0],X.shape[1],X.shape[2])
    return X_ros,y_ros
def baseline_remove(x): 
    y=np.ones([x.shape[0],x.shape[1],x.shape[2]])
    for i in range(x.shape[0]):
      for j in range(x.shape[2]):      
        level=pywt.dwt_max_level(x.shape[1], 'db6')
        coeffs=pywt.wavedec(x[i,:,j],'db6',mode='per',level=level) #Coefficients list [cAn, cDn, cDn-1, …, cD2, cD1]
        coeffs[0] = np.zeros_like(coeffs[0]) #reconstruction without information of cAn
        rec= pywt.waverec(coeffs,'db6',mode='per') 
        y[i,:,j]=rec
    return y

#Config
lr=1e-5
epoch=100
batchsize=64
test_fold = 10

#load processed data
X = np.load("X_100.npy",allow_pickle=True)
Y = pd.read_csv('Y_100.csv') 
lable_list=["['STTC']","['NORM']","['MI']","['CD']","['HYP']"]
Y = Y[Y['diagnostic_superclass'].isin(lable_list)]
X=X[Y.index,:,:] 
Y = Y.reset_index(drop=True)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=int((14620/batchsize))*10,
    decay_rate=0.8)
    

def normolization(X):
    k=X.shape
    X = X.ravel()
    X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_nor.shape=k
    return X_nor 

# preprocessing
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass 
print(y_train.value_counts())
y_train=pd.get_dummies(y_train) 
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
print(y_test.value_counts())
y_test=pd.get_dummies(y_test) #(1652,5)

# baseline_remove
X_train=baseline_remove(X_train)
X_test=baseline_remove(X_test) 
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

# random oversampling
X_train,y_train=ROS(X_train,y_train)


print('Start to train ...')
model=CNN_model((X_train.shape[1], X_train.shape[2]),y_train.shape[-1]) #(1000,1,5)
fractions = 1-y_train.sum(axis=0)/len(y_train)
weights = fractions[y_train.argmax(axis=1)]

#fit the model
time_start = time.time()
hist=model.fit(X_train, y_train, epochs=epoch, batch_size=batchsize, sample_weight=weights,validation_data=(X_test,y_test), callbacks=[WandbCallback(monitor='val_accuracy')],shuffle=True)
time_end = time.time()
print(time_end-time_start)

output=model.predict(X_test) #shape(1652,) 
AUC_ovo=roc_auc_score(y_test, output,multi_class='ovo')
AUC_ovr=roc_auc_score(y_test, output,multi_class='ovr')
print("ovo AUC = ", AUC_ovo)
print("ovr AUC = ", AUC_ovr)
output=np.argmax(output,axis=1)
y_test=np.argmax(y_test,axis=1)
print(confusion_matrix(y_test, output))
print("ACC=",(output == y_test).sum()/len(output))
