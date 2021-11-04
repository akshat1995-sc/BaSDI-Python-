import numpy as np 
import matplotlib.pyplot as plt
import time
import cv2 as cv
from scipy.stats import multivariate_normal
import scipy.io
from basdi_func import *
import time
from scipy.signal import oaconvolve as conv2


#Load Experimental data
# data=np.sort(np.load("data.npy"),axis=0)
# data=data[data[:,0].argsort()]


#Load synthetic data
temp=scipy.io.loadmat("drift_data.mat")
d_real=scipy.io.loadmat("drift.mat")['d']
d_vit=scipy.io.loadmat("dviterbi.mat")['d_out']
d_matlab=scipy.io.loadmat("dout.mat")['d_out']
data=temp['O'][0]
n_frms=data.shape[0]								#No. of frames
brder_size=30
h=500
w=500
scale=2.4
anneal_step=0.4
resolution=2
max_iter=1
d=np.zeros((n_frms,2))
blur=2

theta=gen_theta(data,h,w)
OC=rmv_border(data,h,w,brder_size)
iter_r=0
while(scale>=1.2):
    c=[0,0]
    iter=0
    print("round -",iter_r)
    while((c[0]==0 | c[1]==0) & iter<max_iter):
    	
    #E-step
        print('E-step')
        fs=np.int(np.round(np.exp(scale)))
        theta_temp=conv2(theta,np.ones((fs,fs)))
        theta2=psf_blur(theta_temp,blur)
        # theta2=theta2_convert(theta, scale, resolution)
        obsv_prob_frames=likelihood(OC,theta2,30)	
        drift_distri=marginalize(obsv_prob_frames)
        
    #M-step
        print("M-step")
        theta=update_theta(data,drift_distri,h,w)
        [d_out,_]=drift_eval(drift_distri)
        c=test_convg(d, d_out)
        d=d_out
        iter+=1
    scale=scale-anneal_step
    iter_r+=1

plt.imshow(theta);plt.show()
plt.figure()
plt.plot(d_out[:,0]);plt.plot(d_real[:,0])
# plt.plot(d_out[:,0])
# plt.errorbar(range(1000), d_out[:,0], yerr=np.abs(d_out[:,0]-d_real[:,0]),color='grey',alpha=0.1)
plt.title('Drift in x-axis')
plt.xlabel('frames')
plt.ylabel('Drift')
# plt.ylim([-10,3])
# plt.legend(['BaSDI output','Python Output'])


#a=np.arange(1,26).reshape(5,5).astype('float').T
#b=np.arange(1,5).reshape(2,2).astype('float').T

