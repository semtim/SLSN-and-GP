import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multistate_kernel import MultiStateKernel
from snad.load.curves import OSCCurve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,Matern,ConstantKernel,RationalQuadratic
import os
import math
import warnings
from sklearn.exceptions import ConvergenceWarning
import time

#np.random.seed(43)  #случайные значения не меняются
temp = os.path.abspath("cut.csv")
name = pd.read_csv(temp, sep=",")
name = pd.DataFrame(name)


res=3
sn=[]
warning = []

bands = ['u','g','r','i','z',"u'","g'","r'","i'","z'",'U','B','V','R','I']
for i in range(29):
    temp = ''
    current_bands = OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json')).bands
    for b in bands:
        for b_sn in current_bands:
            if b_sn==b:
                temp = temp + b_sn + ','
    temp = temp[:-1]
    sn.append( OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json'), bands=temp) )
    sn[i] = sn[i].filtered(with_upper_limits=False, with_inf_e_flux=False, sort='filtered')
    sn[i] = sn[i].binned(bin_width=1, discrete_time=True)

# l=[]
# for i in range(100):
#     if len(sn[i].y)<=50:
#         l.append(i)

#flag = [False for i in range(len(sn))]
for i in range(29):
    rbf =  RBF(1,(1e-4, 1e+4))
    c1 = ConstantKernel(constant_value=1.0, constant_value_bounds='fixed')
    c2 = ConstantKernel(constant_value=1.0, constant_value_bounds='fixed')
    noise1 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
    noise2 = WhiteKernel(noise_level=1, noise_level_bounds='fixed')
    rbf2 =  RBF(1,(1e-4, 1e+4))
    rbf3 =  RBF(1,(1e-4, 1e+4))

    const_matrix = np.array([[1,0,0],
                             [0.5,1,0],
                             [0.5,0.5,1]])

    bounds = [np.array([[1e-4,0,0],
                        [-1e3,-1e3,0],
                        [-1e3,-1e3,-1e3]]),
           np.array([[1e4,0,0],
                     [1e3,1e3,0],
                     [1e3,1e3,1e3]])]

    mk1 = MultiStateKernel((rbf, c1,rbf2),const_matrix,bounds)
    mk2 = MultiStateKernel((rbf, rbf2, c1),const_matrix,bounds)
    mk3 = MultiStateKernel((rbf, c1,c2),const_matrix,bounds) #mk2 = MultiStateKernel((k2, k1,noise),const_matrix,bounds)
    mk4 = MultiStateKernel((rbf,rbf2,rbf3),const_matrix,bounds)

    kernels = [ [mk1, 'rbf, c, rbf'], [mk2, 'rbf, rbf, c'] 
           ,[mk3, 'rbf, c, c'], [mk4, 'rbf, rbf, rbf']]
####################################################################
        
    x,y=[],[]
    x=sn[i].X
    y=sn[i].y
    err=sn[i].err
    
    mask1 = (x[:,0] == 0)
    mask2 = (x[:,0] == 1)
    mask3 = (x[:,0] == 2)
    X = np.linspace(min(x[:,1]), max(x[:,1]),1000).reshape(-1,1)
    X = np.block([[np.zeros_like(X), X],
                  [np.ones_like(X), X],
                  [2*np.ones_like(X), X]])

    s=0
    for j in range(np.size(err)):
        if math.isnan(err[j]):
            err[j]=1e-6


    for mk in kernels:
        for add in [0,1]:
            #if flag[i]==False:
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('error', category=ConvergenceWarning)
                #     try:
                        err1 = err**2 + (add*y/10)**2
                        gpr = GaussianProcessRegressor(kernel=mk[0], alpha=err1, 
                                    n_restarts_optimizer=res).fit(x, y)
                        predict = gpr.predict(X)
                    
                        fig, ax = plt.subplots(figsize=(18, 12),dpi=400)#figsize=(18, 12),dpi=400
                        #plt.title( name['Name'][i] + '\n' +  mk[1] + '\n' + 'add_err = ' + str(add) )
                        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[:1000], 'r')
                        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[1000:2000], 'g')
                        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[2000:],'b')
                        plt.errorbar(x[mask1,1], y[mask1], np.sqrt(err1[mask1]), marker='x', ls='', color='r',ms=3)
                        plt.errorbar(x[mask2,1], y[mask2], np.sqrt(err1[mask2]), marker='x', ls='', color='g',ms=3)
                        plt.errorbar(x[mask3,1], y[mask3], np.sqrt(err1[mask3]), marker='x',color='b', ls='',ms=3)
                        ax.set_xlabel('mjd')
                        ax.set_ylabel('Flux')
                        lab = ['полоса пропускания g', 'полоса пропускания r', 'полоса пропускания i', 'фотометрические наблюдения g','фотометрические наблюдения r','фотометрические наблюдения i']
                        plt.legend(lab,framealpha=0.0,loc='upper right',ncol=2)
                        temp =  str(name['Name'][i])+'2' + '.png'
                        fig.savefig( fname = temp  , bbox_inches="tight" , transparent=True)
                        #flag[i] = True
                    # except Warning:
                    #     continue
                       
    # # print(gpr2.kernel_.get_params()) # optimized hyperparameters