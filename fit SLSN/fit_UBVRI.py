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

temp = os.path.abspath("cut.csv")
name = pd.read_csv(temp, sep=",")
name = pd.DataFrame(name)


res=3
sn=[]

# for i in range(29):
#     sn.append( OSCCurve.from_name(name['Name'][i], 
#                 down_args={"baseurl": "https://sne.space/sne/"}) )

bands = ['U','B','V','R','I']

for i in range(12):
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
    
#sn = np.delete(sn,9)


for i in [0,1,2,3,4,5,6,8,10,11]:
    kern_size = len( sn[i].bands )
    if kern_size==3:
        const_matrix = np.array([[1,0,0],
                             [0.5,1,0],
                             [0.5,0.5,1]])

        bounds = [np.array([[1e-4,0,0],
                        [-1e3,-1e3,0],
                        [-1e3,-1e3,-1e3]]),
           np.array([[1e4,0,0],
                     [1e3,1e3,0],
                     [1e3,1e3,1e3]])]
    if kern_size==4:
        const_matrix = np.array([[1,0,0,0],
                             [0.5,1,0,0],
                             [0.5,0.5,1,0],
                             [0.5,0.5,0.5,1]])

        bounds = [np.array([[1e-4,0,0,0],
                        [-1e3,-1e3,0,0],
                        [-1e3,-1e3,-1e3,0],
                        [-1e3,-1e3,-1e3,-1e3]]),
           np.array([[1e+4,0,0,0],
                        [1e3,1e3,0,0],
                        [1e3,1e3,1e3,0],
                        [1e3,1e3,1e3,1e3]])]
    if kern_size==5:
        const_matrix = np.array([[1,0,0,0,0],
                             [0.5,1,0,0,0],
                             [0.5,0.5,1,0,0],
                             [0.5,0.5,0.5,1,0],
                             [0.5,0.5,0.5,0,1]])

        bounds = [np.array([[1e-4,0,0,0,0],
                        [-1e3,-1e3,0,0,0],
                        [-1e3,-1e3,-1e3,0,0],
                        [-1e3,-1e3,-1e3,-1e3,0],
                        [-1e3,-1e3,-1e3,-1e3,-1e3]]),
           np.array([[1e+4,0,0,0,0],
                        [1e3,1e3,0,0,0],
                        [1e3,1e3,1e3,0,0],
                        [1e3,1e3,1e3,1e3,0],
                        [1e3,1e3,1e3,1e3,1e3]])]
    
    mk = MultiStateKernel( [RBF(1,(1e-4, 1e+4)) for k in range(kern_size)],
                          const_matrix, bounds )
####################################################################
        
    x,y=[],[]
    x=sn[i].X
    y=sn[i].y
    err=sn[i].err
    
    mask = []
    for j in range(kern_size):
        mask.append( (x[:,0] == j) )


    col = {"U": 'purple',"B": 'blue', "V": 'green', "R": 'red',"I": 'brown'}

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error', category=ConvergenceWarning)
    #     try:
    err1 = err**2 
    err1 = err1 + [(y[e]/10)**2 if e>7 else 0 for e in range(len(err))]
    gpr = GaussianProcessRegressor(kernel=mk, alpha=err1, 
                                    n_restarts_optimizer=res).fit(x, y)
            
                    
    fig, ax = plt.subplots()#figsize=(18, 12),dpi=400
    plt.title( name['Name'][i] )
            
    b = sn[i].bands
    if kern_size==3:
        X = np.linspace(min(x[:,1]), max(x[:,1]),1000).reshape(-1,1)
        X = np.block([[np.zeros_like(X), X],
                     [np.ones_like(X), X],
                     [2*np.ones_like(X), X]])
        predict = gpr.predict(X)
                
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[:1000], col.get(b[0]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[1000:2000], col.get(b[1]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[2000:],col.get(b[2]))
        plt.errorbar(x[mask[0],1], y[mask[0]], np.sqrt(err1[mask[0]]), marker='x', ls='', color=col.get(b[0]),ms=3)
        plt.errorbar(x[mask[1],1], y[mask[1]], np.sqrt(err1[mask[1]]), marker='x', ls='', color=col.get(b[1]),ms=3)
        plt.errorbar(x[mask[2],1], y[mask[2]], np.sqrt(err1[mask[2]]), marker='x',color=col.get(b[2]), ls='',ms=3)
        ax.set_xlabel('mjd')
        ax.set_ylabel('Flux')
                
    if kern_size==4:
        X = np.linspace(min(x[:,1]), max(x[:,1]),1000).reshape(-1,1)
        X = np.block([[np.zeros_like(X), X],
                      [np.ones_like(X), X],
                      [2*np.ones_like(X), X],
                      [3*np.ones_like(X), X]])
        predict = gpr.predict(X)
        
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[:1000], col.get(b[0]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[1000:2000], col.get(b[1]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[2000:3000],col.get(b[2]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[3000:],col.get(b[3]))
        plt.errorbar(x[mask[0],1], y[mask[0]], np.sqrt(err1[mask[0]]), marker='x', ls='', color=col.get(b[0]),ms=3)
        plt.errorbar(x[mask[1],1], y[mask[1]], np.sqrt(err1[mask[1]]), marker='x', ls='', color=col.get(b[1]),ms=3)
        plt.errorbar(x[mask[2],1], y[mask[2]], np.sqrt(err1[mask[2]]), marker='x',color=col.get(b[2]), ls='',ms=3)
        plt.errorbar(x[mask[3],1], y[mask[3]], np.sqrt(err1[mask[3]]), marker='x',color=col.get(b[3]), ls='',ms=3)
        ax.set_xlabel('mjd')
        ax.set_ylabel('Flux')
            
    if kern_size==5:
        X = np.linspace(min(x[:,1]), max(x[:,1]),1000).reshape(-1,1)
        X = np.block([[np.zeros_like(X), X],
                      [np.ones_like(X), X],
                      [2*np.ones_like(X), X],
                      [3*np.ones_like(X), X],
                      [4*np.ones_like(X), X]])
        predict = gpr.predict(X)
                
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[:1000], col.get(b[0]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[1000:2000], col.get(b[1]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[2000:3000],col.get(b[2]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[3000:4000],col.get(b[3]))
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),predict[4000:],col.get(b[4]))
        plt.errorbar(x[mask[0],1], y[mask[0]], np.sqrt(err1[mask[0]]), marker='x', ls='', color=col.get(b[0]),ms=3)
        plt.errorbar(x[mask[1],1], y[mask[1]], np.sqrt(err1[mask[1]]), marker='x', ls='', color=col.get(b[1]),ms=3)
        plt.errorbar(x[mask[2],1], y[mask[2]], np.sqrt(err1[mask[2]]), marker='x',color=col.get(b[2]), ls='',ms=3)
        plt.errorbar(x[mask[3],1], y[mask[3]], np.sqrt(err1[mask[3]]), marker='x',color=col.get(b[3]), ls='',ms=3)
        plt.errorbar(x[mask[4],1], y[mask[4]], np.sqrt(err1[mask[4]]), marker='x',color=col.get(b[4]), ls='',ms=3)
        ax.set_xlabel('mjd')
        ax.set_ylabel('Flux')
            
    t = np.array(b).tolist()
    t = [ 'error '+b[l] for l in range(kern_size) ]
    lab = np.array(b).tolist()+t
    plt.legend(lab,framealpha=0.0,loc='upper right',ncol=2)
    temp =  str(name['Name'][i])+'_UBVRI' + '.png'
    fig.savefig( fname = temp  , bbox_inches="tight")# , transparent=True

        # except Warning:
        #     continue
                       
    
    