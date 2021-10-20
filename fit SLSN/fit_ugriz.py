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
from scipy import stats

temp = os.path.abspath("cut.csv")
name = pd.read_csv(temp, sep=",")
name = pd.DataFrame(name)


res=3
sn=[[] for i in range(29)]

# for i in range(29):
#     sn.append( OSCCurve.from_name(name['Name'][i], 
#                 down_args={"baseurl": "https://sne.space/sne/"}) )

bands = ["u","g","r","i","z"]
cut = [1,5,7,8,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
for i in cut:
    temp = ''
    current_bands = OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json')).bands
    for b in bands:
        for b_sn in current_bands:
            if b_sn==b:
                temp = temp + b_sn + ','
    temp = temp[:-1]
    sn[i] = OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json'), bands=temp)
    sn[i] = sn[i].filtered(with_upper_limits=False, with_inf_e_flux=False, sort='filtered')
    sn[i] = sn[i].binned(bin_width=1, discrete_time=True)
    

for i in cut:
    kern_size = len( sn[i].bands )
    
    const_matrix = np.eye(kern_size)
    bound_min = np.eye(kern_size) * (-1e1)
    bound_max = np.eye(kern_size) * (1e1)
    for q in range(kern_size):
        const_matrix[q, 0:q] = 1/2
        bound_min[q, 0:q] = -1e1
        bound_max[q, 0:q] = 1e1
    bounds = [bound_min, bound_max]
    
    size_sc = 5
    scales = stats.expon.rvs(size=size_sc, loc=1e-5, scale = 10)
    mk = [MultiStateKernel( [RBF(scales[k],(1e-5,1e3)) for k in range(kern_size)],
                          const_matrix, bounds ) for k in range(size_sc)]
####################################################################
        
    x,y=[],[]
    x=sn[i].X
    y=sn[i].y
    err=sn[i].err
    ############################
    #Удаление дальних от пика точек для некоторых кривых
    # x = np.array([ x[t] for t in range(len(y)) if t!=36])
    # y = np.delete(y,36)
    # err = np.delete(err,36)
    ############################

    mask = []
    for j in range(kern_size):
        mask.append( (x[:,0] == j) )
    
###############################################
    # y_mean = np.array([ np.mean(y*mask[j]) for j in range(kern_size) ])
    # for j in range(kern_size):
    #     y[mask[j]] = y[mask[j]]-y_mean[j]
###############################################
    col = {"u": 'purple',"g": 'green', "r": 'red', "i": 'brown',"z": 'black'}

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error', category=ConvergenceWarning)
    #     try:
    err1 = err**2 + (y/10)**2
    
    gpr = [ GaussianProcessRegressor(kernel=mk[k], alpha=err1, 
                                    n_restarts_optimizer=res).fit(x, y)
           for k in range(size_sc) ]
    
    lml = [ gpr[k].log_marginal_likelihood_value_ for k in range(size_sc) ]
    ind = np.argmax(lml)
    gpr = gpr[ind]
                    
    fig, ax = plt.subplots()#figsize=(18, 12),dpi=400
    plt.title( name['Name'][i] )
            
    b = sn[i].bands
    
    X = np.linspace(min(x[:,1]), max(x[:,1]),1000).reshape(-1,1)
    X = np.block([ [u*np.ones_like(X), X] for u in range(kern_size) ] )

    predict = gpr.predict(X)
    
    # for j in range(kern_size):
    #     predict[1000*j:1000*j+1000] = predict[1000*j:1000*j+1000] + y_mean[j]
    #     y[mask[j]] = y[mask[j]]+y_mean[j]
    
    for u in range(kern_size):
        ax.plot(np.linspace(min(x[:,1]), max(x[:,1]), 1000),
                predict[1000*u:1000*u+1000], col.get(b[u]))
        
        plt.errorbar(x[mask[u],1], y[mask[u]], np.sqrt(err1[mask[u]]),
                     marker='x', ls='', color=col.get(b[u]),ms=3)
    ax.set_xlabel('mjd')
    ax.set_ylabel('Flux')
    

            
    t = np.array(b).tolist()
    t = [ 'error '+b[l] for l in range(kern_size) ]
    lab = np.array(b).tolist()+t
    plt.legend(lab,framealpha=0.0,loc='upper right',ncol=2)
    # temp =  str(name['Name'][i])+'_ugriz' + '.png'
    # fig.savefig( fname = temp  , bbox_inches="tight")# , transparent=True

#print(gpr.kernel_.get_params()) # optimized hyperparameters
#print(gpr.log_marginal_likelihood_value_)
        # except Warning:
        #     continue