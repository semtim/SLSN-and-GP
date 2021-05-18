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

# for i in range(29):
#     sn.append( OSCCurve.from_name(name['Name'][i], 
#                 down_args={"baseurl": "https://sne.space/sne/"}) )
i=10
bands = ["U","B","V","R","I"]

temp = ''
current_bands = OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json')).bands
for b in bands:
    for b_sn in current_bands:
        if b_sn==b:
            temp = temp + b_sn + ','
temp = temp[:-1]
sn = OSCCurve.from_json(os.path.join('./sne', name['Name'][i] + '.json'), bands=temp)
sn = sn.filtered(with_upper_limits=False, with_inf_e_flux=False, sort='filtered')
sn = sn.binned(bin_width=1, discrete_time=True)
    

kern_size = len( sn.bands )
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

x,y=[],[]
x=sn.X
y=sn.y
err=sn.err
    
mask = []
for j in range(kern_size):
    mask.append( (x[:,0] == j) )
    


col = {"U": 'purple',"B": 'blue', "V": 'green', "R": 'red',"I": 'brown'}

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error', category=ConvergenceWarning)
    #     try:
err1 = err**2 
err1 = err1 + [(y[e]/10)**2 if e>10 else 0 for e in range(len(err))]
gpr = GaussianProcessRegressor(kernel=mk, alpha=err1, 
                                    n_restarts_optimizer=res).fit(x, y)

days = 121
X = np.linspace(min(x[:,1])-10, 54710.5,days).reshape(-1,1)
X = np.block([[np.zeros_like(X), X],
                      [np.ones_like(X), X],
                      [2*np.ones_like(X), X],
                      [3*np.ones_like(X), X],
                      [4*np.ones_like(X), X]])

predict, sigma = gpr.predict(X, return_std =True)
data_pred = sn.convert_arrays(X, predict, sigma)

steps = 40
s=3
T = np.linspace(min(x[:,1])-10, 54710.5,days)
f = open(name['Name'][i]+"_U.dat", 'w')
for t in range(steps):
     f.write(str(T[t*s]))
     f.write(' ')
     f.write( str( -2.5*np.log10(data_pred.odict["U"].y)[t*s] ) )
     f.write(' ')
     f.write('0')
     #f.write( str( np.sqrt(-2.5*np.log10(data_pred.odict["r'"].err)[t] )) )
     f.write('\n')
f.close()


f = open(name['Name'][i]+"_B.dat", 'w')
for t in range(steps):
     f.write(str(T[s*t]))
     f.write(' ')
     f.write( str( -2.5*np.log10(data_pred.odict["B"].y)[t*s] ) )
     f.write(' ')
     f.write('0')
     #f.write( str( np.sqrt(-2.5*np.log10(data_pred.odict["i'"].err)[t] )) )
     f.write('\n')
f.close()

f = open(name['Name'][i]+"_V.dat", 'w')
for t in range(steps):
     f.write(str(T[t*s]))
     f.write(' ')
     f.write( str( -2.5*np.log10(data_pred.odict["V"].y)[t*s] ) )
     f.write(' ')
     f.write('0')
     #f.write( str( np.sqrt(-2.5*np.log10(data_pred.odict["g'"].err)[t] )) )
     f.write('\n')
f.close()

f = open(name['Name'][i]+"_R.dat", 'w')
for t in range(steps):
     f.write(str(T[t*s]))
     f.write(' ')
     f.write( str( -2.5*np.log10(data_pred.odict["R"].y)[t*s] ) )
     f.write(' ')
     f.write('0')
     #f.write( str( np.sqrt(-2.5*np.log10(data_pred.odict["z'"].err)[t] )) )
     f.write('\n')
f.close()

f = open(name['Name'][i]+"_I.dat", 'w')
for t in range(steps):
      f.write(str(T[s*t]))
      f.write(' ')
      f.write( str( -2.5*np.log10(data_pred.odict["I"].y)[t*s] ) )
      f.write(' ')
      f.write('0')
      #f.write( str( np.sqrt(-2.5*np.log10(data_pred.odict["u'"].err)[t] )) )
      f.write('\n')
f.close()