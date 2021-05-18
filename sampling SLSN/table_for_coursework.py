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

name = name["Name"].tolist()
name = np.delete(name,[3,4])
sn=[[] for i in range(27)]

for i in range(len(sn)):
    sn[i] = OSCCurve.from_json(os.path.join('./sne', name[i] + '.json'))

data = [ [name[i],0,0,0,0,0] for i in range(len(sn))]
table = pd.DataFrame( data=data, columns=['Name','type','ra','dec','redshift',
                                          'bands'], dtype='string')

Bands = ['u','g','r','i','z',"u'","g'","r'","i'","z'",'U','B','V','R','I']
for i in range(len(sn)):
    b = list(set(Bands) & set(sn[i].bands))
    try:
        table.at[ i, 'ra' ] = sn[i].ra[0]
        table.at[ i, 'dec' ] = sn[i].dec[0]
        table.at[ i, 'redshift' ] = str(sn[i].redshift[0])
        table.at[ i, 'bands' ] = ','.join(np.array(b))
        table.at[ i, 'type' ] = ','.join(np.array(sn[i].claimedtype))
    except IndexError:
        table.at[ i, 'ra' ] = '-'
        table.at[ i, 'dec' ] = '-'
        table.at[ i, 'redshift' ] = '-'
        table.at[ i, 'bands' ] = ','.join(np.array(b))
        table.at[ i, 'type' ] = '-'

table.to_csv('table_for_coursework.csv')