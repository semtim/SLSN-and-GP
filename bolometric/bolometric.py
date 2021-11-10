import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
import numpy as np
import sncosmo
import os
import pandas as pd
from astropy.coordinates import Distance
import astropy.units as u

######################################################
# bands - ugriz

bands_sdss = ['u','g','r','i','z']
band_sdss = []
for b in bands_sdss:
    band_sdss.append( sncosmo.get_bandpass('sdss'+b) )

#bands - UBVRI
bands_snls = ['u','b','v','r','i']
band_snls = []
for b in bands_snls:
    band_snls.append( sncosmo.get_bandpass('standard::'+b) )

UBVRI = ['U','B','V','R','I']
bands = dict( zip( bands_sdss + UBVRI, band_sdss + band_snls ) )
W = np.linspace( 3000, 11000, 100000 )
maximum_bands = [ max(bands[b](W)) for b in  bands_sdss + UBVRI]
maximum_bands = dict( zip( bands_sdss + UBVRI, maximum_bands ) )
######################################################



h = 6.62607015*1e-34 # кг·м2·с−1
c = 299792458   #  м / с
k = 1.380649*1e-23 # Дж/К
Jy = 1e-26 # W /(Hz * m**2) 

L = np.linspace( 3000, 11000, 100000 ) * 1e-10  # м
L_min, L_max = L[0], L[-1]
W = c / L
W_min, W_max = W[-1], W[0]
# b - фильтр   l - длина волны

def Band(w, b):
    #возвращает пропускание фильтра b на частоте w
    l = c / w
    l_ang = l * 1e+10 #angstrom
    
    return bands[b](l_ang) #/ maximum_bands[b]


def band_Plank(w, b, T, z):
    wz = w + w*z/(z + 1)
    f = Band(w,b) * 2*wz**3/(c**2)/(np.exp( h*wz/(k*T) ) - 1) / w 
    
    return f

def band_Plank_prime(w, b, T, z):
    wz = w + w*z/(z + 1)
    f = Band(w,b) * 2*(h)*(wz**4)/(c**2*k)*np.exp( h*wz/(k*T) )/w \
        / (np.exp( h*wz/(k*T) ) - 1)**2 / T**2
    
    return f

def norm_band(w, b):
    f = 3631*Jy*Band(w,b)/(h*w)
    return f

def band_mag(b, T, z):
    d = Distance(z=z, unit=u.m).to_value()
    R = 1e+13
    f = -2.5*np.log10( integrate.quad(band_Plank, W_min, W_max, args=(b,T,z,))[0] \
        / integrate.quad(norm_band, W_min, W_max, args=(b,))[0] * R**2/ d**2)
    return f

def band_mag_prime(b, T, z):
    f = -2.5 * integrate.quad(band_Plank_prime, W_min, W_max, args=(b,T,z,))[0] \
        / integrate.quad(band_Plank, W_min, W_max, args=(b,T,z))[0]
    return f

def mse(T, data, z):
        list_b = data.index[1:]
        f = np.sum( [ band_mag_prime(b, T, z) * (data[b] - band_mag(b, T, z))  
               for b in list_b ] )
        return f

# data в магнитудах       
temp = os.path.abspath("SN2011ke.csv")
ap_data = pd.read_csv(temp, sep=",")
z = 0.1428

bb_T = fsolve(mse, 10000, args=(ap_data.iloc[100], z,))
