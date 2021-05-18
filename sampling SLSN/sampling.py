import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from snad.load.curves import OSCCurve
import os


temp = os.path.abspath("SLSN.csv")
name = pd.read_csv(temp, sep=",")
name = pd.DataFrame(name)
name = name["Name"]

sn=[]
# for i in range(len(name)):
#     sn.append( OSCCurve.from_name(name[i], 
#                 down_args={"baseurl": "https://sne.space/sne/"}) )
name = np.delete( np.array(name), [108])
for i in range(len(name)):
    try:
        sn.append( OSCCurve.from_json( os.path.join('./sne', name[i] + '.json') ))
        sn[-1] = sn[-1].filtered(with_upper_limits=False, with_inf_e_flux=False,
                               sort='filtered')
    except FileNotFoundError:
        continue

Bands = ['u','g','r','i','z',"u'","g'","r'","i'","z'",'U','B','V','R','I']
data = [ [name[i],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for i in range(len(sn))]

table = pd.DataFrame( data=data, columns=['Name','N','u','g','r','i','z',
                                          "u'","g'","r'","i'","z'",'U',
                                          'B','V','R','I'] )

for i in range(len(sn)):
    try:
        for b in Bands:
            for b_in_sn in sn[i].bands:
                if b==b_in_sn:
                    try: #почему то в некоторых есть пустые фильтры
                        temp = OSCCurve.from_json( os.path.join('./sne', name[i] + '.json'),
                                          bands= b_in_sn )
                        temp = temp.filtered(with_upper_limits=False, with_inf_e_flux=False,
                               sort='filtered')
                        table.loc[ table["Name"]==name[i], b_in_sn ] = len(temp.y)
                    except:
                         continue
                    
    except FileNotFoundError:
        continue


N = table[['u','g','r','i','z',"u'","g'","r'","i'","z'",'U','B','V',
           'R','I']].sum(axis='columns')
table["N"] = N
table = table.loc[ table["N"]!=0 ]
#table.to_csv('table_SLSN_redacted.csv')

#минимум 3 полосы пропускания, в каждой из которых минимум 3 значения
cut = table
for i in cut.index:
    flag = 0
    for b in Bands:
        if cut[b][i] >= 3:
            flag = flag + 1
    if flag < 3:
        cut = cut.drop(i)

cut.to_csv('cut.csv')

##########################################################
#####plots

def plot_lc(name, colors):
    sn = OSCCurve.from_json(os.path.join('./sne', name + '.json'), bands=colors.keys())
    plt.figure()
    plt.xlabel('MJD')  # Modified Julian data, days
    plt.ylabel('Flux')
    plt.title(name)
    for band, light_curve in sn.items():
        normal = light_curve[(np.isfinite(light_curve.err)) & (~light_curve.isupperlimit)]
        wo_errors = light_curve[(~np.isfinite(light_curve.err)) & (~light_curve.isupperlimit)]
        upper_limit = light_curve[light_curve.isupperlimit]
        plt.errorbar(normal.x, normal.y, normal.err, marker='x', ls='', color=colors[band])
        plt.plot(wo_errors.x, wo_errors.y, '*', color=colors[band], label=band)
        plt.plot(upper_limit.x, upper_limit.y, 'v', color=colors[band], label=band)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2 )
    plt.grid()

name_cut = cut["Name"].tolist()

col = {"u'": 'blue',"g'": 'green', "r'": 'red', "i'": 'brown',"z'": 'black',
       "u": 'blue',"g": 'green', "r": 'red', "i": 'brown',"z": 'black',
       "U": 'blue',"B": 'green', "V": 'red', "R": 'brown',"I": 'black'}

for n in name_cut:
    band = OSCCurve.from_json( os.path.join('./sne', n + '.json') ).bands
    col_ = {}
    for b in band:
        if col.get(b) != None:
            col_[b] = col.get(b)                                   

    plot_lc(n, col_) 
    plt.savefig( str(n + '.png'), bbox_inches="tight",dpi=200 )
