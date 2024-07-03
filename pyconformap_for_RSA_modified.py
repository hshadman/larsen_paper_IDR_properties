#these packages are needed to run the code 
print('it is incomplete')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import spatial
from itertools import chain
from more_itertools import sliced
from matplotlib.ticker import MaxNLocator
import random as rd

def RSA_based_fC(protein_var,protein_name,poly_id,p_moments_var,
                       GW_moment_var,upto_ind_run,every_ith_snap,GW_every_ith_snap,radius_):
    print('moments_df and ratio_df must match in order of rows for protein/pei')
    x_total=[]
    y_total=[]
    if poly_id=='protein':
        protein_label='protein_'+protein_name
        temp_protein=protein_var.copy()
        temp_protein['RSA']=p_moments_var.RSA.values
        for sim in temp_protein.sim.unique():
            rg_mean=temp_protein.Rg.mean()
            x_total.append(temp_protein[temp_protein.sim==sim].Rg.values/rg_mean)
            y_total.append(temp_protein[temp_protein.sim==sim].RSA.values)                    
        del rg_mean
        x_total=list(chain.from_iterable(x_total))
        y_total=list(chain.from_iterable(y_total))
        poly_var=protein_var.copy()
        poly_var['RSA']=p_moments_var.RSA.values
        poly_var['Rg/Rg_mean']=x_total        
        protein_pro=poly_var[['Rg/Rg_mean','RSA']].iloc[:every_ith_snap,:].copy()
        protein_pro['polymer_id']=np.repeat(protein_label,protein_pro.shape[0])
        del x_total, y_total, temp_protein
        
    elif poly_id=='pei':
        print('for PEI include proton state in input')
        protein_label='pei_'+protein_name
#         x=testeq_pol['Rg/Rg_mean']
#         y=testeq_pol.RSA
#         x_total.append(x)
#         y_total.append(y)
#         x_total=list(chain.from_iterable(x_total))
#         y_total=list(chain.from_iterable(y_total))        
        poly_var=protein_var.copy()
        poly_var['RSA']=p_moments_var.RSA.values
        protein_pro=poly_var[['Rg/Rg_mean','RSA']].iloc[:every_ith_snap,:].copy()
        protein_pro['polymer_id']=np.repeat(protein_label,protein_pro.shape[0])
        del poly_var
    else:
        return print('ERROR')
    GW_po=GW_moment_var[GW_moment_var.run_number<=upto_ind_run][['Rg/Rg_mean','RSA']][::GW_every_ith_snap].copy()
    GW_po['polymer_id']=np.repeat('GW',GW_po.shape[0])
    
    #calculate mean and stdev values (must keep same mean and stdev values)
    upto_snapshots=720000
    GW_mean_Rg_Rg_mean=np.mean(GW_moments_ind_runs_100_['Rg/Rg_mean'].values[0:(upto_snapshots+1)])
    GW_std_Rg_Rg_mean=np.std(GW_moments_ind_runs_100_['Rg/Rg_mean'].values[0:(upto_snapshots+1)])
    GW_mean_RSA=np.mean(GW_moments_ind_runs_100_['RSA'].values[0:(upto_snapshots+1)])
    GW_std_RSA=np.std(GW_moments_ind_runs_100_['RSA'].values[0:(upto_snapshots+1)])
        
    combined_pro_po=pd.concat([GW_po,protein_pro],axis=0,ignore_index=True)
    combined_pro_po['stdd_Rg/Rg_mean']=(combined_pro_po['Rg/Rg_mean'].values-GW_mean_Rg_Rg_mean)/(GW_std_Rg_Rg_mean)
    combined_pro_po['stdd_RSA']=(combined_pro_po['RSA'].values-GW_mean_RSA)/(GW_std_RSA)
    po_x=combined_pro_po[combined_pro_po.polymer_id=='GW']['stdd_Rg/Rg_mean'].values
    po_y=combined_pro_po[combined_pro_po.polymer_id=='GW']['stdd_RSA'].values
    pro_x=combined_pro_po[combined_pro_po.polymer_id==protein_label]['stdd_Rg/Rg_mean'].values
    pro_y=combined_pro_po[combined_pro_po.polymer_id==protein_label]['stdd_RSA'].values
    
    GW_points=np.c_[po_x, po_y]
    protein_points=np.c_[pro_x, pro_y]    
    tree_GW=spatial.cKDTree(GW_points)
    tree_protein=spatial.cKDTree(protein_points)

    GW_not_in_range=[]
    j=0
    for point in GW_points:

        if not tree_protein.query_ball_point(point,radius_):
            GW_not_in_range.append(point)
        j+=1
        if j%100000==0:
            print(f'{j} GW snapshots completed')
            
    fA_by_distance=(GW_points.shape[0]-len(GW_not_in_range))/(GW_points.shape[0])
    return fA_by_distance    
