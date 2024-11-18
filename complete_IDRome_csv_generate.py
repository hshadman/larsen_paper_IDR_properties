import pandas as pd
import mdtraj as md
import numpy as np
from numpy.random import seed
from numpy.random import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
#from __future__ import print_function
import seaborn as sns
from matplotlib.ticker import NullFormatter, MaxNLocator
from pandas.plotting import scatter_matrix
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import scipy as sp
from itertools import chain
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import spatial
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import scipy.stats as stats
import statsmodels.stats.weightstats
from matplotlib import path
import matplotlib
from scipy.stats import probplot,shapiro, sem
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score, mean_squared_error
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler

from matplotlib import cm
from numpy import linspace
#import umap.umap_ as umap
#import pylab
import os
#import PIL
#from scipy.ndimage import gaussian_filter, uniform_filter1d

seq_name_AFRC = pd.read_csv('../holehouse_project/IDRome_shape_mean_size_mean_added.csv')

seq_name_list = []
seq_name_dir = []
protein_name = []
for root, dirs, files in os.walk('IDROME_larsen/IDRome_v4/', topdown=False):
    for name in files:
        seq_name_list.append(''.join(root.split('/')[2:(len(root.split('/'))-1)]+['_']+[root.split('/')[len(root.split('/'))-1]]))
        protein_name.append(''.join(root.split('/')[2:(len(root.split('/'))-1)]))
        seq_name_dir.append(root)
        break
seq_name_dir_df = pd.DataFrame(zip(seq_name_list,seq_name_dir,protein_name),columns=['seq_name','seq_dir','protein_uniprot_id'])
del seq_name_list, seq_name_dir, protein_name

def RSA_Rs_size_compute_3dplot_from_seq_name_ALL():
    global protein_df_original
    protein_rg2 = np.array([])
    protein_rg = np.array([])
    protein_ree2 = np.array([])
    protein_rg_by_rg_theta = np.array([])
    protein_rg_by_rg_mean = np.array([])
    IDR_seq_name = np.array([])
    j=0
    for seq_name_all in seq_name_dir_df.seq_name.values:
        example_protein_dir = seq_name_dir_df[seq_name_dir_df.seq_name==seq_name_all].seq_dir.values[0]
        t = md.load(f'{example_protein_dir}/traj.xtc', top=f'{example_protein_dir}/top.pdb')
        protein_rg_theta = seq_name_AFRC[seq_name_AFRC.seq_name==seq_name_all].AFRC_mean_rg_theta.values[0]
        protein_rg2 = np.append(protein_rg2,np.load(f'{example_protein_dir}/rg.npy')**2)
        protein_rg = np.append(protein_rg,np.load(f'{example_protein_dir}/rg.npy'))
        rg_mean = np.mean(np.load(f'{example_protein_dir}/rg.npy'))
        protein_ree2 = np.append(protein_ree2,np.load(f'{example_protein_dir}/ete.npy')**2)
        protein_rg_by_rg_theta = np.append(protein_rg_by_rg_theta,(np.load(f'{example_protein_dir}/rg.npy')*10)/protein_rg_theta)
        protein_rg_by_rg_mean = np.append(protein_rg_by_rg_mean,(np.load(f'{example_protein_dir}/rg.npy'))/rg_mean)
        
        #randomly use ete.npy as a shape for seq_name repeats
        IDR_seq_name = np.append(IDR_seq_name,np.repeat(seq_name_all,np.load(f'{example_protein_dir}/ete.npy').shape[0]))
        
        if j==0:
            md_principal_moments = md.principal_moments(t)[10:]
        else:
            md_principal_moments = np.append(md_principal_moments,md.principal_moments(t)[10:],axis=0)
        if j%1000==0:
            print(f'{j} snapshots done')
        j+=1
    t_df_moments = pd.DataFrame(md_principal_moments,columns=['R3','R2','R1']).copy()
    t_df_moments['asphericity']=t_df_moments.R1.values-(0.5*(t_df_moments.R2.values+t_df_moments.R3.values))
    t_df_moments['acylindricity']=t_df_moments.R2.values-t_df_moments.R3.values
    t_df_moments['RSA']=(t_df_moments.asphericity.values**2+(0.75*t_df_moments.acylindricity.values**2))/(t_df_moments.R1.values+t_df_moments.R2.values+t_df_moments.R3.values)**2
    protein_df_original = pd.DataFrame(zip(IDR_seq_name,
                                           protein_rg2,
                                 protein_ree2, 
                                           protein_rg,
                                           protein_rg_by_rg_theta,
                                          protein_rg_by_rg_mean),columns=['seq_name',
                                                                            'Rg2',
                                                                            'Ree2',
                                                                            'Rg',
                                                                            'Rg/Rg_theta',
                                                                         'Rg/Rg_mean']).copy()
    protein_df_original['ratio'] = protein_df_original['Ree2']/protein_df_original['Rg2']
    protein_df_original['RSA'] = t_df_moments['RSA']
    del t_df_moments
    print(f'protein_df_original is READY')
    return 

seq_name_fluctations = pd.read_csv('HPC_computed_fC_values_all.csv').set_index('seq_name_list')
# the bounded_frac_size_shape is only for size-shape (through pyconformap_modified)


#add fP to the property df
seq_name_AFRC['fP'] = [seq.count('P')/len(seq) for seq in seq_name_AFRC.fasta.values]

idrome_prop_flucs = pd.concat([seq_name_AFRC.set_index('seq_name'),
           seq_name_fluctations[['fC_shape_shape',
                                 'fA_shape_shape',
                                 'fC_size_shape',
                                 'fA_size_shape',
                                 'bounded_frac_size_shape']]],
          axis=1).reset_index().rename(columns={'index':'seq_name'}).copy()




RSA_Rs_size_compute_3dplot_from_seq_name_ALL()
protein_df_original.to_csv('entire_IDRome_landscape_size_shape_shape.csv',index=False)