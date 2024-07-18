from numba import jit, cuda
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
from scipy.stats import probplot,shapiro, sem
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib import cm
from numpy import linspace
#import umap.umap_ as umap
import os

cuda.detect()
 

@jit(nopython=True)
def compute_quantities_using_HPC_numba():
    global testeq_GW
    testeq_GW= pd.read_csv('../segmented_chain_project/reference_GW_chainlen_100_for_RSA.csv')
    
    
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
    
    
    
    def RSA_based_fC(protein_var,protein_name,poly_id,
                           GW_moment_var,every_ith_snap,GW_every_ith_snap,radius_):
        x_total=[]
        y_total=[]
        if poly_id=='protein':
            protein_label=protein_name
            temp_protein=protein_var.copy()
            x_total.append(temp_protein.RSA.values)
            y_total.append(temp_protein.ratio.values)
            x_total=list(chain.from_iterable(x_total))
            y_total=list(chain.from_iterable(y_total))
            poly_var=protein_var.copy()
            protein_pro=poly_var[['ratio','RSA']].iloc[:every_ith_snap,:].copy()
            protein_pro['polymer_id']=np.repeat(protein_label,protein_pro.shape[0])
            del x_total, y_total, temp_protein
        else:
            return print('ERROR')
    
        GW_po=GW_moment_var[['ratio','RSA']][::GW_every_ith_snap].copy()
        GW_po['polymer_id']=np.repeat('GW',GW_po.shape[0])
        
        #calculate mean and stdev values (must keep same mean and stdev values)
        upto_snapshots=1000000
        GW_mean_ratio=np.mean(GW_po['ratio'].values[0:(upto_snapshots+1)])
        GW_std_ratio=np.std(GW_po['ratio'].values[0:(upto_snapshots+1)])
        GW_mean_RSA=np.mean(GW_po['RSA'].values[0:(upto_snapshots+1)])
        GW_std_RSA=np.std(GW_po['RSA'].values[0:(upto_snapshots+1)])
            
        combined_pro_po=pd.concat([GW_po,protein_pro],axis=0,ignore_index=True)
        combined_pro_po['stdd_ratio']=(combined_pro_po['ratio'].values-GW_mean_ratio)/(GW_std_ratio)
        combined_pro_po['stdd_RSA']=(combined_pro_po['RSA'].values-GW_mean_RSA)/(GW_std_RSA)
        po_x=combined_pro_po[combined_pro_po.polymer_id=='GW']['stdd_RSA'].values
        po_y=combined_pro_po[combined_pro_po.polymer_id=='GW']['stdd_ratio'].values
        pro_x=combined_pro_po[combined_pro_po.polymer_id==protein_label]['stdd_RSA'].values
        pro_y=combined_pro_po[combined_pro_po.polymer_id==protein_label]['stdd_ratio'].values
        
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
            #if j%1000000==0:
            #    print(f'{j} GW snapshots completed')
                    
        fC_by_distance=(GW_points.shape[0]-len(GW_not_in_range))/(GW_points.shape[0])
        return fC_by_distance    
    
    def protein_3dplot_against_GW(protein_var,protein_label,second_obj,provided_color):
        
        x_variable= 'RSA'
        y_variable=r'$R_s$'
        global fC_value, grid_protein, grid_GW, fA_value
    
        testeq_GW= pd.read_csv('../segmented_chain_project/reference_GW_chainlen_100_for_RSA.csv')
        
        plt.rcParams["font.weight"] = "regular"
        plt.rcParams["axes.labelweight"] = "regular"
        
        
        x_polmodel_GW=[]
        y_polmodel_GW=[]
        x_total=[]
        y_total=[]
    
        scatter_markers=['d','x',4,'o','v','^','1','8','s','p','P','*','X','D',9]
    
    
        x_polmodel_GW.append(testeq_GW['RSA'].values)
        y_polmodel_GW.append(testeq_GW['ratio'].values)
    
        if second_obj=='protein':        
            temp_protein =  protein_var.copy()
    
            x_total.append(temp_protein.RSA.values)
            y_total.append(temp_protein.ratio.values)
    
            fC_value=RSA_based_fC(temp_protein,
                           protein_label,
                           second_obj,testeq_GW,
                           temp_protein.shape[0],
                           1,0.1)            
            del temp_protein
        else:
            print('ERROR. FIX ERROR')
    
        x_total=list(chain.from_iterable(x_total))
        y_total=list(chain.from_iterable(y_total))
        x_polmodel_GW=list(chain.from_iterable(x_polmodel_GW))
        y_polmodel_GW=list(chain.from_iterable(y_polmodel_GW))
    
        # Set up default x and y limits
        max_x_val=1
        max_y_val=30
        xlims = [0,max_x_val]
        ylims = [0,max_y_val]
        if testeq_GW['RSA'].max()>max_x_val or testeq_GW['RSA'].min()<0:
            return print('ERROR')
        elif max(x_total)>max_x_val or min(x_total)<0:
            return print('ERROR')        
        elif testeq_GW['ratio'].max()>max_y_val or testeq_GW['ratio'].min()<0:
            return print('ERROR')
        elif max(y_total)>max_y_val or min(y_total)<0:
            return print('ERROR')        
    
        # Set up your x and y labels
        xlabel = x_variable
        ylabel = y_variable
    
        # Define the locations for the axes
    
        left, width = 0.12, 0.55
        bottom, height = 0.12, 0.55
        bottom_h = left_h = left+width
        rect_temperature = [left, bottom, width, height] # dimensions of temp plot
        rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
        rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
    
    
        # Set up the size of the figure
        fig = plt.figure(1, figsize=(5.2,5))
    
        # Make the three plots
        axTemperature = plt.axes(rect_temperature) # temperature plot
        axHistx = plt.axes(rect_histx) # x histogram
        axHisty = plt.axes(rect_histy) # y histogram
    
    
        axHistx.axes.get_xaxis().set_visible(False)
        axHisty.axes.get_yaxis().set_visible(False)
    
    
        # Find the min/max of the data
        xmin = min(xlims)
        xmax = max(xlims)
        ymin = min(ylims)
        ymax = max(ylims)
    
        # Remove the inner axes numbers of the histograms
        nullfmt = NullFormatter()
        #
        # Define the number of bins
        nxbins = 100
        nybins = 50
        nbins = nxbins+nybins
    
        xbins = linspace(start = xmin, stop = xmax, num = nxbins)
        ybins = linspace(start = ymin, stop = ymax, num = nybins)
        xcenter = (xbins[0:-1]+xbins[1:])/2.0
        ycenter = (ybins[0:-1]+ybins[1:])/2.0
        aspectratio = (1.0*(xmax - 0))/(1.38*ymax - 0)
    
    
        X = xcenter
        Y = ycenter
    
        axTemperature.scatter(x_polmodel_GW,y_polmodel_GW,
                          marker=scatter_markers[3],s=1,alpha=0.6,color='black',
                           label='GW')
        
        axTemperature.scatter(x_total,
                   y_total,
                   marker=scatter_markers[0],s=1,alpha=0.6,color=provided_color,
                                          label=protein_label)                
    
    
    
        #Plot the axes labels
        axTemperature.set_xlabel(xlabel,fontsize=12,labelpad = 1)
        axTemperature.set_ylabel(ylabel,fontsize=12, labelpad = 1)
    
        #Set up the plot limits
        axTemperature.set_xlim(xlims)
        axTemperature.set_ylim(ylims)
    
        #Set up the histogram bins
        xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
        ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
    
    
        axHistx.hist(x_polmodel_GW, bins=xbins, color = 'black',histtype='step',
                    linewidth = 0.7,density=True)
        axHistx.hist(x_total, bins=xbins, color = provided_color,histtype='step',
                    label=protein_label,linewidth = 0.7,density=True)
    
        axHisty.hist(y_polmodel_GW, bins=np.arange(min(y_total),max(y_total),1), color = 'black',orientation='horizontal',histtype='step',
                    linewidth = 0.7,density=True)
    
        axHisty.hist(y_total, bins=ybins, color = provided_color,orientation='horizontal',histtype='step',
                    label=protein_label,linewidth = 0.7,density=True)    
        
        #original_n_divisions=100
        step_x=0.05#RSA
        step_y=1.5#Shape ratio
        max_value_x=max(xlims)
        n_divisions_x=round((max_value_x-min(xlims))/step_x)
        
        max_value_y=max(ylims)
        n_divisions_y=round((max_value_y-min(ylims))/step_y)
        #gridx = np.linspace(min(x_polmodel_GW), max(x_polmodel_GW), n_divisions_x)
        gridx = np.linspace(min(xlims), max(xlims), n_divisions_x)
        #gridy = np.linspace(min(y_polmodel_GW), max(y_polmodel_GW), n_divisions_y)
        gridy = np.linspace(min(ylims), max(ylims), n_divisions_y)
        
        grid_protein, _, _ = np.histogram2d(x_total, y_total, bins=[gridx, gridy])
        grid_GW, _, _ = np.histogram2d(x_polmodel_GW, y_polmodel_GW, bins=[gridx, gridy])
    
        fA_value = np.count_nonzero(grid_protein)/np.count_nonzero(grid_GW)
        
        axTemperature.text(0.7,0.8,
                           '$f_C$= '+format(fC_value,'0.3f'),
                          transform=axTemperature.transAxes,fontsize=8)
        axTemperature.text(0.7,0.74,
                           '$f_A$= '+format(fA_value,'0.3f'),
                          transform=axTemperature.transAxes,fontsize=8)    
    
        axTemp_legend=axTemperature.legend(fontsize=7,loc='upper right')
    
        frame = axTemp_legend.get_frame()
        frame.set_linewidth(1)
        frame.set_edgecolor('black')
        #axHisty.hist(y, bins=ybins, orientation='horizontal', color = provided_color)
    
    
    
    
    
    
        #Set up the histogram limits
        axHistx.set_xlim( xmin, xmax )
        axHisty.set_ylim( ymin, ymax )
    
        axHistx.set_ylabel('Prob. Density',fontsize=12)
        axHisty.set_xlabel('Prob. Density',fontsize=12)
    
    
        #Cool trick that changes the number of tickmarks for the histogram axes
        axHisty.xaxis.set_major_locator(MaxNLocator(4))
        axHistx.yaxis.set_major_locator(MaxNLocator(3))
    
        for i in ['top', 'left', 'right', 'bottom']:
            axHistx.spines[i].set_linewidth(1)
            axHisty.spines[i].set_linewidth(1)
            axTemperature.spines[i].set_linewidth(1)
    
        axHistx.minorticks_on()
        axHistx.tick_params(axis='y', which='major', labelsize=10, width = 0.8, pad = 2)
        axHistx.tick_params(axis='y', which='minor', width = 0.6)
        axHisty.minorticks_on()
        axHisty.tick_params(axis='x', which='major', labelsize=10, width = 0.8, pad = 2,labelrotation=270)
        axHisty.tick_params(axis='x', which='minor', width = 0.6)
        axTemperature.minorticks_on()
        axTemperature.tick_params(axis='both', which='major', labelsize=10, width = 0.8, pad = 2)
        axTemperature.tick_params(axis='both', which='minor', width = 0.6)    
        plt.setp(axTemperature.get_yticklabels()[-1], visible=False)
        plt.setp(axTemperature.get_xticklabels()[-1], visible=False)
        #fig.subplots_adjust(hspace=0.85,wspace=0.85)
    
        # Save to a File
        filename = 'GW_3d_plot'
        #plt.savefig(f'figures/GW_{protein_label}_3dplot.png', dpi = 1000, bbox_inches='tight')
        return 
    
    def compute_3dplot_from_seq_name(seq_name):
        global protein_df
        example_protein_dir = seq_name_dir_df[seq_name_dir_df.seq_name==seq_name].seq_dir.values[0]
        t = md.load(f'{example_protein_dir}/traj.xtc', top=f'{example_protein_dir}/top.pdb')
        t_df_moments = pd.DataFrame(md.principal_moments(t),columns=['R3','R2','R1']).iloc[10:,:].copy()
        t_df_moments['asphericity']=t_df_moments.R1.values-(0.5*(t_df_moments.R2.values+t_df_moments.R3.values))
        t_df_moments['acylindricity']=t_df_moments.R2.values-t_df_moments.R3.values
        t_df_moments['RSA']=(t_df_moments.asphericity.values**2+(0.75*t_df_moments.acylindricity.values**2))/(t_df_moments.R1.values+t_df_moments.R2.values+t_df_moments.R3.values)**2
        protein_df = t_df_moments[['RSA']].copy()
        del t_df_moments
        protein_df['Rg2'] = np.load(f'{example_protein_dir}/rg.npy')**2
        rg_mean = np.mean(np.load(f'{example_protein_dir}/rg.npy'))
        rg_by_rg_mean = np.load(f'{example_protein_dir}/rg.npy')/rg_mean
        protein_df['Rg_Rg_mean'] = rg_by_rg_mean
        protein_df['Ree2'] = np.load(f'{example_protein_dir}/ete.npy')**2
        protein_df['ratio'] = protein_df['Ree2']/protein_df['Rg2']
        return protein_3dplot_against_GW(protein_df,seq_name,'protein','magenta')
    
    exec(open("pyconformap_modified_for_HPC.py").read())
    def generate_2d_map_size_shape(seq_name):
        global fC_value_size_shape, fA_value_size_shape, bounded_fraction_size_shape
        location_of_files = seq_name_dir_df[seq_name_dir_df.seq_name==seq_name].seq_dir.values[0]    
        rg2_value = np.load(location_of_files+'/rg.npy')**2
        ree2_value = np.load(location_of_files+'/ete.npy')**2
        pd.DataFrame(zip(rg2_value,ree2_value),columns=['Rg2','Ree2']).to_csv(seq_name+'.csv',index=False)
        map_2d_seq_name = PyConforMap(seq_name+'.csv')
        map_2d_seq_name.plot_protein_against_GW(seq_name+'.csv')
        plt.close()
        os.remove(seq_name+'.csv')
        
        fC_value_size_shape = map_2d_seq_name.fC_value
        fA_value_size_shape = map_2d_seq_name.fA_value
        bounded_fraction_size_shape = map_2d_seq_name.bounded_fraction
        return    
    seq_name_list = []
    fC_shape_shape = []
    fA_shape_shape = []
    mean_of_inst_Rs = []
    mean_of_RSA = []
    fC_value_size_shape_list = []
    fA_value_size_shape_list = []
    bounded_fraction_size_shape_list = []
    for provided_seq_name in seq_name_dir_df.seq_name.values:
        seq_name_list.append(provided_seq_name)
        compute_3dplot_from_seq_name(provided_seq_name)
        plt.close()
        fC_shape_shape.append(fC_value)
        fA_shape_shape.append(fA_value)
        del fC_value, fA_value
        mean_of_inst_Rs.append(protein_df.ratio.mean())
        mean_of_RSA.append(protein_df.RSA.mean())
        del protein_df
        generate_2d_map_size_shape(provided_seq_name)
        fC_value_size_shape_list.append(fC_value_size_shape)
        fA_value_size_shape_list.append(fA_value_size_shape)
        bounded_fraction_size_shape_list.append(bounded_fraction_size_shape)
    
    hpc_computed_quantities = pd.DataFrame(zip(seq_name_list,fC_shape_shape,fA_shape_shape,
                    mean_of_inst_Rs,mean_of_RSA,
                    fC_value_size_shape_list,fA_value_size_shape_list,
                    bounded_fraction_size_shape_list),columns=['seq_name_list','fC_shape_shape','fA_shape_shape',
                                                        'mean_inst_Rs','mean_RSA','fC_size_shape',
                                                               'fA_size_shape','bounded_frac_size_shape']).copy()

    return hpc_computed_quantities.to_csv('HPC_computed_fC_values_all_using_numba_numbatest_larsen.csv',index=False)

compute_quantities_using_HPC_numba()