import matplotlib.pyplot as plt
import numpy as np
import my_matplotlib_style as ms
import colorsys
from corner import corner
import seaborn as sns
import matplotlib as mpl


def plot_residuals(pred, data, range=None, variable_names=['pT', 'eta', 'phi', 'E'], bins=1000, save=None, title=None):
    alph = 0.8
    residuals = (pred - data) / data
    for kk in np.arange(4):
        plt.figure()
        n_hist_pred, bin_edges, _ = plt.hist(residuals[:, kk], label='Residuals', alpha=alph, bins=bins, range=range)
        if title is None:
            plt.suptitle('Residuals of %s' % variable_names[kk])
        else:
            plt.suptitle(title)
        plt.xlabel(r'$(%s_{recon} - %s_{true}) / %s_{true}$' % (variable_names[kk], variable_names[kk], variable_names[kk]))
        plt.ylabel('Number of events')
        ms.sciy()
        if save is not None:
            plt.savefig(save + '_%s' % variable_names[kk])
            
def plot_histograms(pred, data, bins, same_bin_edges=True, colors=['orange', 'c'], variable_list=[r'$p_T$', r'$\eta$', r'$\phi$', r'$E$'], variable_names=['pT', 'eta', 'phi', 'E'], unit_list=['[GeV]', '[rad]', '[rad]', '[GeV]'], title=None):
    alph = 0.8
    n_bins = bins
    for kk in np.arange(4):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        if same_bin_edges:
            n_bins_2 = bin_edges
        else:
            n_bins_2 = bins
        n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=n_bins_2)
        if title is None:
            plt.suptitle(variable_names[kk])
        else:
            plt.suptitle(title)
        plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        plt.ylabel('Number of events')
        ms.sciy()
        plt.legend()  


    
def corr_matrix(input_dim,latent_dim,save_folder,pred_df,data_df,variable_names,corner_groups, save=False):
    residuals = pred_df - data_df
    resid_deviation = residuals.std()
    resid_mean = residuals.mean()
   
    for var in variable_names:
        residuals[var] = residuals[var] / data_df[var]
    
    for i_group, group in enumerate(corner_groups):
        group_df = residuals[group]
        corr = group_df.corr()

        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        cmap = sns.diverging_palette(10,220, as_cmap=True)
        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        mpl.rc_file('matplot_rc_params')
        label_kwargs = {'fontsize': 20, 'rotation': -15, 'ha': 'left'}
        title_kwargs = {"fontsize": 9}
        mpl.rcParams['lines.linewidth'] = 1
        mpl.rcParams['xtick.labelsize'] =12
        mpl.rcParams['ytick.labelsize'] = 12

        group_arr = group_df.values
        qs = np.quantile(group_arr, q=[.0025, .9925], axis = 0)
        ndim = qs.shape[1]
        ranges = [tuple(qs[:,kk]) for kk in np.arange(ndim)]
        figure = corner(group_arr, range=ranges, plot_density=True, plot_contours=True, no_fill_contours=False, #range=[range for i in np.arange(ndim)],
                    bins=50, labels=group, label_kwargs=label_kwargs, #truths=[0 for kk in np.arange(qs.shape[1])],
                    show_titles=True, title_kwargs=title_kwargs, quantiles=(0.16, 0.84),
                    # levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.2e')
                    levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.1e')
        axes = np.array(figure.axes).reshape((ndim,ndim))
        linecol = 'r'
        linstyl = 'dashed'

        for xi in range(ndim):
            ax = axes[0,xi]
            ax.xaxis.set_label_coords(.5,-.8)

        for yi in range(ndim):
            ax = axes[yi, 0]
            ax.yaxis.set_label_coords(-.4, .5)
            ax.set_ylabel(ax.get_ylabel(), rotation=80, ha='right')
        
        
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.set_facecolor(color=mappable.to_rgba(corr.values[yi, xi]))
        #This is how I add extra data to the hists
        for i in range(ndim):
            ax = axes[i,i]
            ax.text(0,0.8, "std: %f" % resid_deviation[i], transform=ax.transAxes, fontsize=7)
            ax.text(0,0.7,"mean: %f" % resid_mean[i], transform=ax.transAxes, fontsize=7)
        
        cax = figure.add_axes([.87, .4, .04, 0.55])
        cbar = plt.colorbar(mappable, cax=cax, format='%.1f', ticks=np.arange(-1., 1.1, 0.2))
        cbar.ax.set_ylabel('Correlation', fontsize=20)

        
        if i_group == 6:
            plt.subplots_adjust(left=0.13, bottom=0.21, right=.82)
        else:
            plt.subplots_adjust(left=0.13, bottom=0.20, right=.83)
        if save:
            fig_name = 'slide_corner_%d_group%d' % (latent_dim, i_group)
            plt.savefig(save_folder + fig_name)
  
