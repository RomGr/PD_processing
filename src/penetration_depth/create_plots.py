from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import pandas as pd

from penetration_depth.mask_analysis import get_params, find_all_folders
from penetration_depth.helpers import load_parameter_maps, load_match_sequence
from penetration_depth.overimposed_img import save_the_imgs


def generate_plots(path_data, data, measurements_types, wavelength, metric: str = 'mean', Flag: bool = False):
    for measurements_type in (tqdm(measurements_types) if Flag else measurements_types):
        path_data_folder, results_path = get_params(path_data, measurements_type, wavelength)
        paths, _, match_sequence = find_all_folders(path_data_folder, measurements_type)
        generate_histogram_master(data[measurements_type], results_path, match_sequence, Flag = Flag)
    
        thicknesses = get_x_thicnkesses(data[measurements_type], measurements_type)
        _, parameters, parameters_std = get_plot_parameters(thicknesses, data[measurements_type], metric)
        parameters, parameters_std = get_df(parameters), get_df(parameters_std)
        save_dfs(parameters, results_path)
        save_dfs(parameters_std, results_path, std = True)
        
        save_the_imgs(paths, results_path, wavelength, Flag = Flag)
    return results_path


def generate_histogram_master(data: dict, results_path: str, match_sequence: str, Flag: bool = False):
    """ -----------------------------------------------------------
    # master function that will generate the histogram for the different parameters
    #
    # Parameters:
    #   data (dict): dictionary containing the data
    #   results_path (str): path to the results folder
    #   match_sequence (str): sequence to match the different folders
    #   Flag (bool): flag to display the progress bar (default: False)
    ----------------------------------------------------------- """
    figures = []
    for path, vals in (tqdm(data.items(), total = len(data)) if Flag else data.items()):
        figures.append(generate_histogram(vals, path, results_path, match_sequence))
        
        
def generate_histogram(vals: dict, path: str, results_path: str, match_sequence: str):
    """ -----------------------------------------------------------
    # master function that will call generate_histogram for each of the sub ROIs
    #
    # Parameters:
    #   vals (dict): dictionary containing the data
    #   path (str): path to the measurement folder
    #   results_path (str): path to the results folder
    #   match_sequence (str): sequence to match the different folders
    ----------------------------------------------------------- """
    results_path_individual = os.path.join(results_path, 'individual')
    try:
        os.mkdir(results_path_individual) 
    except:
        pass
    
    thick = path.split('um')[0].split('-')[-1]
    path_save = os.path.join(results_path_individual, thick)
    try:
        os.mkdir(path_save)
    except FileExistsError:
        pass

    # generate the histograms for the parameters
    path_save = os.path.join(path_save, 'Histogram_{}'.format(path.split('\\')[-1]))
    parameters_histograms(vals, path_save)

    try:
        os.mkdir(os.path.join(results_path, 'azimuth'))
    except FileExistsError:
        pass
    thickness = int(re.findall(match_sequence, path)[0].split('um')[0].split('-')[-1])
    try:
        os.mkdir(os.path.join(results_path, 'azimuth', str(thickness) + 'um'))
    except FileExistsError:
        pass

    path_save_azimuth = os.path.join(results_path, 'azimuth', str(thickness) + 'um', 'Histogram_{}'.format(path.split('\\')[-1] + '.png'))
    generate_azimuth_histogram(vals, path_save_azimuth)
    
    
def parameters_histograms(val: dict, path_save: str, max_ = False):
    """
    generate the histogram for the four parameters

    Parameters
    ----------
    val (dict): dictionary containing the data
    path_save (str): path to save the histogram
    max_ (bool): boolean indicating wether or not the max_ should be printed (default: False)
    """
    parameters_map = load_parameter_maps()
    try:
        parameters_map.pop('M11')
    except:
        pass
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    
    for i, (key, param) in zip(range(0,4), parameters_map.items()):
        row = i%2
        col = i//2
        ax = axes[row, col]
        
        # change the range of the histograms
        if param[2]:
            range_hist = (0, 1)
        elif param[1]:
            range_hist = (0, 180)
        elif param[3]:
            range_hist = (0, 0.20)
        else:
            range_hist = (0, 100)
        
        n_bins = 100
        y, x = np.histogram(
            val[key][-1],
            bins=n_bins,
            density=False,
            range = range_hist)
        
        x_plot = []
        for idx, _ in enumerate(x):
            try: 
                x_plot.append((x[idx] + x[idx + 1]) / 2)
            except:
                assert len(x_plot) == n_bins
        
        # get the mean, max and std
        max_ = x[np.argmax(y)]
        mean = np.nanmean(val[key][-1])
        std = np.nanstd(val[key][-1])
        
        y = y / np.max(y)
        
        # plot the histogram
        ax.plot(x_plot,y, c = 'black', linewidth=3)
        ax.axis(ymin=0,ymax=1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)
    
        if max_:
            ax.text(0.75, 0.85, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}'.format(mean, std, max_), 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                    fontsize=25, fontweight = 'bold')
        else:
            ax.text(0.75, 0.85, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}'.format(mean, std), 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                    fontsize=25, fontweight = 'bold')
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')

        ax.set_title(param[0], fontdict = {'fontsize': 30, 'fontweight': 'bold'})
        ax.set_ylabel('Normalized pixel number', fontdict = {'fontsize': 25, 'fontweight': 'bold'})
        
    # save the figures
    plt.tight_layout()
    plt.savefig(path_save + '.png')
    plt.savefig(path_save + '.pdf')
    plt.close()
    
    
def generate_azimuth_histogram(data: dict, path_save_azimuth: str):
    """ -----------------------------------------------------------
    # generates the circular azimuth of the histograms
    #
    # Parameters:
    #   data (dict): dictionary containing the data
    #   path_save_azimuth (str): path to save the histogram
    ----------------------------------------------------------- """
    data = data['azimuth'][-1]
    max_ = 2* np.pi
    min_ = 0
    a = (max_ - min_)/(max(data) - min(data))
    b = max_ - a * max(data)
    rescaled = a * np.array(data) + b
    
    f = plt.figure(figsize = (15,10))
    ax = f.add_subplot(polar=True)
    
    # actually plot the histogram
    N = 100
    angles = rescaled
    distribution = np.histogram(angles, bins=N, range=(0.0, 2*np.pi))[0]
    theta = (np.arange(N)+0.5)*2*np.pi/N
    width = 2*np.pi / N # Width of bars
    
    # set the ticklabels to correspond to the data!
    ax.bar(theta, distribution, width=width)
    ax.set_rticks([]) # Hides radius tics
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels([r'$0$', r'', r'$45$',r'',
                            r'$90$',r'',r'$135$',r''])
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(labelsize=30)
    ax.set_title('Azimuth histogram', fontsize=30, fontweight='bold')
    
    plt.tight_layout()
    
    # save the histograms
    plt.savefig(path_save_azimuth)
    plt.savefig(path_save_azimuth.replace('png', 'pdf'))
    plt.close()
    
    
def get_x_thicnkesses(data: dict, measurements_type: str):
    """ -----------------------------------------------------------
    # returs the thicknesses indicated in the folder names
    #
    # Parameters:
    #   data (dict): dictionary containing the data
    #   measurements_type (str): type of the measurement
    ----------------------------------------------------------- """
    _, match_sequence = load_match_sequence(measurements_type)
    thicknesses = {}
    paths = list(data.keys())
    for path in paths:
        thicknesses[path] = int(re.findall(match_sequence, path)[0].split('um')[0].split('-')[-1])
    return thicknesses


def get_plot_parameters(thicknesses: dict, data: dict, metric: str = 'mean'):
    """ -----------------------------------------------------------
    # return the data that will be used to plot the parameters (combined) - orders the data using the thicknesses as a reference 
    # (thickness should be increasing)
    #
    # Parameters:
    #   thicknesses (dict): dictionary containing the thicknesses
    #   data (dict): dictionary containing the data
    #   metric (str): metric to use (mean, max, median)
    ----------------------------------------------------------- """
    idx_azimuth = 0
    if metric == 'mean':
        idx = 0
    elif metric == 'max':
        idx = 2
    elif metric == 'median':
        idx = 3
    else:
        raise NotImplementedError
    
    x = []
    parameters = {}
    parameters_std = {}
    
    thicknesses_all = list(set(list(thicknesses.values())))
    
    for thickness in thicknesses_all:
        parameters[thickness] = defaultdict(list)
        parameters_std[thickness] = defaultdict(list)
    
    for path, thickness in thicknesses.items():
        x.append(thickness)
        for parameter, ROI in data[path].items():
            if parameter == 'azimuth':
                parameters[thickness][parameter].append(ROI[idx_azimuth])
            else:
                parameters[thickness][parameter].append(ROI[idx])
            parameters_std[thickness][parameter].append(ROI[1])
         
    return x, parameters, parameters_std


def get_df(parameters: dict):
    """ -----------------------------------------------------------
    # get the data in the form of a dictionnary of dataframes (easier to save later on)
    #
    # Parameters:
    #   parameters (dict): dictionary containing the parameters
    #
    # Returns:
    #   data (dict): dictionary containing the dataframes
    ----------------------------------------------------------- """
    data = defaultdict(list)
    for thickness, vals in parameters.items():
        for param, lst in vals.items():
            for ls in lst:
                data[param].append([thickness, ls])
    for param, val in data.items():
        data[param] = pd.DataFrame(val)
        data[param].columns = ['thickness', param]
        data[param] = data[param].sort_values(by='thickness').reset_index(drop=True)
    return data


def save_dfs(parameters: dict, results_path: str, std: bool = False):
    """ -----------------------------------------------------------
    # save the dataframes in the results folder
    #
    # Parameters:
    #   parameters (dict): dictionary containing the parameters' dataframes
    #   results_path (str): path to the results folder
    #   std (bool): whether to save the add the std or not to the filename
    ----------------------------------------------------------- """
    path_res = os.path.join(results_path, 'excel')
    path_res_csv = os.path.join(results_path, 'csv')
    
    try:
        os.mkdir(path_res)
    except:
        pass
    
    try:
        os.mkdir(path_res_csv)
    except:
        pass
        
    for param, value in parameters.items():
        if std:
            path_xlsx = 'std_data.xlsx'
        else:
            path_xlsx = 'data.xlsx'

        value.to_excel(os.path.join(path_res, param + '_' + path_xlsx))
        value.to_csv(os.path.join(path_res_csv, param + '_' + path_xlsx.replace('xlsx', 'csv')))