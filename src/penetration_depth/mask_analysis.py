import os
import re
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from scipy import ndimage
import cv2
import pickle
import pandas as pd
import math
import time
import traceback

from penetration_depth.helpers import load_plot_parameters, load_parameter_maps
from penetration_depth import overimposed_img

    
def check_the_annotations(to_clean: list):
    for clean in to_clean:
        incorrect = True
        while incorrect:
            path, filename_mask, _ = clean
            imgs_split = []
            annotation_folder = os.path.join(path, 'annotation')
            for annotation in os.listdir(annotation_folder):
                if filename_mask in annotation and annotation.endswith('.tif'):
                    imgs_split.append(os.path.join(annotation_folder, annotation))
            blobs = get_blobs(imgs_split)
            plt.imshow(blobs[0]) 
            plt.savefig(os.path.join('./', 'tmp', 'blobs_fig.png'))
            cv2.imshow("Blobs", np.asarray(cv2.imread(os.path.join('./', 'tmp', 'blobs_fig.png'))))
            print(clean)
            print("Number of expected blobs: {}, Number of blobs: {}".format(len(imgs_split), blobs[1]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if len(imgs_split) != blobs[1]:
                input("Press enter to re-process...")
            else:
                incorrect = False
                
def load_data(path_data, measurements_types, wavelength, parameters_save):
    data_measurement = {}
    data_to_clean = {}
    for measurements_type in tqdm(measurements_types):
        nothing_to_clean = False
        while not nothing_to_clean:
            data_measurement[measurements_type], data_to_clean[measurements_type] = process_one_measurement(path_data, measurements_type, 
                                                                                                            wavelength, parameters_save, Flag = False)
            if len(data_to_clean[measurements_type]) == 0:
                nothing_to_clean = True
            else:
                check_the_annotations(data_to_clean[measurements_type])
    return data_measurement, data_to_clean

def get_params(path_data: str, measurements: str, wavelength: str):
    """Get parameters from the path of the data."""
    path_folder = os.path.join(path_data, measurements)
    results_path = os.path.join(path_data, 'results')
    try:
        os.mkdir(os.path.join(path_data, 'results'))
    except FileExistsError:
        pass
    
    results_path = os.path.join(results_path, measurements)
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    
    results_path = os.path.join(results_path, wavelength)
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    
    return path_folder, results_path


def check_input_parameters(measurements_type: str, wavelength: str):
    """Check if the input parameters are correct."""
    assert measurements_type == '90_overimposition' or measurements_type == '45_overimposition' or measurements_type == 'splitted' or measurements_type == '0_overimposition' or measurements_type == '100+x'
    
    
def reorganize_data(data):
    data_reorganized = {}

    for idx in range(len(data['retardance'])):
        data_reorganized[idx + 1] = {}

    for parameter, values in data.items():
        for idx, val in values.items():
            data_reorganized[idx][parameter] = val
    return data_reorganized
    
def process_one_measurement(path_data: str, measurements_type: str, wavelength: str, parameters_save, Flag: bool = False):
    """Process one measurement type."""
    path_data, results_path = get_params(path_data, measurements_type, wavelength)
    check_input_parameters(measurements_type, wavelength)

    # get the paths of all folders for the analysis
    paths, filename_mask, _ = find_all_folders(path_data, measurements_type)
    check_annotation(paths, measurements_type, filename_mask)

    # get the data for the ROIs and generate the single figures
    
    data, data_to_clean = get_data(paths, filename_mask, wavelength, Flag = Flag)
    
    save_raw_data(data, results_path, parameters_save, measurements_type)
    return data, data_to_clean

def generate_plots(path_data, data, measurements_type, wavelength, metric: str = 'mean', Flag: bool = False):
    path_data, results_path = get_params(path_data, measurements_type, wavelength)
    paths, _, match_sequence = find_all_folders(path_data, measurements_type)
    generate_histogram(data, results_path, match_sequence, Flag = Flag)
    
    
    
    thicknesses = get_x_thicnkesses(data, measurements_type)
    _, parameters, parameters_std = get_plot_parameters(thicknesses, data, metric)
    parameters, parameters_std = get_df(parameters), get_df(parameters_std)
    save_dfs(parameters, results_path)
    save_dfs(parameters_std, results_path, std = True)
    
    overimposed_img.save_the_imgs(paths, results_path, wavelength, Flag = Flag)
    return results_path
    
def get_match_sequence(measurements_type: str):
    # -----------------------------------------------------------
    # returns the match sequence that should be used for finding all the folders of interest
    # and the filenames of the masks to be used
    # value returned depends on the parameters given as an input
    # -----------------------------------------------------------
    zero_degree = measurements_type == '0_overimposition'
    ninety_degree = measurements_type == '90_overimposition'
    splitted = measurements_type == 'splitted'
    hundred_um = measurements_type == '100+x'
    fortyfive_degree = measurements_type == '45_overimposition'
    
    if zero_degree or ninety_degree or hundred_um or fortyfive_degree:
        filename_mask = 'overimposed'
        if zero_degree:
            match_sequence = '-0-\d+um'
        elif ninety_degree:
            match_sequence = '-90-\d+um'
        elif hundred_um:
            match_sequence = '-100-\d+um'
        elif fortyfive_degree:
            match_sequence = '-45-\d+um'

    elif splitted:
        match_sequence = 'split-\d+um'
        filename_mask = 'splitted'
        
    else:
        raise ValueError
        
    return match_sequence, filename_mask


def find_all_folders(path_data: str, measurements_type: str):
    # -----------------------------------------------------------
    # find all the folders that corresponds to the measurements of interest based on the matching
    # of the folder name to a specific regex sequence - return the list of all the folders of interest
    # and the name of files used for the masks
    # -----------------------------------------------------------
    match_sequence, filename_mask = get_match_sequence(measurements_type)

    paths = []
    # iterate over each result of the results folder
    for folder in os.listdir(path_data):
        if should_add_folder(match_sequence, folder):
            paths.append(os.path.join(path_data, folder))
    return paths, filename_mask, match_sequence


def should_add_folder(match_sequence: str, folder: str):
    # -----------------------------------------------------------
    # searches for the folder matching the match sequence and add it to the list of all folders
    # -----------------------------------------------------------
    
    # if a match can be found with the match sequence
    if (re.findall(match_sequence, folder)):
        return True
    else:
        raise ValueError
    
    
def check_annotation(paths: list, measurements_type: str, filename_mask: str):
    # -----------------------------------------------------------
    # check if all the images have been correctly annotated
    # -----------------------------------------------------------
    measurements_numbers = {'90_overimposition': [1, 4], 'splitted': [8, 4, 3], '0_overimposition': [4, 2], '100+x': [1, 4],
                       '45_overimposition': [4]}
    numbers = measurements_numbers[measurements_type]
    for folder in paths:
        annotated = False
        counter = 0
        files = os.listdir(os.path.join(folder, 'annotation'))
        for f in files:
            if '.tif' in f:
                old_name = os.path.join(folder, 'annotation', f)
                new_name = os.path.join(folder, 'annotation', f.replace(filename_mask, 'ROI_'))
                os.rename(old_name, new_name)
                counter += 1
                    
        for length in numbers:
            if counter == length:
                annotated = True
        if not annotated:
            raise ValueError('The folder {} has not been correctly annotated'.format(folder))
        
        
def get_data(paths: list, filename_mask: str, wavelength: str, Flag: bool = False):
    # -----------------------------------------------------------
    # gather the data for each of the folder, calls the function used to generate the histogram for each file
    # -----------------------------------------------------------
    data_combined = {}
    data_to_curate = []
    if Flag:
        for path in tqdm(paths):
            dat = create_histogram(path, filename_mask, wavelength)
            if type(dat) == tuple:
                data_to_curate.append(dat)
            else: 
                data = reorganize_data(dat)  
                for idx, dat in data.items():
                    data_combined[path + f'_ROI_' + str(idx)] = dat
    else:
        for path in paths:
            try:
                dat = create_histogram(path, filename_mask, wavelength)
                if type(dat) == tuple:
                    data_to_curate.append(dat)
                else: 
                    data = reorganize_data(dat) 
                    for idx, dat in data.items():
                        data_combined[path + f'_ROI_' + str(idx)] = dat
            except FileNotFoundError :
                if wavelength != '550nm' and wavelength != '650nm':
                    pass
                else:
                    traceback.print_exc()
    return data_combined, data_to_curate

def get_blobs(img_split, eroding: bool = True, kernel_size: int = 8):
    imgs_splits = []
    for img in img_split:
        imgs_splits.append(np.array(Image.open(img)))
        
    mask_combined = np.zeros(imgs_splits[0].shape)
    mask_combined_blob = np.zeros(imgs_splits[0].shape)
    
    for idx, img in enumerate(imgs_splits):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(img, kernel) 
        mask_combined += (idx + 1) * (mask != 0)
        mask_combined_blob += mask
        
    blobs = find_blob(mask_combined_blob)
    
    return mask_combined, blobs
                           
def find_blob(mask):
    blur_radius = 0
    threshold = 50
    img = mask
    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(img, blur_radius)
    threshold = 50
    return ndimage.label(imgf > threshold) 
    
def save_raw_data(data, results_path, parameters_save: dict, measurements_type: str):
    match_sequence, _ = get_match_sequence(measurements_type)
    results_path = os.path.join(results_path, 'raw_data')
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    for path, vals in data.items():
        thickness = int(re.findall(match_sequence, path)[0].split('um')[0].split('-')[-1])
        for parameter in parameters_save:
            raw_data_path = os.path.join(results_path, parameter)
            try:
                os.mkdir(raw_data_path)
            except:
                pass
            data_path = os.path.join(raw_data_path, str(thickness))
            try:
                os.mkdir(data_path)
            except FileExistsError:
                pass
            with open(os.path.join(data_path, path.split('\\')[-1] + '.pickle'), 'wb') as handle:
                pickle.dump(vals[parameter][-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    
def create_histogram(path: str, filename_mask: str, wavelength: str):
    # -----------------------------------------------------------
    # extract the data from the ROI(s) for the different parameters and calls the function used to 
    # compute the relevant metrics (mean, stdev and maximum) for each of the parameters. takes into
    # account the presence of multiple ROIs for azimuth and correct the difference in orientation
    # for the two ROIs
    # return the metrics for the parameters in the specified ROIs
    # -----------------------------------------------------------    
  
    imgs_split = []
    annotation_folder = os.path.join(path, 'annotation')
    for annotation in os.listdir(annotation_folder):
        if annotation.endswith('.tif'):
            imgs_split.append(os.path.join(annotation_folder, annotation))
    print(imgs_split)
    mask_combined, blobs = get_blobs(imgs_split)
    try:
        assert blobs[1] == len(imgs_split)
    except:
        return path, filename_mask, wavelength

    MM = np.load(os.path.join(path, 'polarimetry', wavelength, 'MM.npz'))
    ROI_values = get_area_of_interest_values(MM, mask_combined)
    
    return ROI_values


def load_and_verify_parameters(mat, name):
    # -----------------------------------------------------------
    # load the parameter of interest (name) from the (light) mueller matrix (MM) computed in MATLAB (mat or light matrix)
    # check for correct size of the MM
    # -----------------------------------------------------------
    out = mat[name]
    assert len(out) == 388
    assert len(out[0]) == 516
    
    return out

def get_area_of_interest_values(MM, image_array, iq_size = 95):
    # -----------------------------------------------------------
    # extract the data from the ROI of interest and return the values of the parameters in that
    # particular ROI - called by create_histogram for each ROI
    # -----------------------------------------------------------    
    linear_retardance = load_and_verify_parameters(MM, 'linR')
    diattenuation = load_and_verify_parameters(MM, 'totD')
    azimuth = load_and_verify_parameters(MM, 'azimuth')
    depolarization = load_and_verify_parameters(MM, 'totP')

    linear_retardance_dict = defaultdict(list)
    diattenuation_dict = defaultdict(list)
    azimuth_dict = defaultdict(list)
    depolarization_dict = defaultdict(list)

    # select the pixels contained in the ROI(s)
    for x, line in enumerate(image_array):
        for y, pixel in enumerate(line):
            if pixel != 0:
                linear_retardance_dict[pixel].append(linear_retardance[x][y])
                diattenuation_dict[pixel].append(diattenuation[x][y])
                azimuth_dict[pixel].append(azimuth[x][y])
                depolarization_dict[pixel].append(depolarization[x][y])
    
    parameters = load_plot_parameters()
    
    # select the pixels contained in the ROI(s)
    retardance = get_area_of_interest(linear_retardance_dict, parameters['retardance'])
    diattenua = get_area_of_interest(diattenuation_dict, parameters['diattenuation'])
    depol = get_area_of_interest(depolarization_dict, parameters['depolarization'])
    azi = get_area_of_interest(azimuth_dict, parameters['azimuth'], 'azimuth', iq_size = iq_size)
    
    return {'retardance': retardance, 'diattenuation': diattenua, 'azimuth': azi, 'depolarization': depol}

def get_area_of_interest(parameter_dict: list, param: dict, parameter: str = '', iq_size: int = 95):
    # -----------------------------------------------------------
    # retrieves the data (parameter of interest) in the ROI previously defined. computes the mean, standard deviation 
    # and the histogram bin with the maximum value
    # for the azimuth, computes a circular mean and circular standard deviation
    # returns the mean, standard deviation, maximum and a list containing all the data points
    # -----------------------------------------------------------
    
    
    results_dict = {}
    for idx, listed in parameter_dict.items():
        if parameter == 'azimuth':
            azimuth_centered = []
            circstd_lst = scipy.stats.circstd(listed, high = 180)
            for az in listed:
                azimuth_centered.append((az - circstd_lst + 90)%180)
            
            diff = (100-iq_size)/100
            inter_quantile_90 = np.quantile(azimuth_centered, 1-diff) - np.quantile(azimuth_centered, diff)
            mean = inter_quantile_90
            stdev = scipy.stats.circstd(listed, high = 180)
            
        else:
            mean = np.mean(listed)
            stdev = np.std(listed)
        
        bins = np.linspace(param['cbar_min'], param['cbar_max'], num = 100 )# param['n_bins'])
        data = plt.hist(listed, bins = bins)
        plt.close()
        arr = data[0]
        max_idx = np.where(arr == np.amax(arr))[0][0]
        maximum = data[1][max_idx]
        results_dict[idx] = [mean, stdev, maximum, listed]
    
    return results_dict


def generate_histogram(data: dict, results_path: str, match_sequence: str, Flag: bool = False):
    
    figures = []
    if Flag:
        for path, vals in tqdm(data.items(), total = len(data)):
            figures.append(generate_histogram_master(vals, path, results_path, match_sequence))
    else:
        for path, vals in data.items():
            figures.append(generate_histogram_master(vals, path, results_path, match_sequence))
        
        
    
def generate_histogram_master(vals: dict, path: str, results_path: str, match_sequence: str):
    # -----------------------------------------------------------
    # calls generate_histogram for each of the sub ROIs
    # -----------------------------------------------------------
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
    generate_azimuth_histograms(vals, path_save_azimuth)

        
def parameters_histograms(val: dict, path_save: str, max_ = False):
    """
    generate the histogram for the four parameters

    Parameters
    ----------
    MuellerMatrices : dict
        the dictionnary containing the computed Mueller Matrices
    folder : str
        the name of the current processed folder
    max_ : bool
        boolean indicating wether or not the max_ should be printed
    """
    parameters_map = load_parameter_maps()
    
    try:
        parameters_map.pop('M11')
    except:
        pass
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    
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


def get_plot_parameters(thicknesses: dict, data: dict, metric: str = 'mean'):
    # -----------------------------------------------------------
    # return the data that will be used to plot the parameters (combined) - orders the data using the thicknesses
    # as a reference (thickness should be increasing)
    # -----------------------------------------------------------    
    if metric == 'mean':
        idx = 2
    elif metric == 'max':
        idx = 0
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
            parameters[thickness][parameter].append(ROI[idx])
            parameters_std[thickness][parameter].append(ROI[1])
         
    return x, parameters, parameters_std

def get_x_thicnkesses(data: dict, measurements_type: str):
    # -----------------------------------------------------------
    # return the thicknesses indicated in the folder name
    # -----------------------------------------------------------
    match_sequence ,_ = get_match_sequence(measurements_type)
    thicknesses = {}
    paths = list(data.keys())
    for path in paths:
        thicknesses[path] = int(re.findall(match_sequence, path)[0].split('um')[0].split('-')[-1])
    return thicknesses


def get_df(parameters):
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
            
            

def generate_azimuth_histograms(val, path_save_azimuth):
    # -----------------------------------------------------------
    # call generate_azimuth_histogram for each of the measurement for which data was obtained
    # -----------------------------------------------------------
    data = val['azimuth'][-1]
    generate_azimuth_histogram(data, path_save_azimuth)
    
def generate_azimuth_histogram(data, path_save_azimuth):
    # -----------------------------------------------------------
    # generate circular azimuth histogram for the azimuth - saves the histogram as .png and .pdf
    # -----------------------------------------------------------
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
    


def get_min_interval(listed, proportion = 0.90):
    cc = listed
    cc.sort()
    n = int(len(cc)*proportion)
    k = len(cc) - n + 1
    d = []

    for i in np.arange(1, k):
        d.append(cc[i+n-1]-cc[i])
        
    if min(d) > 90:
        return np.nan
    else:
        return min(d)
    
def load_raw_data_azimuth(results_path, data: dict):
    azimuth_results = []
    
    data_processed = []
    for key in data.keys():
        data_processed.append(key.split('\\')[-1] +'.pickle')
        
    for thickness in os.listdir(os.path.join(results_path, 'raw_data', 'azimuth')):
        for file in os.listdir(os.path.join(results_path, 'raw_data', 'azimuth', str(thickness))):
            if file in data_processed:
                with open(os.path.join(results_path, 'raw_data', 'azimuth', str(thickness), file), 'rb') as handle:
                    azimuth = pickle.load(handle)
                    azimuth = (azimuth - np.mean(azimuth) + 90) % 180
                    azimuth_results.append(get_min_interval(azimuth, proportion = 0.80))
    return azimuth_results

def create_output_pickle(data: dict, parameters: list, path_data, measurements_type, wavelength):
    path_data, results_path = get_params(path_data, measurements_type, wavelength)
    data_folders = os.listdir(os.path.join(results_path, 'excel'))
    data_combined = {}
    for data_folder in data_folders:
        df = pd.read_excel(os.path.join(results_path, 'excel', data_folder))
        data_combined[data_folder.split('_data')[0]] = df.drop(['Unnamed: 0'], axis = 1)
        
    all_measurements = []
    all_measurements_std = []
    for measurement, values in data_combined.items():
        if 'std' in measurement:
            all_measurements_std.append(values)
        else:
            all_measurements.append(values)
    all_measurements = pd.concat(all_measurements, axis=1, ignore_index=False).T.drop_duplicates().T
    all_measurements_std = pd.concat(all_measurements_std, axis=1, ignore_index=False).T.drop_duplicates().T
    
    all_measurements['azimuth_iv'] = load_raw_data_azimuth(results_path, data)
    all_measurements['azimuth_sd'] = all_measurements_std['azimuth']
    
    combined_data = {}
    thickness = all_measurements['thickness']
    for param in parameters:
        combined_data[param] = all_measurements[['thickness', param]]
        
    combined_data_per_thickness = {}
    for param in parameters:
        combined_data_per_thickness[param] = defaultdict(list)

    for param in parameters:
        for _, row in combined_data[param].iterrows():
            combined_data_per_thickness[param][row['thickness']].append(row[param])
            
    for param, val in combined_data_per_thickness.items():
        for _, lst in val.items():
            while(len(lst) < 15):
                lst.append(math.nan)
            
    with open(os.path.join(results_path, 'combined_data_thickness.pickle'), 'wb') as handle:
        pickle.dump(combined_data_per_thickness, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return combined_data_per_thickness

    
def get_data_df(measurements_types, paths_data, wavelength, parameters):
    
    data_thickness = {}
    for measurements_type in measurements_types:
        path = os.path.join(paths_data, 'results', measurements_type, wavelength, 'combined_data_thickness.pickle')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        data_thickness[measurements_type] = data

    data_df = {}
    for measurements_type, vals in data_thickness.items():
        data_measurement_type = {}
        for parameter, val in vals.items():
            val = dict(val)
            df = pd.DataFrame({ key: pd.Series(v) for key, v in val.items() }).T
            data_measurement_type[parameter] = df
        data_df[measurements_type] = data_measurement_type
        
    data_parameter = {}
    for parameter in parameters:
        data_param = {}
        for measurements_type in measurements_types:
            data_param[measurements_type] = data_df[measurements_type][parameter]
        data_parameter[parameter] = data_param
    
    return data_parameter

def save_data_prism(measurements_types: list, path_data: str, wavelength: str, parameters: list):
    data_parameter = get_data_df(measurements_types, path_data, wavelength, parameters)
    results_path = os.path.join(path_data, 'results', 'prism_files')
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    results_path = os.path.join(results_path, wavelength)
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    
    for parameter in parameters:
        dfs = []
        for _, df in data_parameter[parameter].items():
            dfs.append(df)
        results_path_excel = os.path.join(results_path, parameter + '_prism.xlsx')
        concat = pd.concat(dfs, axis = 1)
        concat = concat.sort_index()
        concat.to_excel(results_path_excel)
    
