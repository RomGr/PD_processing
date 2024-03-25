import os
import re
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import cv2
import pickle
import pandas as pd
import math
import copy
import traceback
import warnings

from penetration_depth.helpers import load_plot_parameters, load_match_sequence, get_measurement_numbers, load_parameters_save
from penetration_depth import overimposed_img
from penetration_depth.check_annotations import check_the_annotations, get_blobs
        
        
def load_data(path_data: str, measurements_types: list, wavelength: str, parameters_save: list, iq_size: int = 95, Flag: bool = False, 
              CX_overimposed = False, CC_overimposed: bool = False, small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # loads the data for each measurement type, save the raw data for the different ROIs and creates the overlay images
    #
    # Parameters:
    #   path_data (str): path to the data
    #   measurements_types (list): list of the different measurement types
    #   wavelength (str): wavelength of the data
    #   parameters_save (list): list of the parameters to save
    #   iq_size (int): size of the IQ (default: 95)
    #   Flag (bool): flag to display the progress bar (default: False)
    #
    # Returns:
    #   data_measurement (dict): dictionary containing the data for each measurement type
    #   data_to_clean (dict): dictionary containing the data to clean for each measurement type
    ----------------------------------------------------------- """
    data_measurement = {}

    # iterate over the different measurement types
    for measurements_type in (tqdm(measurements_types) if Flag else measurements_types):
        
        nothing_to_clean = False
        
        print()
        print(f"Loading data for {measurements_type}...")
            
        # while there are some annotations to clean
        while not nothing_to_clean:
            
            # process the measurements
            if CX_overimposed or CC_overimposed:
                key = tuple(measurements_type)
            else:
                key = measurements_type
            data_measurement[key], data_to_clean = process_one_measurement(path_data, measurements_type, wavelength, 
                                                                                         parameters_save, Flag = False, iq_size = iq_size,
                                                                                         CX_overimposed = CX_overimposed,
                                                                                         CC_overimposed = CC_overimposed,
                                                                                         small_ROIs = small_ROIs)
            
            # check if there are some annotations to clean
            if len(data_to_clean) == 0:
                nothing_to_clean = True
            else:
                # if yes, clean them
                check_the_annotations(data_to_clean)
        
        print(f"Data loaded for {measurements_type}...")
        print()
    return data_measurement


def process_one_measurement(path_data: str, measurements_type: str, wavelength: str, parameters_save: list, iq_size: int = 95, Flag: bool = False, 
                            metric: str = "median", CX_overimposed: bool = False, CC_overimposed: bool = False, small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # process the data for one measurement type
    #
    # Parameters:
    #   path_data (str): path to the data
    #   measurements_type (str): measurement type
    #   wavelength (str): wavelength of the data
    #   parameters_save (list): list of the parameters to save
    #   iq_size (int): size of the IQ (default: 95)
    #   Flag (bool): flag to display the progress bar (default: False)
    #   metric (str): metric to use to compute the statistics (default: "median")
    #
    # Returns:
    #   data (dict): dictionary containing the data for the measurement type
    #   data_to_clean (dict): dictionary containing the data to clean for the measurement type
    ----------------------------------------------------------- """
    check_input_parameters(measurements_type, wavelength, metric, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
    path_data, results_path = get_params(path_data, measurements_type, wavelength, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)

    # get the paths of all folders for the analysis
    paths, filename_mask, _ = find_all_folders(path_data, measurements_type, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
    check_annotation(paths, measurements_type, filename_mask, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed, 
                     small_ROIs = small_ROIs)

    data, data_to_clean = get_data(paths, filename_mask, wavelength, Flag = Flag, iq_size = iq_size, CX_overimposed = CX_overimposed, 
                                   CC_overimposed = CC_overimposed, small_ROIs = small_ROIs, measurements_type = measurements_type)

    save_raw_data(data, results_path, parameters_save, measurements_type, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
    
    return data, data_to_clean


def check_input_parameters(measurements_type: str, wavelength: str, metric: str, CX_overimposed: bool = False, CC_overimposed: bool = False):
    """ -----------------------------------------------------------
    # check if the input parameters are correct
    #
    # Parameters:
    #   measurements_type (str): measurement type
    #   wavelength (str): wavelength of the data
    #   metric (str): metric to use to compute the statistics
    ----------------------------------------------------------- """
    if CX_overimposed or CC_overimposed:
        assert measurements_type[0] in [r'0_overimposition', r'100+x', r'45_overimposition', r'90_overimposition', r'splitted']
        assert measurements_type[1] in [r'GM', r'WM']
    else:
        assert measurements_type in [r'0_overimposition', r'100+x', r'45_overimposition', r'90_overimposition', r'splitted']
    assert wavelength in [r'450nm', r'500nm', r'550nm', r'600nm', r'650nm', r'700nm']
    assert metric in ["median", "mean", "max"]
    

def get_params(path_data: str, measurements: str, wavelength: str, CX_overimposed: bool = False, CC_overimposed: bool = False):
    """ -----------------------------------------------------------
    # get the path to the data folder and the results folder
    #
    # Parameters:
    #   path_data (str): path to the data
    #   measurements (str): measurement type
    #   wavelength (str): wavelength of the data
    #
    # Returns:
    #   path_folder (str): path to the data folder
    #   results_path (str): path to the results folder
    ----------------------------------------------------------- """
    if CX_overimposed or CC_overimposed:
        path_folder = os.path.join(path_data, measurements[0])
    else:
        path_folder = os.path.join(path_data, measurements)
        
    results_path = os.path.join(path_data, 'results')
    try:
        os.mkdir(os.path.join(path_data, 'results'))
    except FileExistsError:
        pass
    
    if CX_overimposed or CC_overimposed:
        results_path = os.path.join(results_path, measurements[0] + '_' + measurements[1])
    else:
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


def find_all_folders(path_data: str, measurements_type: str, CX_overimposed: bool = False, CC_overimposed: bool = False):
    """ -----------------------------------------------------------
    # find all the folders that corresponds to the measurements of interest based on the matching sequence
    #
    # Parameters:
    #   path_data (str): path to the data
    #   measurements_type (str): measurement type
    #
    # Returns:
    #   paths (list): list of the paths to the folders of interest
    #   filename_mask (str): name of the file used for the masks
    #   match_sequence (str): matching sequence used to find the folders of interest
    ----------------------------------------------------------- """
    filename_mask, match_sequence = load_match_sequence(measurements_type, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)

    paths = []
    # iterate over each result of the results folder
    for folder in os.listdir(path_data):
        if should_add_folder(match_sequence, folder):
            paths.append(os.path.join(path_data, folder))
            
    return paths, filename_mask, match_sequence


def should_add_folder(match_sequence: str, folder: str):
    """ -----------------------------------------------------------
    # check if the folder should be added to the list of folders of interest (i.e. if it contains the matching sequence)
    #
    # Parameters:
    #   match_sequence (str): matching sequence used to find the folders of interest
    #   folder (str): name of the folder to check
    #
    # Returns:
    #   boolean: True if the folder should be added, False otherwise
    ----------------------------------------------------------- """
    # if a match can be found with the match sequence
    if (re.findall(match_sequence, folder)):
        return True
    else:
        print(f"Folder {folder} does not match the sequence {match_sequence}")
        return False


def check_annotation(paths: list, measurements_type: str, filename_mask: str, CX_overimposed: bool = False, CC_overimposed: bool = False,
                     small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # check if all the images have been correctly annotated (i.e. contains the correct number of ROIs, otherwise, raise a warning) and rename the files
    #
    # Parameters:
    #   paths (list): list of the paths to the folders of interest
    #   measurements_type (str): measurement type
    #   filename_mask (str): name of the file used for the masks
    #
    # Returns:
    #   boolean: True if the folder should be added, False otherwise
    ----------------------------------------------------------- """
    if small_ROIs:
        numbers = get_measurement_numbers(measurements_type[0], small_ROIs)
    elif CX_overimposed:
        numbers = get_measurement_numbers(measurements_type[0])
    else:
        numbers = get_measurement_numbers(measurements_type)
    
    # iterate over the paths
    for folder in paths:
        annotated = False
        counter = 0
        if small_ROIs:
            files = os.listdir(os.path.join(folder, 'annotation', 'small_ROIs'))
        else:
            files = os.listdir(os.path.join(folder, 'annotation'))
            
        for f in files:
            if '.tif' in f:
                if not small_ROIs:
                    old_name = os.path.join(folder, 'annotation', f)
                    new_name = os.path.join(folder, 'annotation', f.replace(filename_mask, 'ROI_'))
                    os.rename(old_name, new_name)
                counter += 1
        
        # check if the correct number of ROIs has been found  
        if not small_ROIs:          
            for length in numbers:
                if counter == length:
                    annotated = True
        else:
            if counter > numbers:
                annotated = True
        
        # else raise a warning
        if not small_ROIs: 
            if not annotated:
                warnings.warn('The number of ROIs is not correct for the folder {}'.format(folder))
        else:
            if not annotated:
                warnings.warn('The number of ROIs is not sufficient for the folder {}'.format(folder))
            


def get_data(paths: list, filename_mask: str, wavelength: str, Flag: bool = False, iq_size: int = 95, CX_overimposed: bool = False,
             CC_overimposed: bool = False, measurements_type: str = None, small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # actually load the data, compute the statistics and return the results
    #
    # Parameters:
    #   paths (list): list of the paths to the folders of interest
    #   filename_mask (str): name of the file used for the masks
    #   wavelength (str): wavelength of the data
    #   Flag (bool): if True, display a progress bar
    #   iq_size (int): size of the interquartile range
    #
    # Returns:
    #   data_combined (dict): dictionary containing the results
    #   data_to_curate (list): list of the data that needs to be curated
    ----------------------------------------------------------- """
    data_combined = {}
    data_to_curate = []
    
    # iterate over the paths
    for path in (tqdm(paths) if Flag else paths):
        try:
            # load the data and compute the statistics
            dat = get_statistics(path, filename_mask, wavelength, iq_size = iq_size, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed,
                                 small_ROIs = small_ROIs, measurements_type = measurements_type)
            
            # if the data is a tuple, it means that there was a problem with it
            if type(dat) == tuple:
                data_to_curate.append(dat)
            else: 
                # if the data is a dictionary, it means that it has been correctly loaded, and is now stored in a dictionnary
                data = reorganize_data(dat) 
                for idx, dat in data.items():
                    data_combined[path + f'_ROI_' + str(idx)] = dat
        except FileNotFoundError :
            traceback.print_exc()
            # if the wavelength is not 550nm or 650nm, there could be an issue as the data is not always present
            if wavelength != '550nm' and wavelength != '650nm':
                pass
            else:
                traceback.print_exc()
    return data_combined, data_to_curate


def get_statistics(path: str, filename_mask: str, wavelength: str, iq_size: int = 95,
                   measurements_type: str = None, CX_overimposed: bool = False, CC_overimposed: bool = False,
                   small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # load the data and compute the different statistical descriptors for one measurement folder
    #
    # Parameters:
    #   path (str): path to the folder of interest
    #   filename_mask (str): name of the file used for the masks
    #   wavelength (str): wavelength of the data
    #   iq_size (int): size of the interquartile range
    #
    # Returns:
    #   ROI_values (dict): dictionary containing the results
    ----------------------------------------------------------- """  
    imgs_split = []
    
    # get all the annotations and check if they are not overlapping
    if small_ROIs:
        annotation_folder = os.path.join(path, 'annotation', 'small_ROIs')
    else:
        annotation_folder = os.path.join(path, 'annotation')
    for annotation in os.listdir(annotation_folder):
        if annotation.endswith('.tif'):
            if CX_overimposed:
                if measurements_type[1] in annotation:
                    imgs_split.append(os.path.join(annotation_folder, annotation))
            else:
                imgs_split.append(os.path.join(annotation_folder, annotation))
                

    # try to get the different blobs
    try:
        mask_combined, blobs = get_blobs(imgs_split)
    except:
        traceback.print_exc()
        
    # if the blobs are not None (meaning that there was an issue), check if the number of blobs is the same as the number of annotations
    if blobs != None:
        try:
            assert blobs[1] == len(imgs_split)
        except:
            return path, filename_mask, wavelength

    # load the MM and compute the statistics
    MM = np.load(os.path.join(path, 'polarimetry', wavelength, 'MM.npz'))
    ROI_values = get_area_of_interest_values(MM, mask_combined, iq_size = iq_size)
    return ROI_values


def get_area_of_interest_values(MM: dict, mask_combined: np.array, iq_size: int = 95):
    """ -----------------------------------------------------------
    # get the values of the different parameters for the different ROIs
    #
    # Parameters:
    #   MM (dict): dictionary containing the different parameters
    #   mask_combined (np.array): mask containing the annotations
    #   iq_size (int): size of the interquartile range
    #
    # Returns:
    #   ROI_values (dict): dictionary containing the results
    ----------------------------------------------------------- """    
    # load the parameters
    parameters = load_parameters_save()

    pol_parameters, pol_parameters_dict = load_and_verify_parameters(MM, parameters)

    # select the pixels contained in the ROI(s)
    for x, line in enumerate(mask_combined):
        for y, pixel in enumerate(line):
            if pixel != 0:
                for key, _ in parameters.items():
                    pol_parameters_dict[key][pixel].append(pol_parameters[key][x][y])
    
    plt_parameters = load_plot_parameters()
        
    # select the pixels contained in the ROI(s)
    ROI_values = {}
    for key, _ in parameters.items():
        ROI_values[key] = get_area_of_interest(pol_parameters_dict[key], plt_parameters[key])
    return ROI_values


def load_and_verify_parameters(mat: dict, parameters: dict):
    """ -----------------------------------------------------------
    # load the parameters of interest from the (light) mueller matrix (MM) and check for correct size of the MM
    #
    # Parameters:
    #   mat (dict): dictionary containing the different parameters
    #   parameters (dict): dictionary containing the different parameters and the corresponding index in the MM
    #
    # Returns:
    #   out (np.array): parameter of interest
    ----------------------------------------------------------- """  
    parameters = load_parameters_save()
    param_MM = {}
    param_dict = {}
    for key, value in parameters.items():
        param_MM[key] = mat[value]
        assert len(mat[value]) == 388
        assert len(mat[value][0]) == 516
        param_dict[key] = defaultdict(list)
  
    return param_MM, param_dict


def get_area_of_interest(parameter_dict: dict, param: dict, parameter: str = '', iq_size: int = 95):
    """ -----------------------------------------------------------
    # computes the statistics for the different ROIs for one folder
    #
    # Parameters:
    #   parameter_dict (dict): dict containing the parameters for the different ROIs
    #   param (dict): dictionary containing the parameters for one parameter
    #   parameter (str): name of the parameter (default = ''), used to specify the azimuth
    #   iq_size (int): size of the interquartile range (default = 95)
    #
    # Returns:
    #   results_dict (dict): dictionary containing the results
    ----------------------------------------------------------- """  
    results_dict = {}
    for idx, listed in parameter_dict.items():
        if parameter == 'azimuth':
            azimuth_centered = []
            circstd_lst = scipy.stats.circstd(listed, high = 180)

            for az in listed:
                azimuth_centered.append((az - circstd_lst + 90)%180)
            
            diff = (100-iq_size)/100
            mean = np.abs(np.quantile(azimuth_centered, 1-diff) - np.quantile(azimuth_centered, diff))
            median = mean
            stdev = scipy.stats.circstd(listed, high = 180)
        
        else:
            mean = np.mean(listed)
            median = np.median(listed)
            stdev = np.std(listed)
        
        bins = np.linspace(param['cbar_min'], param['cbar_max'], num = 100 )# param['n_bins'])
        data = plt.hist(listed, bins = bins)
        plt.close()
        arr = data[0]
        max_idx = np.where(arr == np.amax(arr))[0][0]
        maximum = data[1][max_idx]
        results_dict[idx] = [mean, stdev, maximum, median, listed]
    
    return results_dict


def reorganize_data(data: dict):
    """ -----------------------------------------------------------
    # reogranize the data given by the function get_area_of_interest_values
    #
    # Parameters:
    #   data (dict): dict containing the parameters given by the function get_area_of_interest_values
    #
    # Returns:
    #   data_reorganized (dict): dictionary containing the results in a different format
    ----------------------------------------------------------- """  
    data_reorganized = {}

    for idx in range(len(data['retardance'])):
        data_reorganized[idx + 1] = {}

    for parameter, values in data.items():
        for idx, val in values.items():
            data_reorganized[idx][parameter] = val
    return data_reorganized


def save_raw_data(data: dict, results_path: str, parameters_save: dict, measurements_type: str, CX_overimposed: bool = False,
                  CC_overimposed: bool = False):
    """ -----------------------------------------------------------
    # save the raw data for all parameters in the different ROIs
    #
    # Parameters:
    #   data (dict): dict containing the parameters given by the function get_area_of_interest_values
    #   results_path (str): path to the results folder
    #   parameters_save (dict): dictionary containing the parameters to save
    #   measurements_type (str): type of the measurement
    ----------------------------------------------------------- """  
    _, match_sequence = load_match_sequence(measurements_type, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
    results_path = os.path.join(results_path, 'raw_data')
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    
    # iterate over the different ROIs
    for path, vals in data.items():
        
        # find the thickness of the sample
        thickness = int(re.findall(match_sequence, path)[0].split('um')[0].split('-')[-1])
        
        
        # iterate over the different parameters
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























