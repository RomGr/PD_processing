from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np
import math
from collections import defaultdict

from penetration_depth.mask_analysis import get_params

def create_output_pickle_master(data_measurement: dict, measurements_types: list, parameters: list, path_data: str, wavelength: str, 
                                proportion_azimuth_values: float = 0.80, Flag: bool = False):
    """ -----------------------------------------------------------
    # create_output_pickle_master is the master function calling create_output_pickle for each measurements type
    #
    # Parameters:
    #   data_measurement (dict): dictionary containing the data
    #   measurements_types (list): list of the measurements types to be considered
    #   parameters (list): list of the parameters of interest
    #   path_data (str): path to the data folder
    #   wavelength (str): wavelength of the measurements
    #   proportion_azimuth_values (float): proportion of azimuth values to be considered (default: 0.80)
    #   Flag (bool): flag to display the progress bar (default: False)
    # 
    # Returns:
    #   combined_data_per_thickness (dict): dictionary containing the data for each parameter and for each thickness
    ----------------------------------------------------------- """
    combined_data_per_thickness = {}
    for measurements_type in (tqdm(measurements_types) if Flag else measurements_types):
        combined_data_per_thickness[measurements_type] = create_output_pickle(data_measurement[measurements_type], parameters, path_data, 
                                                                measurements_type, wavelength, proportion_azimuth_values = proportion_azimuth_values)
    return combined_data_per_thickness


def create_output_pickle(data: dict, parameters: list, path_data: str, measurements_type: str, wavelength: str, proportion_azimuth_values: float = 0.80):
    """ -----------------------------------------------------------
    # creates a dictionary containing the data for each parameter and for each thickness
    #
    # Parameters:
    #   data (dict): dictionary containing the data
    #   parameters (list): list of the parameters to be considered
    #   path_data (str): path to the data folder
    #   measurements_type (str): type of the measurements
    #   wavelength (str): wavelength of the measurements
    #   proportion_azimuth_values (float): proportion of azimuth values to be considered (default: 0.80)
    # 
    # Returns:
    #   combined_data_per_thickness (dict): dictionary containing the data for each parameter and for each thickness
    ----------------------------------------------------------- """
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
    
    # save the different metrics used for the azimuth
    all_measurements['azimuth_pr'] = load_raw_data_azimuth(results_path, data, proportion_azimuth_values)
    all_measurements['azimuth_iq'] = all_measurements['azimuth']
    all_measurements['azimuth_sd'] = all_measurements_std['azimuth']
    
    combined_data = {}
    for param in parameters:
        combined_data[param] = all_measurements[['thickness', param]]
        
    combined_data_per_thickness = {}
    for param in parameters:
        combined_data_per_thickness[param] = defaultdict(list)

    for param in parameters:
        for _, row in combined_data[param].iterrows():
            combined_data_per_thickness[param][row['thickness']].append(row[param])
            
    # create the excel structure for the prism data file
    for param, val in combined_data_per_thickness.items():
        for _, lst in val.items():
            while(len(lst) < 15):
                lst.append(math.nan)
            
    with open(os.path.join(results_path, 'combined_data_thickness.pickle'), 'wb') as handle:
        pickle.dump(combined_data_per_thickness, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return combined_data_per_thickness


def load_raw_data_azimuth(results_path: str, data: dict, proportion_azimuth_values: float = 0.80):
    """ -----------------------------------------------------------
    # load_raw_data_azimuth allows to load the azimuth data and get the minimal interval containing proportion_azimuth_values of the data
    #
    # Parameters:
    #   results_path (str): path to the results folder
    #   data (dict): dictionary containing the data
    #   proportion_azimuth_values (float): proportion of the data to be contained in the interval (default = 0.80)
    # 
    # Returns:
    #   azimuth_results (list): list of the minimal intervals containing proportion_azimuth_values of the data
    ----------------------------------------------------------- """
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
                    azimuth_results.append(get_min_interval(azimuth, proportion = proportion_azimuth_values))
    return azimuth_results


def get_min_interval(data: list, proportion: float = 0.90):
    """ -----------------------------------------------------------
    # gets the smallest interval containing a proportion of the data
    #
    # Parameters:
    #   data (list): list of the data
    #   proportion (float): proportion of the data to be contained in the interval (default = 0.90)
    # 
    # Returns:
    #   min(d) (float): smallest interval containing a proportion of the data
    ----------------------------------------------------------- """
    cc = data
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